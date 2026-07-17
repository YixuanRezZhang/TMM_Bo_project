import ray, os, logging
import numpy as np
from math import lgamma, sqrt, pi
import os, pickle, torch
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut
from sklearn.linear_model import RidgeCV, LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.ensemble import StackingRegressor, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from src.surrogate_model import SurrogateModel, hyperparameter_optimization
from src.model_uncertainty import predict_mean_and_internal_variance
from scipy.stats import iqr, mode
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN, KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors

LEIDEN_IMPORT_ERROR = None
HDBSCAN_IMPORT_ERROR = None
try:
    import faiss
    import igraph as ig
    import leidenalg
    from scipy.sparse import coo_matrix
except ImportError as exc:
    faiss = None
    ig = None
    leidenalg = None
    coo_matrix = None
    LEIDEN_IMPORT_ERROR = exc

try:
    import hdbscan
except ImportError as exc:
    hdbscan = None
    HDBSCAN_IMPORT_ERROR = exc

if torch.cuda.is_available():
    device = torch.device('cuda')
    import cuml, cupy
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.cluster import KMeans as cuKMeans
    from cuml.neighbors import NearestNeighbors as cuKNN
else:
    device = torch.device('cpu')


def _require_leiden_dependencies():
    if any(dep is None for dep in (faiss, ig, leidenalg, coo_matrix)):
        raise ImportError(
            "Leiden clustering requires faiss, igraph, leidenalg, and scipy."
        ) from LEIDEN_IMPORT_ERROR


def _require_hdbscan_dependency():
    if hdbscan is None:
        raise ImportError(
            "HDBSCAN clustering on CPU requires the optional `hdbscan` package."
        ) from HDBSCAN_IMPORT_ERROR


def _prepare_clustering_input(X):
    """Return a finite, contiguous float32 matrix plus degeneracy diagnostics."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Clustering input must be 2-D, got shape {X.shape}.")

    n_rows, n_features = X.shape
    if n_features == 0:
        raise ValueError("Clustering input must contain at least one feature.")

    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Clustering input must be numeric.") from exc

    nonfinite_mask = ~np.isfinite(X)
    nonfinite_values = int(nonfinite_mask.sum())
    if nonfinite_values:
        X = X.copy()
        for column_idx in np.flatnonzero(nonfinite_mask.any(axis=0)):
            finite_values = X[np.isfinite(X[:, column_idx]), column_idx]
            fill_value = float(np.median(finite_values)) if finite_values.size else 0.0
            X[nonfinite_mask[:, column_idx], column_idx] = fill_value

    # Constant dimensions add no clustering information and can make GPU distance
    # kernels take degenerate paths, especially for very small bootstrap samples.
    if n_rows:
        informative_columns = np.ptp(X, axis=0) > 0.0
    else:
        informative_columns = np.ones(n_features, dtype=bool)
    constant_columns = int(n_features - informative_columns.sum())
    all_features_constant = bool(n_rows and not informative_columns.any())
    if informative_columns.any():
        X = X[:, informative_columns]
    else:
        # Keep the matrix structurally valid. The caller will short-circuit this
        # no-information case rather than invoking a clustering implementation.
        X = np.zeros((n_rows, 1), dtype=np.float64)

    X = np.ascontiguousarray(X, dtype=np.float32)
    unique_rows = int(np.unique(X, axis=0).shape[0]) if n_rows else 0
    diagnostics = {
        'n_rows': n_rows,
        'original_features': n_features,
        'kept_features': X.shape[1],
        'nonfinite_values': nonfinite_values,
        'constant_columns': constant_columns,
        'unique_rows': unique_rows,
        'duplicate_rows': n_rows - unique_rows,
        'all_features_constant': all_features_constant,
    }
    return X, diagnostics


def _norm_pdf(z):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)

def _norm_cdf(z):
    # Use the common GELU-style approximation to Phi(z); it is accurate enough for CRPS.
    # Phi(z) ≈ 0.5 * (1 + tanh( √(2/π) * (z + 0.044715 z^3) ))
    a = 0.044715
    t = np.sqrt(2.0 / np.pi) * (z + a * (z ** 3))
    return 0.5 * (1.0 + np.tanh(t))

def _gaussian_logpdf(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    return -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma) + z * z)

def _gaussian_crps(y, mu, sigma):
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    phi = _norm_pdf(z)
    Phi = _norm_cdf(z)
    # Closed form: sigma * (z * (2*Phi - 1) + 2*phi - 1/sqrt(pi)).
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))

def _student_t_logpdf(res, s, nu):
    # res = y - mu；s>0；nu>2
    s = np.maximum(s, 1e-12)
    const = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * np.log(np.pi * nu)
    return const - np.log(s) - 0.5 * (nu + 1.0) * np.log1p((res * res) / (nu * s * s))

def _sample_student_t(m, n_samples, nu, s, mu, rng=None):
    """Sample values of the form mu + s * T_nu; s and mu are one-dimensional arrays of length m."""
    rng = rng or np.random.default_rng()
    Z = rng.standard_normal((m, n_samples))         # N(0,1)
    U = rng.chisquare(df=nu, size=(m, n_samples))   # χ²_ν
    T = Z / np.sqrt(U / nu)                         # Student-t
    return mu[:, None] + s[:, None] * T

def _student_t_crps_mc(y, mu, s, nu, n_samples=256, rng=None):
    """Unbiased Monte Carlo CRPS estimate: E|X - y| - 0.5 * E|X - X_prime|"""
    S1 = _sample_student_t(len(y), n_samples, nu, s, mu, rng)
    S2 = _sample_student_t(len(y), n_samples, nu, s, mu, rng)
    term1 = np.mean(np.abs(S1 - y[:, None]), axis=1)
    term2 = 0.5 * np.mean(np.abs(S1 - S2), axis=1)
    return term1 - term2

def _get_pred_std_if_available(model, X):
    """
    Return predictive standard deviation when the model supports it; otherwise return None.
    Priority: predict_std -> predict(return_std=True) -> predict(return_cov=True)
    -> bagging/forest member variance -> None.
    Also tries the wrapped estimator inside SurrogateModel when present.
    """
    import numpy as np, inspect

    # 1) Direct predict_std support.
    if hasattr(model, "predict_std"):
        try:
            std = model.predict_std(X)
            return np.asarray(std).reshape(-1)
        except Exception:
            pass

    # 2) scikit-learn-style return_std / return_cov support.
    if hasattr(model, "predict"):
        try:
            sig = inspect.signature(model.predict)
            if "return_std" in sig.parameters:
                _, std = model.predict(X, return_std=True)
                return np.asarray(std).reshape(-1)
            if "return_cov" in sig.parameters:
                _, cov = model.predict(X, return_cov=True)
                cov = np.asarray(cov)
                if cov.ndim == 2:
                    std = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
                    return std.reshape(-1)
        except Exception:
            pass

    # 3) Internal variance for GP/forest/bagging models. Boosting stages are not treated as samples.
    try:
        _, variance = predict_mean_and_internal_variance(model, X)
        if variance is not None:
            return np.sqrt(np.maximum(np.asarray(variance, dtype=float), 0.0)).reshape(-1)
    except Exception:
        pass

    # 4) If this is a wrapper, try the underlying estimator.
    for attr in ("model", "estimator", "regressor", "base_model"):
        if hasattr(model, attr):
            try:
                return _get_pred_std_if_available(getattr(model, attr), X)
            except Exception:
                pass

    # 5) Give up and let OOB residual estimation handle uncertainty.
    return None

def _choose_student_t_nu_by_grid(res, sigma, candidates=(2.5,3,4,5,7,10,15,30,1000.0)):
    """Choose nu by maximizing ELPD on a grid given OOB residuals and homoscedastic or heteroscedastic sigma."""
    best_nu, best_ll = None, -np.inf
    for nu in candidates:
        # Map target standard deviation sigma to Student-t scale s so that Var = sigma^2.
        # For t_nu(mu, s): Var = s^2 * nu/(nu-2), so s = sigma * sqrt((nu-2)/nu).
        c = np.sqrt((nu - 2.0) / nu) if nu < 1e6 else 1.0
        s = np.maximum(sigma * c, 1e-12)
        ll = np.sum(_student_t_logpdf(res, s, nu))
        if ll > best_ll:
            best_ll, best_nu = ll, nu
    return best_nu


class AbstractSurrogateModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_name, models):
        self.model_name = model_name
        self.models = models

    @ray.remote
    def model_predict(self, X, model):
        if self.model_name == 'GP_gpu':
            Gmodel_res = model.model.posterior(torch.tensor(X, dtype=torch.float32))
            Gmean = Gmodel_res.mean.detach().cpu().numpy().reshape(-1)
            return Gmean
        elif self.model_name in ['KAN', 'FastKAN']:
            return model.model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy().flatten()
        else:
            return model.predict(X)

    def fit(self, X, y):
        pass  # Already fitted; no additional fit is needed.

    def predict(self, X):
        predictions = ray.get([self.model_predict.remote(self, X, model) for model in self.models])
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)

    def predict_proba(self, X):
        if hasattr(self.models[0], "predict_proba"):
            probas = np.array([model.predict_proba(X) for model in self.models])
            return np.mean(probas, axis=0)
        else:
            predictions = ray.get([self.model_predict.remote(self, X, model) for model in self.models])
            predictions = np.array(predictions)
            positive_proba = np.clip(np.mean(predictions, axis=0).reshape(-1), 0.0, 1.0)
            return np.column_stack((1.0 - positive_proba, positive_proba))

@ray.remote
def _fit_and_score_cpu(model_name, params, X_for_selection):
    """
    A Ray remote function to fit and score a single CPU clustering configuration.
    """
    labels = None
    clusterer = None
    
    if model_name == 'hdbscan':
        _require_hdbscan_dependency()
        clusterer = hdbscan.HDBSCAN(**params, core_dist_n_jobs=-1, approx_min_span_tree=True)
        X = np.ascontiguousarray(X_for_selection)
        if not X.flags.writeable:
            X = X.copy()
        labels = clusterer.fit_predict(X)
                
    elif model_name == 'gmm':
        clusterer = GaussianMixture(**params, covariance_type='full', n_init=1)
        X = np.ascontiguousarray(X_for_selection)
        if not X.flags.writeable:
            X = X.copy()
        labels = clusterer.fit_predict(X)

    elif model_name == 'kmeans':
        clusterer = KMeans(**params, n_init=10)
        X = np.ascontiguousarray(X_for_selection)
        if not X.flags.writeable:
            X = X.copy()
        labels = clusterer.fit_predict(X)

    elif model_name == 'leiden':
        _require_leiden_dependencies()
        try:
            k = int(params.get('k_neighbors', 15))
            resolution = params.get('resolution_parameter', 1.0)
            X_faiss = np.ascontiguousarray(X_for_selection, dtype=np.float32)
            index = faiss.IndexFlatL2(X_faiss.shape[1])
            index.add(X_faiss)
            D, I = index.search(X_faiss, k + 1)
            
            row, col = np.arange(X_faiss.shape[0]).repeat(k), I[:, 1:].flatten()
            dist = D[:, 1:].flatten()
            mask = dist > 0
            row, col, dist = row[mask], col[mask], dist[mask]
            weight = 1.0 / dist
            
            A_sparse = coo_matrix((weight, (row, col)), shape=(X_faiss.shape[0], X_faiss.shape[0]))
            sources, targets = A_sparse.nonzero()
            g = ig.Graph(edges=list(zip(sources, targets)), directed=False)
            g.es['weight'] = A_sparse.data
            
            partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution)
            labels = np.array(partition.membership)

        except Exception as e:
            logging.warning("Leiden clustering failed: %s", e)
            labels = np.zeros(X_for_selection.shape[0])

    if labels is None:
        return -2.0, model_name, params

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    
    if n_clusters < 2:
        score = -1.0
    else:
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            score = -1.0
        else:
            score = silhouette_score(X_for_selection[valid_mask], labels[valid_mask])
            
    return score, model_name, params

class AdaptiveClusterer:
    """
    Finds the best clustering configuration from a set of candidates using a 
    scalable 'Compete-and-Select' approach on a data subsample.
    """
    def __init__(
        self,
        batch_size=None,
        sample_threshold=20000,
        gpu_available=False,
        use_gpu_hdbscan=False,
        gpu_hdbscan_min_rows=256,
    ):
        self.sample_threshold = sample_threshold*4 if gpu_available else sample_threshold
        self.sample_size = sample_threshold*2 if gpu_available else sample_threshold
        self.gpu_available = gpu_available
        # cuML HDBSCAN may abort the whole Python process from C++/CUDA, so a
        # Python try/except cannot provide a reliable fallback. Keep it opt-in;
        # sklearn HDBSCAN is the safe default while other candidates may use GPU.
        self.use_gpu_hdbscan = bool(use_gpu_hdbscan and gpu_available)
        self.gpu_hdbscan_min_rows = int(gpu_hdbscan_min_rows)
        
        # Define candidate models and their parameter grids to test
        if self.gpu_available:
            self.candidate_configs = {
                'hdbscan': [
                    {'min_cluster_size': 5, 'min_samples': 3},
                    {'min_cluster_size': 10, 'min_samples': 6},
                    {'min_cluster_size': 20, 'min_samples': 12},
                    # {'min_cluster_size': 40, 'min_samples': 24},
                ],
                'kmeans': [
                    {'n_clusters': max(3, int(batch_size/2)) if batch_size is not None else 3},
                    {'n_clusters': max(6, int(batch_size)) if batch_size is not None else 6},
                    {'n_clusters': max(9, int(batch_size*1.5)) if batch_size is not None else 9},
                    {'n_clusters': max(12, int(batch_size*2)) if batch_size is not None else 12},
                ],
                # 'leiden': [
                #     {'k_neighbors': 5, 'resolution_parameter': 0.5},
                #     {'k_neighbors': 10, 'resolution_parameter': 1.0},
                #     {'k_neighbors': 15, 'resolution_parameter': 1.5},
                # ]
            }
        else:
            self.candidate_configs = {
                'kmeans': [
                    {'n_clusters': max(3, int(batch_size/2)) if batch_size is not None else 3},
                    {'n_clusters': max(6, int(batch_size)) if batch_size is not None else 6},
                    {'n_clusters': max(9, int(batch_size*1.5)) if batch_size is not None else 9},
                    {'n_clusters': max(12, int(batch_size*2)) if batch_size is not None else 12},
                ],
                'gmm': [
                    {'n_components': max(3, int(batch_size/2)) if batch_size is not None else 3},
                    {'n_components': max(6, int(batch_size)) if batch_size is not None else 6},
                    {'n_components': max(9, int(batch_size*1.5)) if batch_size is not None else 9},
                    {'n_components': max(12, int(batch_size*2)) if batch_size is not None else 12},
                ],
                # 'leiden': [
                #     {'k_neighbors': 5, 'resolution_parameter': 0.5},
                #     {'k_neighbors': 10, 'resolution_parameter': 1.0},
                #     {'k_neighbors': 15, 'resolution_parameter': 1.5},
                # ]
            }
            if hdbscan is not None:
                self.candidate_configs['hdbscan'] = [
                    {'min_cluster_size': 5, 'min_samples': 3},
                    {'min_cluster_size': 10, 'min_samples': 6},
                    {'min_cluster_size': 20, 'min_samples': 12},
                    # {'min_cluster_size': 40, 'min_samples': 24},
                ]

    def _get_validation_score(self, X, labels):
        """Calculates a validation score for a given clustering."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        if n_clusters < 2: return -1.0
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2: return -1.0
        return silhouette_score(X[valid_mask], labels[valid_mask])

    def _predict_on_full_dataset(self, X_full, clusterer_on_sample, best_model_info, X_sample):
        """Applies a trained clusterer to a large dataset by predicting in chunks."""
        n_samples = X_full.shape[0]
        final_labels = np.full(n_samples, -2, dtype=np.int32)
        chunk_size = 100000

        model_name = best_model_info['name']
        backend = best_model_info.get('backend')
        logging.info(
            "Stage 2: Predicting on %s samples for winning model '%s'",
            n_samples,
            model_name,
        )

        if model_name == 'hdbscan' and backend == 'cpu_sklearn':
            sample_labels = np.asarray(clusterer_on_sample.labels_, dtype=np.int32)
            if X_full.shape == X_sample.shape and np.array_equal(X_full, X_sample):
                logging.info("Reusing sklearn HDBSCAN labels fitted on the full dataset.")
                return sample_labels.copy()

            n_neighbors = min(10, len(X_sample))
            logging.info(
                "sklearn HDBSCAN winner: using CPU k-NN label propagation with k=%s.",
                n_neighbors,
            )
            knn_index = NearestNeighbors(n_neighbors=n_neighbors).fit(X_sample)
            for i in range(0, n_samples, chunk_size):
                start, end = i, min(i + chunk_size, n_samples)
                neighbor_indices = knn_index.kneighbors(
                    X_full[start:end], return_distance=False
                )
                neighbor_labels = sample_labels[neighbor_indices]
                final_labels[start:end] = mode(
                    neighbor_labels, axis=1, keepdims=False
                ).mode

        elif self.gpu_available:
            # --- GPU Prediction Logic ---
            
            # For models without a .predict() method, we use k-NN for label propagation.
            if model_name == 'leiden' or model_name == 'hdbscan':
                if model_name == 'leiden':
                    _require_leiden_dependencies()
                logging.info("%s winner: Initializing cuML k-NN for label propagation.", model_name.upper())
                knn_index = cuKNN(n_neighbors=min(10, n_samples-1))
                knn_index.fit(X_sample)
                
                # Get the labels from the subsample training
                if model_name == 'leiden':
                    subsample_labels = cupy.asarray(clusterer_on_sample.membership)
                else: # HDBSCAN
                    subsample_labels = clusterer_on_sample.labels_

            for i in range(0, n_samples, chunk_size):
                start, end = i, min(i + chunk_size, n_samples)
                X_chunk_gpu = cupy.asarray(X_full[start:end])
                chunk_labels_gpu = None

                if model_name == 'kmeans':
                    chunk_labels_gpu = clusterer_on_sample.predict(X_chunk_gpu)
                # ✅ CORRECTED LOGIC: Use k-NN for both HDBSCAN and Leiden on GPU
                elif model_name == 'hdbscan' or model_name == 'leiden':
                    _, I_gpu = knn_index.kneighbors(X_chunk_gpu)
                    neighbor_labels = subsample_labels[I_gpu]
                    # Mode calculation is faster on CPU for now
                    chunk_labels_gpu = cupy.asarray(mode(cupy.asnumpy(neighbor_labels), axis=1, keepdims=False)[0])
                
                final_labels[start:end] = cupy.asnumpy(chunk_labels_gpu)

        else:
            # --- CPU Prediction Logic (remains the same) ---
            if model_name == 'leiden':
                _require_leiden_dependencies()
                logging.info("Leiden winner: Initializing Faiss index for k-NN label propagation.")
                faiss_index = faiss.IndexFlatL2(X_sample.shape[1])
                faiss_index.add(np.ascontiguousarray(X_sample, dtype=np.float32))
                subsample_labels = np.array(clusterer_on_sample.membership)

            for i in range(0, n_samples, chunk_size):
                start, end = i, min(i + chunk_size, n_samples)
                X_chunk = X_full[start:end]
                
                if model_name == 'gmm':
                    chunk_labels = clusterer_on_sample.predict(X_chunk)
                elif model_name == 'kmeans':
                    chunk_labels = clusterer_on_sample.predict(X_chunk)
                elif model_name == 'hdbscan':
                    _require_hdbscan_dependency()
                    # Use the dedicated function for predicting new points on CPU
                    chunk_labels, _ = hdbscan.approximate_predict(clusterer_on_sample, X_chunk)
                elif model_name == 'leiden':
                    _, I = faiss_index.search(np.ascontiguousarray(X_chunk, dtype=np.float32), 10)
                    neighbor_labels = subsample_labels[I]
                    chunk_labels = mode(neighbor_labels, axis=1, keepdims=False)[0]

                final_labels[start:end] = chunk_labels

        logging.info("Prediction complete.")
        return final_labels

    def find_best_clustering(self, X):
        """
        Identifies the best clustering algorithm and parameters for the data X.

        Returns:
            final_labels (np.array): The labels for the full dataset X.
            best_model_info (dict): Details of the winning model.
        """
        X, diagnostics = _prepare_clustering_input(X)
        logging.info(
            "Clustering input: rows=%s, unique=%s, duplicate=%s, "
            "features=%s->%s, repaired_nonfinite=%s.",
            diagnostics['n_rows'],
            diagnostics['unique_rows'],
            diagnostics['duplicate_rows'],
            diagnostics['original_features'],
            diagnostics['kept_features'],
            diagnostics['nonfinite_values'],
        )
        if diagnostics['n_rows'] < 2 or diagnostics['unique_rows'] < 2:
            logging.warning(
                "Clustering input has fewer than two unique rows; returning all-noise labels."
            )
            labels = np.full(diagnostics['n_rows'], -1, dtype=np.int32)
            return labels, {'name': 'degenerate', 'params': {}, 'backend': 'none'}

        # --- Stage 1: Compete on a subsample if the dataset is large ---
        if X.shape[0] > self.sample_threshold:
            logging.info(
                "Dataset is large (%s samples). Running selection on a subsample of %s.",
                X.shape[0],
                self.sample_size,
            )
            X_for_selection = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]
        else:
            X_for_selection = X

        best_score = -np.inf
        best_model_name, best_params, clusterer_on_sample = None, None, None
        best_backend = None
        
        if self.gpu_available:
            use_gpu_hdbscan = (
                self.use_gpu_hdbscan
                and diagnostics['duplicate_rows'] == 0
                and diagnostics['nonfinite_values'] == 0
                and len(X_for_selection) >= self.gpu_hdbscan_min_rows
            )
            hdbscan_backend = 'gpu_cuml' if use_gpu_hdbscan else 'cpu_sklearn'
            logging.info(
                "GPU execution enabled; HDBSCAN backend=%s, other GPU candidates unchanged.",
                hdbscan_backend,
            )
            if self.use_gpu_hdbscan and not use_gpu_hdbscan:
                logging.warning(
                    "Unsafe/degenerate HDBSCAN input detected; falling back from cuML to sklearn."
                )
            X_for_selection_gpu = None
            for model_name, param_list in self.candidate_configs.items():
                for params in param_list:
                    logging.info("Testing %s with params: %s", model_name, params)
                    labels, clusterer = None, None
                    candidate_backend = None
                    if model_name == 'hdbscan':
                        if (
                            len(X_for_selection) >= params['min_cluster_size']
                            and len(X_for_selection) > params['min_samples']
                        ):
                            if use_gpu_hdbscan:
                                if X_for_selection_gpu is None:
                                    X_for_selection_gpu = cupy.asarray(X_for_selection)
                                clusterer = cuHDBSCAN(**params).fit(X_for_selection_gpu)
                                labels = cupy.asnumpy(clusterer.labels_)
                            else:
                                clusterer = SklearnHDBSCAN(
                                    **params, n_jobs=-1, copy=True
                                ).fit(X_for_selection)
                                labels = np.asarray(clusterer.labels_)
                            candidate_backend = hdbscan_backend
                    elif model_name == 'kmeans':
                        if len(X_for_selection) > params['n_clusters']:
                            if X_for_selection_gpu is None:
                                X_for_selection_gpu = cupy.asarray(X_for_selection)
                            clusterer = cuKMeans(
                                n_clusters=params['n_clusters'],
                                init='scalable-k-means++',
                            )
                            clusterer.fit(X_for_selection_gpu)
                            labels = cupy.asnumpy(clusterer.labels_)
                            candidate_backend = 'gpu_cuml'
                    elif model_name == 'leiden':
                        _require_leiden_dependencies()
                        if X_for_selection_gpu is None:
                            X_for_selection_gpu = cupy.asarray(X_for_selection)
                        knn_cuml = cuKNN(
                            n_neighbors=params['k_neighbors'] + 1
                        ).fit(X_for_selection_gpu)
                        D_gpu, I_gpu = knn_cuml.kneighbors(X_for_selection_gpu)
                        I_cpu, D_cpu = cupy.asnumpy(I_gpu), cupy.asnumpy(D_gpu)
                        row = np.arange(X_for_selection.shape[0]).repeat(
                            params['k_neighbors']
                        )
                        col = I_cpu[:, 1:].flatten()
                        dist = D_cpu[:, 1:].flatten()
                        mask = dist > 0
                        row, col, dist = row[mask], col[mask], dist[mask]
                        weight = 1.0 / dist
                        A_sparse = coo_matrix(
                            (weight, (row, col)),
                            shape=(X_for_selection.shape[0], X_for_selection.shape[0]),
                        )
                        sources, targets = A_sparse.nonzero()
                        g = ig.Graph(edges=list(zip(sources, targets)), directed=False)
                        g.es['weight'] = A_sparse.data
                        clusterer = leidenalg.find_partition(
                            g,
                            leidenalg.RBConfigurationVertexPartition,
                            weights='weight',
                            resolution_parameter=params['resolution_parameter'],
                        )
                        labels = np.asarray(clusterer.membership)
                        candidate_backend = 'hybrid_gpu_cpu'

                    if labels is None:
                        logging.info(
                            "  -> Skipped: insufficient samples for this configuration."
                        )
                        continue
                    score = self._get_validation_score(X_for_selection, labels)
                    logging.info("  -> Score: %.4f", score)

                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        best_params = params
                        clusterer_on_sample = clusterer
                        best_backend = candidate_backend

        else: # This is the new parallel CPU logic
            logging.info("CPU execution. Parallelizing search with Ray.")
            X_ref = ray.put(X_for_selection)
            futures = []
            for model_name, param_list in self.candidate_configs.items():
                for params in param_list:
                    if model_name == 'hdbscan':
                        if (
                            len(X_for_selection) >= params['min_cluster_size']
                            and len(X_for_selection) > params['min_samples']
                        ):
                            futures.append(_fit_and_score_cpu.remote(model_name, params, X_ref))
                    elif model_name == "gmm":
                        if len(X_for_selection) > params['n_components']:
                            futures.append(_fit_and_score_cpu.remote(model_name, params, X_ref))
                    elif model_name == "kmeans":
                        if len(X_for_selection) > params['n_clusters']:
                            futures.append(_fit_and_score_cpu.remote(model_name, params, X_ref))
            
            results = ray.get(futures)
            if results:
                best_score, best_model_name, best_params = max(
                    results, key=lambda item: item[0]
                )
            else:
                labels = np.full(X.shape[0], -1, dtype=np.int32)
                return labels, {'name': 'degenerate', 'params': {}, 'backend': 'none'}

            # ✅ ADDED STEP: Perform a final fit on the subsample to get the trained model object
            logging.info("Performing final fit on subsample with winning parameters...")
            if best_model_name == 'hdbscan':
                _require_hdbscan_dependency()
                logging.info("Enabling prediction_data for the winning HDBSCAN model.")
                clusterer_on_sample = hdbscan.HDBSCAN(**best_params, core_dist_n_jobs=-1, prediction_data=True).fit(X_for_selection)
                best_backend = 'cpu_hdbscan'
            elif best_model_name == 'gmm':
                clusterer_on_sample = GaussianMixture(**best_params, covariance_type='full').fit(X_for_selection)
                best_backend = 'cpu_sklearn'
            elif best_model_name == 'kmeans':
                clusterer_on_sample = KMeans(**best_params, n_init=3).fit(X_for_selection)
                best_backend = 'cpu_sklearn'
            else:
                raise ValueError(f"Unsupported model type: {best_model_name}")
            
        logging.info("Stage 1 complete.")
        logging.info(
            "Best model found on subsample: %s with params %s (Score: %.4f)",
            best_model_name,
            best_params,
            best_score,
        )

        # --- STAGE 2: HYBRID STRATEGY FOR FINAL LABELING ---
        
        if best_model_name is None:
            labels = np.full(X.shape[0], -1, dtype=np.int32)
            return labels, {'name': 'degenerate', 'params': {}, 'backend': 'none'}

        final_params = best_params.copy()
        best_model_info = {
            'name': best_model_name,
            'params': final_params,
            'backend': best_backend,
        }

        if best_model_name == 'leiden':
            _require_leiden_dependencies()
            # --- PATH 1: LEIDEN WINS -> Re-fit on the full dataset for maximum accuracy ---
            logging.info("Stage 2 (Leiden): Re-fitting on the full dataset for maximum accuracy...")
            
            # This block is your original Leiden code, now applied to the full dataset X
            k = int(final_params.get('k_neighbors', 15))
            resolution = final_params.get('resolution_parameter', 1.0)

            logging.info("Building k-NN graph for %s samples with k=%s...", X.shape[0], k)
            if self.gpu_available:
                from cuml.neighbors import NearestNeighbors as cuKNN
                knn_cuml = cuKNN(n_neighbors=k + 1).fit(cupy.asarray(X))
                D_gpu, I_gpu = knn_cuml.kneighbors(cupy.asarray(X))
                I_cpu, D_cpu = cupy.asnumpy(I_gpu), cupy.asnumpy(D_gpu)
                row, col = np.arange(X.shape[0]).repeat(k), I_cpu[:, 1:].flatten()
                dist = D_cpu[:, 1:].flatten()
            else: # CPU
                X_faiss = np.ascontiguousarray(X, dtype=np.float32)
                index = faiss.IndexFlatL2(X_faiss.shape[1])
                index.add(X_faiss)
                dist, col = index.search(X_faiss, k + 1)
                dist, col = dist[:, 1:].flatten(), col[:, 1:].flatten()
                row = np.arange(X.shape[0]).repeat(k)
            
            logging.info("Constructing igraph object and running Leiden algorithm...")
            mask = dist > 0
            row, col, dist = row[mask], col[mask], dist[mask]
            weight = 1.0 / dist
            A_sparse = coo_matrix((weight, (row, col)), shape=(X.shape[0], X.shape[0]))
            sources, targets = A_sparse.nonzero()
            g = ig.Graph(edges=list(zip(sources, targets)), directed=False)
            g.es['weight'] = A_sparse.data
            partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution)
            final_labels = np.array(partition.membership)
            
        else: # GMM or HDBSCAN wins
            # --- PATH 2: GMM/HDBSCAN WINS -> Use memory-safe "Predict on Full" method ---
            logging.info("Stage 2 (GMM/HDBSCAN): Predicting on full dataset in chunks for memory safety...")
            final_labels = self._predict_on_full_dataset(X, clusterer_on_sample, best_model_info, X_for_selection)

        logging.info("Clustering of the full dataset is complete.")
        return final_labels, best_model_info

class ClusterBootstrapSampler:
    def __init__(
        self,
        noise_weight_factor=0.1,
        target_opt='maximize',
        batch_size=10,
        # ============ UMAP dimensionality-reduction parameters ==============
        enable_umap=False,
        umap_n_min_components=20,
        umap_n_max_components=100,
        high_dim_threshold=10000,
        # ============ GMM parameters, used as fallback or if the logic is extended ==============
        gmm_min_cluster_size=3,
        # cuML HDBSCAN can terminate the process on CUDA errors. CPU sklearn is
        # the safe default; explicitly opt in only after validating the RAPIDS stack.
        use_gpu_hdbscan=False,
    ):
        self.enable_umap = enable_umap
        self.umap_n_min_components = umap_n_min_components
        self.umap_n_max_components = umap_n_max_components
        self.high_dim_threshold = high_dim_threshold
        self.noise_weight_factor = noise_weight_factor
        self.target_opt = target_opt
        self.gmm_min_cluster_size = gmm_min_cluster_size
        self.use_gpu_hdbscan = use_gpu_hdbscan

    def rescale_features(self, X_scaled):
        n_samples, n_features = X_scaled.shape
        scale_factor = np.sqrt(n_features)
        X_rescaled = X_scaled * scale_factor
        return X_rescaled, scale_factor

    def refine_large_clusters(self, X, labels, min_points=50, ratio_threshold=0.2):
        total_samples = len(X)
        new_labels = labels.copy()
        next_cluster_id = max(labels) + 1
    
        for cid in np.unique(labels):
            if cid == -1:
                continue
            idx = np.where(labels == cid)[0]
            if len(idx) >= max(min_points, int(total_samples * ratio_threshold)):
                logging.info("[Refine] Cluster %s with size %s -> refining...", cid, len(idx))
    
                X_sub = X[idx]
                sub_clusterer = AdaptiveClusterer(
                    gpu_available=torch.cuda.is_available(),
                    use_gpu_hdbscan=self.use_gpu_hdbscan,
                )
                sub_labels, _ = sub_clusterer.find_best_clustering(X_sub)
    
                sub_unique = np.unique(sub_labels)
                for sub_cid in sub_unique:
                    if sub_cid == -1:
                        continue
                    sub_idx = np.where(sub_labels == sub_cid)[0]
                    global_idx = idx[sub_idx]
                    new_labels[global_idx] = next_cluster_id
                    next_cluster_id += 1
    
                # Points outside all subclusters are labeled as noise.
                noise_idx = idx[sub_labels == -1]
                new_labels[noise_idx] = -1
    
        return new_labels

    def compute_bootstrap_probabilities_clustering(self, X, enable_refine=False):
        """
        Performs adaptive clustering and returns the cluster labels.
        This method now uses the AdaptiveClusterer to find the best clustering.
        """
        # =============== 0. Input Sanitization, Scaling, and GPU Check ===============
        X_clean, diagnostics = _prepare_clustering_input(X)
        if diagnostics['nonfinite_values'] or diagnostics['constant_columns']:
            logging.warning(
                "Sanitized clustering data: repaired %s non-finite values and "
                "removed %s constant columns.",
                diagnostics['nonfinite_values'],
                diagnostics['constant_columns'],
            )
        if diagnostics['duplicate_rows']:
            logging.warning(
                "Detected %s duplicate clustering rows; preserving their sampling "
                "frequency and using the safe HDBSCAN backend.",
                diagnostics['duplicate_rows'],
            )
        if diagnostics['n_rows'] == 0:
            return np.empty(0, dtype=np.int32)
        if diagnostics['unique_rows'] < 2:
            logging.warning(
                "No meaningful clustering is possible; returning all-noise labels."
            )
            return np.full(diagnostics['n_rows'], -1, dtype=np.int32)

        X_scaled, scale = self.rescale_features(X_clean)
        logging.info("Applied dimension related scale factor: %.3f", scale)

        gpu_available = torch.cuda.is_available()
        logging.info(f"CUDA available: {gpu_available}")

        # Automatically determine if UMAP should be enabled
        if X_clean.shape[1] >= self.high_dim_threshold:
            self.enable_umap = True
        else:
            self.enable_umap = False

        # =============== 1. Dimensionality Reduction (if enabled) ===============
        X_for_cluster = X_scaled
        if self.enable_umap:
            if gpu_available:
                from cuml import UMAP
                logging.info("[Info] Using cuML GPU UMAP for dimensionality reduction.")
            else:
                from umap import UMAP
                logging.info("[Info] Using CPU UMAP for dimensionality reduction.")
            
            target_dim = max(min(self.umap_n_max_components, int(X_scaled.shape[0] / 2)), self.umap_n_min_components)
            umap_model = UMAP(n_components=target_dim, n_neighbors=30, min_dist=0.1, metric='euclidean')
            X_for_cluster = umap_model.fit_transform(X_scaled)
            if hasattr(X_for_cluster, 'copy_to_host'):
                X_for_cluster = X_for_cluster.copy_to_host()
            X_for_cluster = np.nan_to_num(X_for_cluster)

        # =============== 2. Adaptive Clustering via Compete-and-Select ===============
        adaptive_clusterer = AdaptiveClusterer(
            batch_size=10,
            gpu_available=gpu_available,
            use_gpu_hdbscan=self.use_gpu_hdbscan,
        )
        labels, best_model_info = adaptive_clusterer.find_best_clustering(X_for_cluster)
        
        logging.info(f"[AdaptiveClustering] Final model: {best_model_info['name']} with params {best_model_info['params']}")
        logging.info(f"Noise ratio: {np.mean(labels == -1):.3f}")

        # Post-processing: For GMM-like models, enforce a minimum cluster size
        if 'gmm' in best_model_info['name'] or 'kmeans' in best_model_info['name']:
            unique_labels, counts = np.unique(labels, return_counts=True)
            small_clusters = unique_labels[counts < self.gmm_min_cluster_size]
            for cid in small_clusters:
                labels[labels == cid] = -1
            logging.info(f"Filtered {len(small_clusters)} small clusters to noise. New noise ratio: {np.mean(labels == -1):.3f}")

        if enable_refine:
            labels = self.refine_large_clusters(X_for_cluster, labels)
            
        return labels
            
    def compute_bootstrap_probabilities_prob(self, labels, y):
        
        n = y.shape[0]
        if n == 0:
            return np.array([])
    
        if not (0.0 <= self.noise_weight_factor <= 1.0):
            raise ValueError("noise_weight_factor should be within 0-1")
    
        raw_scores = np.zeros(n, dtype=float)
        unique_labels = np.unique(labels)
    
        for label_val in unique_labels:
            if label_val == -1:
                continue
    
            idx = np.where(labels == label_val)[0]
            if len(idx) == 0:
                continue
            
            y_cluster = y[idx].flatten()
            
            median_val = np.median(y_cluster)
            try:
                spread = iqr(y_cluster, nan_policy='omit')
            except TypeError:
                y_cluster_no_nan = y_cluster[~np.isnan(y_cluster)]
                if len(y_cluster_no_nan) >=2 :
                     spread = iqr(y_cluster_no_nan)
                else:
                     spread = 0
    
            if spread == 0 or np.isnan(spread):
                spread = np.nanstd(y_cluster)
            
            if spread == 0 or np.isnan(spread):
                raw_scores[idx] = 1.0 
            else:
                if self.target_opt == 'maximize':
                    score_diff = y_cluster - median_val
                else:  # 'minimize'
                    score_diff = median_val - y_cluster
                
                # Score points relative to the median and normalize by dispersion.
                point_scores_in_cluster = np.maximum(score_diff, 0) / spread
                
                if len(point_scores_in_cluster) > 0:
                    max_point_score = np.max(point_scores_in_cluster)
                    weights = np.exp(point_scores_in_cluster - max_point_score)
                    raw_scores[idx] = weights
                else:
                    raw_scores[idx] = 0.0
    
        noise_idx = np.where(labels == -1)[0]
        if len(noise_idx) > 0:
            raw_scores[noise_idx] = 1.0
    
        # =============== 2. Allocate probability mass according to noise_weight_factor ===============
        probs = np.zeros(n, dtype=float)
        non_noise_indices = np.where(labels != -1)[0]
    
        # --- Handle non-noise points. ---
        actual_prob_mass_for_non_noise = 1.0 - self.noise_weight_factor
        if len(noise_idx) == 0:
            actual_prob_mass_for_non_noise = 1.0
        
        if len(non_noise_indices) > 0:
            scores_of_non_noise_points = raw_scores[non_noise_indices]
            sum_raw_scores_non_noise = np.sum(scores_of_non_noise_points)
    
            if sum_raw_scores_non_noise > 0:
                probs[non_noise_indices] = (scores_of_non_noise_points / sum_raw_scores_non_noise) * actual_prob_mass_for_non_noise
            else:
                probs[non_noise_indices] = (1.0 / len(non_noise_indices)) * actual_prob_mass_for_non_noise
        
        # --- Handle noise points. ---
        actual_prob_mass_for_noise = self.noise_weight_factor
        if len(non_noise_indices) == 0:
            actual_prob_mass_for_noise = 1.0
    
        if len(noise_idx) > 0:
            probs[noise_idx] = (1.0 / len(noise_idx)) * actual_prob_mass_for_noise
    
        # =============== 3. Final normalization, mainly for numerical precision ===============
        current_sum_probs = np.sum(probs)    
        if current_sum_probs > 0:
            probs = probs / current_sum_probs
        else:
            if n > 0:
                probs = np.ones(n) / n
    
        return probs


class ModelEvaluator:
    def __init__(
        self,
        X_train,
        y_train,
        file_path=None,
        bs_sample_number=None,
        optimization_goal='maximize',
        max_cap=None,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.file_path = file_path if file_path is not None else f'{os.getcwd()}/model_weights'
        candidate_bs = 10 * (X_train.shape[1] ** 2)
        prop_bs = int(0.5 * X_train.shape[0])
        self.bs_sample_number = min(X_train.shape[0], 10000) if bs_sample_number is None else bs_sample_number
        self.optimization_goal = optimization_goal
        self.max_cap = max_cap
        
        self.Clustersampler = ClusterBootstrapSampler(enable_umap=False)
        self.global_labels = self.Clustersampler.compute_bootstrap_probabilities_clustering(self.X_train, enable_refine=True)

    def save_models(self, model_name, optimized_params, models, model_errors, residuals, file_name, elpds, oob_sizes, elpd_scores, elpd_per_point_mean, crps, **extra):
        if not os.path.exists(f'{self.file_path}'):
            os.mkdir(f'{self.file_path}')
        payload = {
            'model_name': model_name,
            'optimized_params': optimized_params,
            'models': models,
            'errors': model_errors,
            'residuals': residuals,
            'elpds': elpds, 
            'oob_sizes': oob_sizes, 
            'elpd_scores': elpd_scores,
            'elpd_per_point_mean': elpd_per_point_mean, 
            'crps': crps
        }
        payload.update(extra)

        with open(f'{self.file_path}/{file_name}', 'wb') as f:
            pickle.dump(payload, f)

    def load_models(self, file_name):
        with open(f'{self.file_path}/{file_name}', 'rb') as f:
            data = pickle.load(f)
        return data
        
    def bootstrap_evaluation(self, model_name, optimized_params, num_target,
                             n_bootstrap_sample_nums=20, cls=False, use_full_eval=False,
                             cross_val=False, cv_n_splits=5, uni_params=True):
        n_samples = len(self.X_train)
        errors, models, residuals = [], [], []
        elpds, oob_sizes, elpd_scores, crps = [], [], [], []
    
        # OOB statistic accumulators.
        oob_var_list, oob_mad_var_list, oob_n_list = [], [], []
    
        X_bs = self.X_train
        y_bs = self.y_train[:, num_target]
        X_bs_ref = ray.put(X_bs)
        y_bs_ref = ray.put(y_bs)
    
        if n_bootstrap_sample_nums < 2:
            n_bootstrap_sample_nums = 2
        cv_n_splits = min(n_bootstrap_sample_nums, cv_n_splits)
    
        if cross_val:
            cross_val_tasks = []
            kf = KFold(n_splits=cv_n_splits, shuffle=True)
            for train_idx, val_idx in kf.split(X_bs):
                if optimized_params is None or not uni_params:
                    optimized_params = hyperparameter_optimization(
                        model_name, X_bs, y_bs, cls=cls, max_cap=self.max_cap
                    )
                cross_val_tasks.append(self._train_model.remote(self, model_name, optimized_params,
                                                                X_bs_ref, y_bs_ref, train_idx, cls, use_full_eval))
            results = ray.get(cross_val_tasks)
    
            for res in results:
                models.append(res['model'])
                errors.append(res['error'])
                residuals.append(res['residual'])
                elpds.append(res.get('elpd', 0.0))
                crps.append(res.get('crps', 0.0))
                oob_sizes.append(res.get('oob_size', 0))
                elpd_scores.append(float(res['elpd']) / max(res.get('oob_size', 1), 1))
    
                # Append OOB statistics.
                oob_var_list.append(res.get('oob_var', 0.0))
                oob_mad_var_list.append(res.get('oob_mad_var', 0.0))
                oob_n_list.append(int(res.get('oob_size', 0)))
    
        else:
            if model_name == 'GP_gpu':
                # GP branch: complete the OOB statistics.
                X_tr, X_te, y_tr, y_te = train_test_split(X_bs, y_bs, test_size=0.2)
                model = SurrogateModel(model_name, optimized_params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
    
                if cls:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_te)
                        p_true = proba[np.arange(len(y_te)), y_te]
                        elpd = float(np.sum(np.log(p_true + 1e-12)))
                        y_pred_lbl = np.argmax(proba, axis=1)
                        error = float(np.mean(y_pred_lbl == y_te))
                        brier = float(np.mean(np.sum((proba - np.eye(proba.shape[1])[y_te])**2, axis=1)))
                        crps_val = brier
                    else:
                        y_pred = model.predict(X_te)
                        error = float(accuracy_score(y_te, y_pred))
                        elpd = float(len(y_te) * np.log(error + 1e-6))
                        crps_val = 1.0 - error
                    # OOB statistics for classification.
                    res_cls = (np.argmax(model.predict_proba(X_te), axis=1) - y_te).astype(float) if hasattr(model, "predict_proba") else (y_pred - y_te).astype(float)
                    oob_n = int(len(y_te))
                    oob_var = float(np.var(res_cls, ddof=1)) if oob_n > 1 else 0.0
                    oob_mad = np.median(np.abs(res_cls - np.median(res_cls))) if oob_n > 0 else 0.0
                    oob_mad_var = float((oob_mad / 0.67448975) ** 2) if oob_n > 0 else 0.0
                else:
                    res = y_te - preds
                    eps = 1e-12
                    sigma2 = max(float(np.var(res, ddof=1)), eps) if len(res) > 1 else eps
                    elpd = float(-0.5 * np.sum(np.log(2.0 * np.pi * sigma2) + (res ** 2) / sigma2))
                    error = float(np.clip(r2_score(y_te, preds), 0, np.inf))
                    # Simple Gaussian CRPS approximation.
                    crps_val = float(np.mean(_gaussian_crps(y_te, preds, np.sqrt(sigma2))))
    
                    # OOB statistics for regression.
                    oob_n = int(len(res))
                    oob_var = float(np.var(res, ddof=1)) if oob_n > 1 else 0.0
                    oob_mad = np.median(np.abs(res - np.median(res))) if oob_n > 0 else 0.0
                    oob_mad_var = float((oob_mad / 0.67448975) ** 2) if oob_n > 0 else 0.0
    
                y_pred = model.predict(X_bs)
                residual = y_bs - y_pred
    
                errors.append(error)
                models.append(model)
                residuals.append(residual)
                elpds.append(elpd)
                crps.append(crps_val)
                oob_sizes.append(oob_n)
                elpd_scores.append(float(elpd) / max(oob_n, 1))
                oob_var_list.append(oob_var)
                oob_mad_var_list.append(oob_mad_var)
                oob_n_list.append(oob_n)
    
            else:
                # Standard or clustered bootstrap branch; same logic, but reads the new _train_model keys.
                if n_samples < 10 * X_bs.shape[1]**2:
                    bootstrap_tasks = []
                    for i in range(n_bootstrap_sample_nums):
                        bootstrap_indices = np.random.choice(np.arange(n_samples), size=self.bs_sample_number, replace=True)
                        if optimized_params is None or not uni_params:
                            optimized_params = hyperparameter_optimization(
                                model_name,
                                X_bs[bootstrap_indices],
                                y_bs[bootstrap_indices],
                                cls=cls,
                                max_cap=self.max_cap,
                            )
                        bootstrap_tasks.append(self._train_model.remote(self, model_name, optimized_params, X_bs_ref, y_bs_ref, bootstrap_indices, cls, use_full_eval))
                    results = ray.get(bootstrap_tasks)
                else:
                    probs = self.Clustersampler.compute_bootstrap_probabilities_prob(self.global_labels, y_bs)
                    weights_baseline = np.ones_like(probs) / len(probs)
                    bootstrap_tasks = []
                    for i in range(n_bootstrap_sample_nums):
                        if i < int(n_bootstrap_sample_nums/2):
                            ratio = np.random.uniform(0.6, 1.0)
                            final_weights = ratio * weights_baseline + (1-ratio) * probs
                            final_weights /= np.sum(final_weights)
                            bootstrap_indices = np.random.choice(np.arange(n_samples), size=int(self.bs_sample_number), replace=True, p=final_weights)
                        else:
                            bootstrap_indices = np.random.choice(np.arange(n_samples), size=self.bs_sample_number, replace=True)
                        if optimized_params is None or not uni_params:
                            optimized_params = hyperparameter_optimization(
                                model_name,
                                X_bs[bootstrap_indices],
                                y_bs[bootstrap_indices],
                                cls=cls,
                                max_cap=self.max_cap,
                            )
                        bootstrap_tasks.append(self._train_model.remote(self, model_name, optimized_params, X_bs_ref, y_bs_ref, bootstrap_indices, cls, use_full_eval))
                    results = ray.get(bootstrap_tasks)
    
                for res in results:
                    models.append(res['model'])
                    errors.append(res['error'])
                    residuals.append(res['residual'])
                    elpds.append(res.get('elpd', 0.0))
                    crps.append(res.get('crps', 0.0))
                    oob_sizes.append(res.get('oob_size', 0))
                    elpd_scores.append(float(res['elpd']) / max(res.get('oob_size', 1), 1))
                    # Append OOB statistics.
                    oob_var_list.append(res.get('oob_var', 0.0))
                    oob_mad_var_list.append(res.get('oob_mad_var', 0.0))
                    oob_n_list.append(int(res.get('oob_size', 0)))
    
        # ===== Compute the overall mean ELPD per point. =====
        total_oob = max(sum(oob_sizes), 1)
        elpd_sum = float(sum(elpds))
        elpd_per_point_mean = elpd_sum / total_oob
        elpd_scores = np.array(elpd_scores)
    
        # ======= Compute and save the aggregated OOB variance. =======
        oob_n_arr = np.array(oob_n_list, dtype=float)
        w = np.where(oob_n_arr > 0, oob_n_arr, 0.0)
        if w.sum() > 0:
            oob_var_mean = float(np.sum(np.array(oob_var_list) * w) / np.sum(w))
            oob_mad_var_mean = float(np.sum(np.array(oob_mad_var_list) * w) / np.sum(w))
        else:
            # Fall back to a simple average if OOB data is missing or unexpected.
            oob_var_mean = float(np.mean(oob_var_list)) if len(oob_var_list) else 0.0
            oob_mad_var_mean = float(np.mean(oob_mad_var_list)) if len(oob_mad_var_list) else 0.0
    
        # Save results.
        self.save_models(
            model_name, optimized_params, models, errors, residuals, f"{model_name}_{num_target}.pkl",
            elpds=elpds, oob_sizes=oob_sizes, elpd_scores=elpd_scores,
            elpd_per_point_mean=elpd_per_point_mean, crps=crps,
            # Additional OOB statistics.
            oob_var_list=oob_var_list, oob_mad_var_list=oob_mad_var_list, oob_n_list=oob_n_list,
            oob_var_mean=oob_var_mean, oob_mad_var_mean=oob_mad_var_mean
        )
        logging.info(f"{model_name} | OOB-score(mean)={np.mean(errors):.4f}, std={np.std(errors):.4f} | "
                     f"ELPD/pt={elpd_per_point_mean:.4f} | CRPS(mean)={np.mean(crps):.4f} | "
                     f"OOB σ²(mean)={oob_var_mean:.4e} (MAD²={oob_mad_var_mean:.4e})")
    
        return [models, errors, residuals, elpds, oob_sizes, elpd_scores, elpd_per_point_mean, crps, oob_var_list, oob_mad_var_list, oob_n_list, oob_var_mean, oob_mad_var_mean]

    @ray.remote
    def _train_model(self, model_name, optimized_params, X_bs, y_bs, bootstrap_indices, cls, use_full_eval,
                     likelihood: str = "student_t",     # "gaussian" or "student_t".
                     student_t_nu: float | None = None, # None selects the value automatically by grid search.
                     heteroscedastic: bool = True,      # Use predict_std/var if the model supports it.
                     crps_mc_samples: int = 256,        # Number of Monte Carlo samples for Student-t CRPS.
                     rng_seed: int | None = None):
        
        rng = np.random.default_rng(rng_seed)
    
        model = SurrogateModel(model_name, optimized_params)
        model.fit(X_bs[bootstrap_indices], y_bs[bootstrap_indices])
    
        # Evaluation set: OOB or full data.
        if use_full_eval:
            X_eval = X_bs
            y_eval = y_bs
        else:
            eval_indices = np.setdiff1d(np.arange(len(X_bs)), bootstrap_indices)
            X_eval = X_bs[eval_indices] if len(eval_indices) != 0 else X_bs
            y_eval = y_bs[eval_indices] if len(eval_indices) != 0 else y_bs
    
        preds = model.predict(X_eval)
        eps = 0.0001
        min_sigma = 0.03 
    
        if cls:
            # ---------- Classification: ELPD = sum(log p_true), CRPS maps to Brier score. ----------
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_eval)
            elif hasattr(model, "decision_function"):
                z = model.decision_function(X_eval)
                if z.ndim == 1:
                    z = np.stack([-z, z], axis=1)
                z = z - np.max(z, axis=1, keepdims=True)
                proba = np.exp(z); proba /= np.sum(proba, axis=1, keepdims=True)
            else:
                # Fallback for robustness only.
                y_pred_lbl = model.predict(X_eval)
                acc = float(np.mean(y_pred_lbl == y_eval)) if len(y_eval) else 0.0
                elpd = len(y_eval) * np.log(acc + 1e-6)
                brier = 1.0 - acc
                y_pred_full = model.predict(X_bs)
                residual = y_bs - y_pred_full
                # OOB statistics using Brier residual approximation for classification.
                oob_n = int(len(y_eval))
                oob_var = float(np.var((y_pred_lbl - y_eval).astype(float), ddof=1)) if oob_n > 1 else 0.0
                oob_mad = np.median(np.abs((y_pred_lbl - y_eval).astype(float) - np.median((y_pred_lbl - y_eval).astype(float)))) if oob_n > 0 else 0.0
                oob_mad_var = float((oob_mad / 0.67448975) ** 2) if oob_n > 0 else 0.0
                return {'model': model,
                        'error': acc,
                        'residual': residual,
                        'elpd': float(elpd),
                        'crps': float(brier),
                        'oob_size': oob_n,
                        'oob_var': oob_var,
                        'oob_mad_var': oob_mad_var}
    
            p_true = proba[np.arange(len(y_eval)), y_eval]
            elpd = float(np.sum(np.log(np.maximum(p_true, eps))))
            y_pred_lbl = np.argmax(proba, axis=1)
            acc = float(np.mean(y_pred_lbl == y_eval)) if len(y_eval) else 0.0
    
            # Multiclass Brier score: the classification analogue of CRPS.
            K = proba.shape[1]
            Y_onehot = np.eye(K)[y_eval]
            brier_per = np.sum((proba - Y_onehot) ** 2, axis=1)
            brier = float(np.mean(brier_per)) if len(brier_per) else 1.0
    
            error = acc
            crps = brier
    
            # OOB statistics using 1 - max class probability as a noise proxy.
            oob_n = int(len(y_eval))
            # A 0/1 residual from (y_pred_lbl == y_eval) is another possible approximation.
            oob_var = float(np.var((y_pred_lbl != y_eval).astype(float), ddof=1)) if oob_n > 1 else 0.0
            oob_mad = np.median(np.abs((y_pred_lbl != y_eval).astype(float) - np.median((y_pred_lbl != y_eval).astype(float)))) if oob_n > 0 else 0.0
            oob_mad_var = float((oob_mad / 0.67448975) ** 2) if oob_n > 0 else 0.0
    
        else:
            # ---------- Regression: ELPD + CRPS. ----------
            res = y_eval - preds
            # Predictive uncertainty, heteroscedastic when available; otherwise estimate it from OOB residuals.
            pred_std = _get_pred_std_if_available(model, X_eval) if heteroscedastic else None
    
            if likelihood.lower() == "gaussian":
                if pred_std is not None:
                    sigma = np.maximum(pred_std, min_sigma)
                else:
                    var = float(np.var(res, ddof=1)) if len(res) > 1 else 1.0
                    raw_sigma_val = np.sqrt(max(var, eps))
                    final_sigma_val = np.maximum(raw_sigma_val, min_sigma)
                    sigma = np.full_like(res, final_sigma_val, dtype=float)
    
                elpd = float(np.sum(_gaussian_logpdf(y_eval, preds, sigma)))
                # R^2 reference metric.
                tss = np.sum((y_eval - np.mean(y_eval)) ** 2)
                r2 = 1.0 - (np.sum(res ** 2) / (tss + eps)) if tss > 0 else 0.0
                error = float(np.clip(r2, 1e-8, np.inf))
                # Gaussian closed-form CRPS.
                crps = float(np.mean(_gaussian_crps(y_eval, preds, sigma)))
                chosen_nu = None
    
            elif likelihood.lower() == "student_t":
                if pred_std is not None:
                    sigma = np.maximum(pred_std, min_sigma)
                else:
                    var = float(np.var(res, ddof=1)) if len(res) > 1 else 1.0
                    raw_sigma_val = np.sqrt(max(var, eps))
                    final_sigma_val = np.maximum(raw_sigma_val, min_sigma)
                    sigma = np.full_like(res, final_sigma_val, dtype=float)
    
                if (student_t_nu is None) or (student_t_nu <= 2.0):
                    chosen_nu = _choose_student_t_nu_by_grid(res, sigma)
                else:
                    chosen_nu = float(student_t_nu)
    
                c = np.sqrt((chosen_nu - 2.0) / chosen_nu) if chosen_nu < 1e6 else 1.0
                s = np.maximum(sigma * c, 1e-12)
    
                elpd = float(np.sum(_student_t_logpdf(res, s, chosen_nu)))
                # R^2 reference metric.
                tss = np.sum((y_eval - np.mean(y_eval)) ** 2)
                r2 = 1.0 - (np.sum(res ** 2) / (tss + eps)) if tss > 0 else 0.0
                error = float(np.clip(r2, 1e-8, np.inf))
                # Student-t CRPS using an unbiased Monte Carlo estimate.
                crps_vals = _student_t_crps_mc(y_eval, preds, s, chosen_nu,
                                               n_samples=crps_mc_samples, rng=rng)
                crps = float(np.mean(crps_vals))
            else:
                raise ValueError(f"Unknown likelihood: {likelihood}")
    
            # OOB residual variance as data-level noise.
            oob_n = int(len(res))
            oob_var = float(np.var(res, ddof=1)) if oob_n > 1 else 0.0
            oob_mad = np.median(np.abs(res - np.median(res))) if oob_n > 0 else 0.0
            oob_mad_var = float((oob_mad / 0.67448975) ** 2) if oob_n > 0 else 0.0
    
        # Full residuals, preserving the existing save logic.
        y_pred_full = model.predict(X_bs)
        residual = y_bs - y_pred_full
    
        out = {
            'model': model,
            'error': error,              # Classification: accuracy; regression: R^2 reference metric.
            'residual': residual,
            'elpd': float(elpd),
            'crps': float(crps),
            'oob_size': oob_n,
            'oob_var': oob_var,
            'oob_mad_var': oob_mad_var
        }
        if not cls and likelihood.lower() == "student_t":
            out['nu'] = float(chosen_nu)
        return out

    def evaluate(self, model_names, num_target, n_bootstrap_sample_nums, cls=False, use_full_eval=False, cross_val=False, uni_hyper=False):
        model_results = {}
    
        for model_name in model_names:
            if model_name == 'GP_gpu' and len(self.X_train) <= 500:
                cross_val = True
            elif model_name == 'GP_gpu' and len(self.X_train) > 500:
                cross_val = False
    
            if uni_hyper:
                probs = self.Clustersampler.compute_bootstrap_probabilities_prob(self.global_labels, self.y_train[:, num_target])
                weights_baseline = np.ones_like(probs) / len(probs)
                ratio = np.random.uniform(0.5, 1.0)
                unihypr_weights = ratio * weights_baseline + (1-ratio) * probs
                unihypr_weights /= np.sum(unihypr_weights)
                indices = np.arange(len(self.X_train))
                bootstrap_indices = np.random.choice(indices, size=int(len(self.X_train)), replace=True, p=unihypr_weights)
                optimized_params = hyperparameter_optimization(
                    model_name,
                    self.X_train[bootstrap_indices],
                    self.y_train[:, num_target][bootstrap_indices],
                    cls=cls,
                    max_cap=self.max_cap,
                )
            else:
                optimized_params = None

            models, errors, residuals, elpds, oob_sizes, elpd_scores, elpd_per_point_mean, crps, oob_var_list, oob_mad_var_list, oob_n_list, oob_var_mean, oob_mad_var_mean = self.bootstrap_evaluation(
                model_name, optimized_params, num_target,
                n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=cls,
                use_full_eval=use_full_eval, cross_val=cross_val, uni_params=uni_hyper
            )
    
            # Store the score directly here.
            model_results[model_name] = {
                'models': models,
                'errors': errors,              # Reference metric: regression = OOB R^2; classification = OOB accuracy.
                'residuals': residuals,
                'elpd': elpds,                 # Per-round ELPD sum.
                'oob_sizes': oob_sizes,        # Per-round OOB size.
                'elpd_scores': elpd_scores,            # Key score: ELPD per point, where larger is better.
                'elpd_per_point_mean': elpd_per_point_mean,
                'crps': crps,
                'oob_var_list': oob_var_list, 
                'oob_mad_var_list': oob_mad_var_list, 
                'oob_n_list': oob_n_list, 
                'oob_var_mean': oob_var_mean, 
                'oob_mad_var_mean': oob_mad_var_mean
            }
    
        return model_results

    ### possible meta classifiers: RidgeCV, LogisticRegression, RidgeClassifier, LinearRegression, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
    def train_stacking_model(self, model_results=None, num_target=0, cls=False, meta_classifier=None, use_probas=False, model_name_list=None, cv=None):
        if model_results is None:
            if model_name_list is None:
                raise ValueError("When model_results is None, model_name_list must be provided.")
            model_results = {}
            for model_name in model_name_list:
                file_path = f"{self.file_path}/{model_name}_{num_target}.pkl"
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                model_results[model_name] = data
        
        base_models = [(model_name, AbstractSurrogateModel(model_name, model_info['models'])) for model_name, model_info in model_results.items()]
        
        if meta_classifier is None:
            meta_classifier = LogisticRegression(penalty='l2') if cls else RidgeCV(alphas=np.logspace(-6, 6, 13))

        if cv is None:
            cv = 5

        if use_probas and cls:
            stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier, stack_method='predict_proba', cv=cv)
        else:
            stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier, cv=cv) if cls else StackingRegressor(estimators=base_models, final_estimator=meta_classifier, cv=cv)

        X_meta = self.X_train
        y_meta = self.y_train[:, num_target]
        stacking_model.fit(X_meta, y_meta)
        
        # --- Calculate residuals ---
        y_pred = stacking_model.predict(X_meta)
        residuals = y_meta - y_pred
        
        if hasattr(stacking_model.final_estimator_, 'coef_'):
            base_model_contributions = stacking_model.final_estimator_.coef_
        elif hasattr(stacking_model.final_estimator_, 'feature_importances_'):
            base_model_contributions = stacking_model.final_estimator_.feature_importances_
        else:
            base_model_contributions = None

        base_model_contributions = base_model_contributions/(max(base_model_contributions)-min(base_model_contributions))
        base_model_errors = {}
        for i, (model_name, _) in enumerate(base_models):
            if base_model_contributions is not None:
                contribution_score = base_model_contributions[i]
            else:
                contribution_score = None
            base_model_errors[model_name] = contribution_score

        return stacking_model, base_model_errors, residuals

    def evaluate_with_stacking(self, model_names, num_target, n_bootstrap_sample_nums, cls=False, use_full_eval=False, cross_val=False, meta_classifier=None, use_probas=False, uni_hyper=False):
        model_results = self.evaluate(model_names, num_target, n_bootstrap_sample_nums, cls=cls, use_full_eval=use_full_eval, cross_val=cross_val, uni_hyper=uni_hyper)
        stacking_model, base_model_errors, stacking_residuals = self.train_stacking_model(model_results, num_target, cls=cls, meta_classifier=meta_classifier, use_probas=use_probas, model_name_list=model_names)
        
        model_results['stacking_models']={'errors':base_model_errors, 'models':stacking_model, 'residuals':stacking_residuals}
        
        with open(f'{self.file_path}/stacking_models_{num_target}.pkl', 'wb') as f:
            pickle.dump(model_results['stacking_models'], f)
        
        return model_results

    def calculate_and_save_residual_correlation(self, model_names, stacking=False):
        """
        Assembles pre-calculated residuals from saved files and computes the
        correlation matrix. This is now much faster.
        """
        logging.info("--- Assembling Pre-Calculated Residuals for Correlation ---")

        # 1. Load the pre-calculated residuals for each target of each model
        if stacking:
            all_models = model_names + ['stacking_models']
        else:
            all_models = model_names
            
        residual_dict = {}
        n_samples, num_of_targets = self.X_train.shape[0], self.y_train.shape[1]
        
        for model in all_models:
            all_residuals = np.zeros((n_samples, num_of_targets))
            for i in range(num_of_targets):
                logging.info(f"Loading residuals for  target {i}...")
                try:
                    with open(f'{self.file_path}/{model}_{i}.pkl', 'rb') as f:
                        data = pickle.load(f)
                    all_residuals[:, i] = np.mean(np.array(data['residuals']), axis=0)
                except (FileNotFoundError, KeyError):
                    raise FileNotFoundError(
                        f"{model}_{i}.pkl not found or is missing 'stacking_residuals'. "
                        "Please ensure `evaluate_with_stacking` was run with the updated code."
                    )
    
            # 2. Compute and save the correlation matrix (same as before)
            residual_corr_matrix = np.corrcoef(all_residuals, rowvar=False)
            
            if np.any(np.isnan(residual_corr_matrix)):
                logging.warning("NaNs found in residual correlation matrix. Replacing with 0.")
                residual_corr_matrix = np.nan_to_num(residual_corr_matrix)
    
            logging.info(f"Calculated Residual Correlation Matrix of {model}:\n{residual_corr_matrix}")
            residual_dict[model] = residual_corr_matrix
            
        save_path = f'{self.file_path}/residual_correlation.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(residual_dict, f)
        logging.info(f"All model's residual correlation matrix saved to {save_path}")

        return residual_dict

    def MT_train_stacking_model(self, model_names, corr_model_save_paths, n_bootstrap_sample_nums=20, num_target=0, cls=False, meta_classifier=None, use_probas=False):
        n_samples = len(self.X_train)
        model_results = {}
        for model_name in model_names:
            for path in corr_model_save_paths:
                file_path = f"{path}/{model_name}_{num_target}.pkl"
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                model_results[f'{path[-1]}_{model_name}'] = data

        base_models = [(model_name, AbstractSurrogateModel(model_name, model_info['models'])) for model_name, model_info in model_results.items()]

        X_meta = self.X_train
        y_meta = self.y_train[:, num_target]

        model_tasks = []
        for i in range(n_bootstrap_sample_nums):
        
            bootstrap_indices = np.random.choice(np.arange(n_samples), size=int(n_samples-2), replace=True)
            X_sample = X_meta[bootstrap_indices]
            y_sample = y_meta[bootstrap_indices]
            
            if meta_classifier is None:
                meta_classifier = RandomForestClassifier() if cls else RandomForestRegressor()
        
            if use_probas and cls:
                stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier, stack_method='predict_proba')
            else:
                stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier) if cls else StackingRegressor(estimators=base_models, final_estimator=meta_classifier)
                
            stacking_model.fit(X_sample, y_sample)
            model_tasks.append(stacking_model)
            
        if not os.path.exists(f'{self.file_path}'):
            os.mkdir(f'{self.file_path}')
            
        with open(f'{self.file_path}/correlated_stacking_results_{num_target}.pkl', 'wb') as f:
            pickle.dump(model_tasks, f)
    
        return model_tasks
