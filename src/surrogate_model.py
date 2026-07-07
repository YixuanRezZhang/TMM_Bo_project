import os, logging, math, warnings, numbers, random, inspect, torch, optuna
import numpy as np
if not hasattr(np, "int"): np.int = int
from optuna.pruners import MedianPruner, HyperbandPruner, PercentilePruner, PatientPruner

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from botorch.models.transforms import Standardize
from fastkan import FastKAN as FastKAN
from kan import KAN
from kan.utils import create_dataset_from_data

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from skdim.id import MLE, TwoNN
from sliced import SlicedInverseRegression as SIR
from sliced import SlicedAverageVarianceEstimation as SAVE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Suppress warning messages.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_DEFAULT_EARLY_STOP_ROUNDS = 50

def _intrinsic_dim_svd(X, energy_ratio=0.95):
    Xs = StandardScaler().fit_transform(X)
    U, S, VT = randomized_svd(Xs, n_components=min(X.shape)-1)
    cumsum = np.cumsum(S**2) / np.sum(S**2)
    return int(np.searchsorted(cumsum, energy_ratio) + 1)

def _sir_dim_auto(
    X,
    y,
    n_slices=10,
):

    selector = VarianceThreshold(threshold=0.0)
    try:
        X_filtered = selector.fit_transform(X)

        if X_filtered.shape[1] == 0:
            logging.warning("All features have zero variance in this data slice. SIR cannot run. Defaulting to d_sup=1.")
            return 1 
            
    except ValueError:
        logging.warning("Input X to _sir_dim_auto is empty. Defaulting to d_sup=1.")
        return 1

    n_samples, n_features = X_filtered.shape
    n_comp = max(1, min(n_samples - 1, n_features))
    if n_features > n_samples:
        pca = PCA(n_components=n_comp, svd_solver="auto")
        fit_X = pca.fit_transform(X_filtered)
    else:
        fit_X = X_filtered
        
    # ----- 1) SIR -----
    sir = SIR(
        n_slices=min(n_slices, max(2, n_samples // 2)),
        n_directions="auto",
    )
    try:
        T_sir = sir.fit_transform(fit_X, y)
        d_sir = T_sir.shape[1]
        if d_sir < 1:
            raise ValueError("SIR returned zero directions.")
    except (ValueError, np.linalg.LinAlgError) as err:
        logging.warning(f"SIR failed ({err}); default d_sup=1.")
        d_sir = 1

    return d_sir

def rescale_features(X_scaled):
    n_samples, n_features = X_scaled.shape
    scale_factor = np.sqrt(n_features)
    X_rescaled = X_scaled * scale_factor
    return X_rescaled, scale_factor

def estimate_budget(
    X_ori,
    y_ori,
    base=32,
    max_cap=1000,
    lambda_unsup=1.0,
    growth="loglinear",
):

    _, unique_idx = np.unique(X_ori, axis=0, return_index=True)
    X = X_ori[sorted(unique_idx)]
    X, _ = rescale_features(X)
    y = y_ori[sorted(unique_idx)]
    logging.info("estimate_budget input: X_shape=%s, has_non_finite=%s", X.shape, np.any(~np.isfinite(X)))
    # 1) Unsupervised dimension estimate.
    d_unsup = _intrinsic_dim_svd(X)

    # 2) Supervised dimension estimate.
    if np.unique(y).size > 2:
        d_sup = _sir_dim_auto(X, y)
    else:
        d_sup = 1

    try:
        d_est = int(np.ceil(MLE().fit_transform(X)))
        assert np.isfinite(d_est) and d_est >= 1
        logging.info(f"[estimate_budget] MLE ID = {d_est}")
    except Exception as e_mle:
        logging.warning(f"[estimate_budget] MLE failed: {e_mle}. Trying DANCo…")
        try:
            # The default k is 10; reduce it for small sample sizes if needed.
            d_est = int(np.ceil(TwoNN().fit_transform(X)))
            assert np.isfinite(d_est) and d_est >= 1
            logging.info(f"[estimate_budget] TwoNN ID = {d_est}")
        except Exception as e_twonn:
            logging.warning(f"[estimate_budget] TwoNN also failed: {e_twonn}. Falling back to PCA.")
            try:
                n_components = min(300, X.shape[0], X.shape[1])
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X)
                var = np.var(X_pca, axis=0)
                k = np.searchsorted(np.cumsum(var) / np.sum(var), 0.95) + 1
                d_est = max(1, k)
                logging.info(f"[estimate_budget] PCA fallback estimated ID: {d_est}")
            except Exception as e3:
                logging.error(f"[estimate_budget] PCA fallback also failed: {e3}. Using d_est=1 as last resort.")
                d_est = 1

    
    logging.info("estimate_budget dims: unsup=%s, sup=%s, est=%s", d_unsup, d_sup, d_est)
    # Alternative effective dimension rule: max(d_sup, lambda_unsup*d_unsup, d_mle).
    d_eff = int((d_sup + d_est + lambda_unsup * d_unsup)/3)
    d_all = d_sup+d_unsup+d_est

    # 3) Budget curve.
    if growth == "linear":
        budget = base * d_eff
    elif growth == "loglinear":
        budget = base * d_eff * math.log1p(d_sup)
    elif growth == "sqrt":
        budget = base * math.sqrt(d_eff)
    elif growth == "hybrid":
        budget = (
            base * d_eff * math.log1p(d_sup)
            if d_eff < 50
            else base * math.sqrt(d_eff) * math.log1p(d_sup)
        )
    else:
        raise ValueError(growth)

    return int(min(max_cap, max(10, round(budget)))), d_eff, d_all

class RegressionToClassificationWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return (self.regressor.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        positive_proba = np.clip(np.asarray(self.regressor.predict(X)).reshape(-1), 0.0, 1.0)
        return np.column_stack((1.0 - positive_proba, positive_proba))

def filter_gp_params(params):
    allowed_keys = {"alpha", "optimizer", "n_restarts_optimizer", "copy_X_train", "random_state"}
    return {k: v for k, v in params.items() if k in allowed_keys}

class SurrogateModel:
    def __init__(self, model_name=None, params=None):
        self.model_name = model_name
        if params:
            self.params = {k: v for k, v in params.items() if k != 'resource'}
        else:
            self.params = {}
        self.model = self._initialize_model()

    def _initialize_model(self):
        models = {
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'KNeighborsRegressor': KNeighborsRegressor,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'RandomForest': RandomForestRegressor,
            'SVR': SVR,
            'MLPRegressor': self._initialize_mlp_model,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'AdaBoostRegressor': AdaBoostRegressor,
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'KRR': KernelRidge, 
            'XGBoost': xgb.XGBRegressor,
            'CatBoost': lambda **kwargs: cb.CatBoostRegressor(verbose=0, **kwargs),
            'LightGBM': lambda **kwargs: lgb.LGBMRegressor(verbose=-1, **kwargs),
            'GP_cpu': lambda **kwargs: GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + 
                       WhiteKernel(noise_level=1e-5),
                # random_state=0,
                # normalize_y=True,
                **filter_gp_params(kwargs)),
            'GP_gpu': self._initialize_gp_model,
            'KAN': self._initialize_kan_model,
            'FastKAN': self._initialize_fast_kan_model,
        }
        if self.model_name is None or self.model_name not in models.keys():
            raise ValueError(f"Unknown model name: {self.model_name}")
        model_class = models.get(self.model_name)

        return model_class(**self.params)

    def _initialize_gp_model(self):
        return None  # Placeholder for the Gaussian process model

    def _initialize_mlp_model(self, **kwargs):

        if "hidden_layer_sizes" in self.params:
            hidden_layers = list(self.params["hidden_layer_sizes"])
        else:
            hidden_layers = [v for k, v in sorted(self.params.items()) if k.startswith("neurons_layer_")]

        if not hidden_layers:
            raise ValueError("MLPRegressor requires at least one hidden_layer_sizes parameter")

        # forbidden = {"width_scale", "decay_scale", "resource", "r_ratio"}
        # clean_params = {k: v for k, v in self.params.items() if not k.startswith('neurons_layer_') and k not in forbidden}

        def _filter_to_valid_params(param_dict, cls):
            valid = inspect.signature(cls.__init__).parameters
            return {k: v for k, v in param_dict.items() if k in valid}
        
        clean_params = _filter_to_valid_params(self.params, MLPRegressor)
        clean_params['hidden_layer_sizes'] = tuple(hidden_layers)

        return MLPRegressor(**clean_params)

    def _initialize_kan_model(self, **kwargs):

        # Dynamically configure width based on input/output dimensions
        feature_dim = self.params.get("feature_dim")  # Placeholder, dynamically assigned later
        target_dim = self.params.get("target_dim")  # Placeholder, dynamically assigned later
        hidden_layers = self.params.get("hidden_layers")  # Default: 2 hidden layers with 2 nodes each
        
        width = [feature_dim] + hidden_layers + [target_dim]
        
        # Only keep the parameters that KAN explicitly requires
        kan_params = {
            "width": width,
            "grid": self.params.get("grid"),
            "k": self.params.get("k"),
            "mult_arity": self.params.get("mult_arity", 2),
            "noise_scale": self.params.get("noise_scale", 0.3),
            "base_fun": self.params.get("base_fun", 'silu'),
            "symbolic_enabled": self.params.get("symbolic_enabled", True),
            "affine_trainable": self.params.get("affine_trainable", False),
            "grid_eps": self.params.get("grid_eps", 0.02),
            "grid_range": self.params.get("grid_range", [-1, 1]),
            "sp_trainable": self.params.get("sp_trainable", True),
            "sb_trainable": self.params.get("sb_trainable", True),
            # "seed": self.params.get("seed", 1),
            "save_act": self.params.get("save_act", True),
            "sparse_init": self.params.get("sparse_init", False),
            "auto_save": self.params.get("auto_save", False),
            "ckpt_path": self.params.get("ckpt_path", './model'),
            # "state_id": self.params.get("state_id", 0),
            # "round": self.params.get("round", 0),
            "device": device
        }

        return KAN(**kan_params).to(device)
    
    def _initialize_fast_kan_model(self, **kwargs):
        
        # Dynamically configure layers_hidden based on input/output dimensions
        feature_dim = self.params.get("feature_dim")  # Placeholder, dynamically assigned later
        target_dim = self.params.get("target_dim")  # Placeholder, dynamically assigned later
        hidden_layers = self.params.get("hidden_layers")  # Default: 2 hidden layers with 2 nodes each
    
        width = [feature_dim] + hidden_layers + [target_dim]

        kan_params = {
            "layers_hidden": width,
            "grid_min": self.params.get("grid_min", -2.0),
            "grid_max": self.params.get("grid_max", 2.0),
            "num_grids": self.params.get("num_grids"),
            "use_base_update": self.params.get("use_base_update", True),
            "base_activation": self.params.get("base_activation", torch.nn.functional.silu),
            "spline_weight_init_scale": self.params.get("spline_weight_init_scale", 0.1),
        }
    
        return FastKAN(**kan_params).to(device)

    def fit(self, X, y):
        ### MaternKernel as an example, people can add more in the future follow this way
        if self.model_name == 'GP_gpu':
            # covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[1], lengthscale_constraint=Interval(0.1, 4.0)))
            outcome_transform = Standardize(m=1)
            self.model = SingleTaskGP(torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1), outcome_transform=outcome_transform)#, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)
        elif self.model_name in ['KAN', 'FastKAN']:
            self.params["feature_dim"] = X.shape[1]
            self.params["target_dim"] = 1 if len(y.shape) == 1 else y.shape[1]
            logging.info("KAN params: %s", self.params)
            if self.model_name == 'KAN':
                # self.model = self._initialize_kan_model(self.params)
                X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
                dataset = create_dataset_from_data(X_tensor, y_tensor)
                self.model.fit(dataset, opt="LBFGS", steps=self.params.get("steps", 50), lamb=self.params.get("lamb", 0.002), lamb_entropy=self.params.get("lamb_entropy", 2.0))
            elif self.model_name == 'FastKAN':
                # self.model = self._initialize_fast_kan_model(self.params)
                X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
                y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.get("lr"))
                self.model.to(device)
                loss_fn = torch.nn.MSELoss()
                for step in range(self.params.get("steps", 500)):
                    optimizer.zero_grad()
                    predictions = self.model(X_tensor)
                    loss = loss_fn(predictions, y_tensor)
                    loss.backward()
                    optimizer.step()
        else:
            self.model.fit(X, y)

    def predict(self, X):
        ### MaternKernel as an example, people can add more in the future follow this way
        if self.model_name == 'GP_gpu':
            model_pred = self.model.posterior(torch.tensor(X, dtype=torch.float32))
            mean = model_pred.mean.detach().cpu().numpy().reshape(-1)
            # std = torch.sqrt(model_pred.variance).detach().cpu().numpy().reshape(-1)
            return mean
        elif self.model_name == 'KAN':
            return self.model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy().flatten()
        elif self.model_name == 'FastKAN':
            with torch.no_grad():
                return self.model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy().flatten()
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        positive_proba = np.clip(np.asarray(self.model.predict(X)).reshape(-1), 0.0, 1.0)
        return np.column_stack((1.0 - positive_proba, positive_proba))
    
    def predict_std(self, X):
        # Optional hook for evaluation: per-point predictive std if available
        if self.model_name == 'GP_cpu':
            # sklearn GPR supports return_std.
            _, std = self.model.predict(X, return_std=True)
            return std
        if self.model_name == 'GP_gpu':
            post = self.model.posterior(torch.tensor(X, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            return torch.sqrt(post.variance).detach().cpu().numpy().reshape(-1)
        # Other models usually do not expose native predictive standard deviation.
        raise AttributeError("predict_std not available for this model")
    
    def predict_var(self, X):
        std = self.predict_std(X)
        return std ** 2

def hyperparameter_optimization(
    model_name,
    X_train,
    y_train,
    cls=False,
    n_trials=20,
    cv_n_splits=5,
    alpha=0.2,
    expect_N: int=3,
    eta: int=3
):
    
    n_samples, feature_dim = X_train.shape
    target_dim = 1 if y_train.ndim == 1 else y_train.shape[1]

    # The budget is adaptively provided by estimate_budget.
    max_resource, d_eff, d_all = estimate_budget(X_train, y_train)   # 1 ≤ resource ≤ max_resource
    logging.info(f'{model_name} max_resource: {max_resource}')
    if max_resource < 10:          # Guard for very small datasets.
        max_resource = 10
    ratio_global = min(n_samples / max_resource, 1.0)

    # pruner = HyperbandPruner(min_resource=10,max_resource=max_resource,reduction_factor=eta)
    # pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    base = PercentilePruner(percentile=25, n_startup_trials=5, n_warmup_steps=3)
    pruner = optuna.pruners.PatientPruner(base, patience=2)

    study = optuna.create_study(direction="minimize",pruner=pruner,sampler=optuna.samplers.TPESampler(n_startup_trials=5, multivariate=True, warn_independent_sampling=False))  
    
    # study = optuna.create_study(direction='minimize', pruner=pruner, sampler=optuna.samplers.TPESampler(n_startup_trials=3, multivariate=True))

    def objective(trial):
        params = {}
        # if model_name in {"XGBoost", "LightGBM", "RandomForest", "ExtraTreesRegressor"}:
        #     params.setdefault("n_jobs", 1)
        resource = max_resource

        # ---------------- Tree and boosting model families ----------------
        if model_name == "RandomForest":
            r = trial.suggest_int("resource", 1, min(expect_N*n_samples, max_resource))
            # r = trial.suggest_int("resource", 1, max_resource)
            resource = r
            r_ratio = r / max_resource
            params["n_estimators"] = max(10, r)            # Use r directly for iteration-count parameters.
            params["max_depth"] = trial.suggest_int("max_depth", 1, int(1 + (64 - 1) * r_ratio))

        elif model_name in ["XGBoost", "LightGBM"]:
            r = trial.suggest_int("resource", 1, min(expect_N*n_samples, max_resource))
            # r = trial.suggest_int("resource", 1, max_resource)
            resource = r
            r_ratio = r / max_resource
            params["n_estimators"] = max(10, r)
            params["max_depth"] = trial.suggest_int("max_depth", 1, int(1 + (64 - 1) * r_ratio))
            lr_lower, lr_upper = 1e-4, 1.0
            # params["learning_rate"] = trial.suggest_loguniform("learning_rate", lr_lower, lr_upper)
            params["learning_rate"] = trial.suggest_float("learning_rate", lr_lower, lr_upper, log=True)
            if model_name == "LightGBM":
                params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, int(20 + (200 - 20) * r_ratio))

        elif model_name == "CatBoost":
            r = trial.suggest_int("resource", 1, min(expect_N*n_samples, max_resource))
            # r = trial.suggest_int("resource", 1, max_resource)
            resource = r
            r_ratio = r / max_resource
            params["iterations"] = max(100, r)
            # params["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-4, 0.5)
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 0.3, log=True)
            params["depth"] = trial.suggest_int("depth", 4, min(12, feature_dim))
            # params["l2_leaf_reg"] = trial.suggest_loguniform("l2_leaf_reg", 1e-2, 10.0)
            params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True)
            # params["bagging_temperature"] = trial.suggest_uniform("bagging_temperature", 0.0, 1.0)
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
            # params["random_strength"] = trial.suggest_uniform("random_strength", 0.0, 1.0)
            params["random_strength"] = trial.suggest_float("random_strength", 0.0, 1.0)
            params["boosting_type"] = trial.suggest_categorical("boosting_type", ["Ordered", "Plain"])

        elif model_name in ["GradientBoostingRegressor","AdaBoostRegressor","ExtraTreesRegressor",]:
            r = trial.suggest_int("resource", 1, min(expect_N*n_samples, max_resource))
            # r = trial.suggest_int("resource", 1, max_resource)
            resource = r
            r_ratio = r / max_resource
            params["n_estimators"] = max(10, r)
            if model_name in ["GradientBoostingRegressor", "AdaBoostRegressor"]:
                # params["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-4, 1.0)
                params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 0.3, log=True)
            if model_name in ["GradientBoostingRegressor", "ExtraTreesRegressor"]:
                params["max_depth"] = trial.suggest_int("max_depth", 1, int(1 + (64 - 1) * r_ratio))

        elif model_name == "MLPRegressor":
            r = trial.suggest_int("resource", 1, min(expect_N*n_samples, max_resource))
            # r = trial.suggest_int("resource", 1, max_resource)
            resource = r
            r_ratio = r / max_resource
            # params["max_iter"] = int(max(200, r))
            params["max_iter"] = 100
            
            # max_layers_by_samples = int(np.floor(np.log(max(n_samples,2)))) + 1
            base_layers = 2 + int(np.log1p(d_eff)*r_ratio)
            n_layers = int(np.clip(base_layers, 2, np.inf))
            
            hidden_sizes = []
            first_width = trial.suggest_int("neurons_layer_0", 
                                            max(128, int(d_eff * 8 * np.log1p(d_eff))), 
                                            max(1024, int(d_all * 8 * np.log1p(d_all))), 
                                            log=True)
            hidden_sizes.append(first_width)

            last_width = first_width
            for i in range(1, n_layers):
                width = trial.suggest_int(f"neurons_layer_{i}", 32, last_width, log=False)
                hidden_sizes.append(width)
                last_width = width

            # width_scale = trial.suggest_float("width_scale", 0.5, 3.0, log=True)
            # decay_scale = trial.suggest_float("decay_scale", 0.75, 1.25, log=True)

            # first_width = int(round((64 + 32 * d_eff) * width_scale))
            # first_width = max(128, first_width)

            # hidden_sizes = []
            # for i in range(n_layers):
            #     units = int(max(8, first_width * (decay_scale ** i)))
            #     hidden_sizes.append(units)
            #     params[f"neurons_layer_{i}"] = trial.suggest_int(f"neurons_layer_{i}", units, units)

            max_batch_size = min(256, int(n_samples * 0.8)) # Protective upper bound.
            if max_batch_size < 32:
                # For very small datasets, use a smaller fixed batch size or the full batch.
                possible_batch_sizes = [16, 32] 
                possible_batch_sizes = [bs for bs in possible_batch_sizes if bs <= max_batch_size]
                if not possible_batch_sizes: possible_batch_sizes = [max_batch_size]
            else:
                possible_batch_sizes = [32, 64, 128, 256]
                possible_batch_sizes = [bs for bs in possible_batch_sizes if bs <= max_batch_size]
            
            params["batch_size"] = trial.suggest_categorical("batch_size", possible_batch_sizes)

            params["hidden_layer_sizes"] = tuple(hidden_sizes)
            params["activation"] = trial.suggest_categorical("activation", ["tanh", "relu"])
            # params["solver"] = trial.suggest_categorical("solver", ["adam", "sgd"])
            params["solver"] = "adam"
            params["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
            params["learning_rate"] = trial.suggest_categorical("learning_rate_schedule", ["constant", "adaptive"])
            
            # params["alpha"] = trial.suggest_loguniform("alpha", 1e-4, 0.3)
            params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            if n_samples < 50:
                params["early_stopping"] = False
            else:
                params["early_stopping"] = True
            params["n_iter_no_change"] = 15

        # ---------------- Models without incremental capacity; reuse ratio_global ----------------
        elif model_name == "SVR":
            C_upper, eps_upper = 1e3, 1e1
            # params["C"] = trial.suggest_loguniform("C", 1e-2, C_upper * ratio_global)
            # params["epsilon"] = trial.suggest_loguniform("epsilon", 1e-3, eps_upper * ratio_global)
            params["C"] = trial.suggest_float("C", 1e-2, C_upper * ratio_global, log=True)
            params["epsilon"] = trial.suggest_float("epsilon", 1e-3, eps_upper * ratio_global, log=True)
    
        elif model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            # params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e3)
            params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e3, log=True)
            if model_name == 'ElasticNet':
                # params['l1_ratio'] = trial.suggest_uniform('l1_ratio', 0, 1)
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        elif model_name == 'KNeighborsRegressor':
            n_neighbors_upper = min(n_samples // 2, 100)
            params['n_neighbors'] = trial.suggest_int('n_neighbors', 1, n_neighbors_upper)

        elif model_name == 'DecisionTreeRegressor':
            max_depth_upper = int(1 + (64 - 1) * ratio_global)
            params['max_depth'] = trial.suggest_int('max_depth', 1, max_depth_upper)

        elif model_name == 'KRR':
            # params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e3)
            # params['gamma'] = trial.suggest_loguniform('gamma', 1e-5, 1e1)
            params["alpha"] = trial.suggest_float("alpha", 1e-5, 1e3, log=True)
            params["gamma"] = trial.suggest_float("gamma", 1e-5, 1e1, log=True)

        elif model_name == 'GP_cpu':
            # Compute the dynamic factor.
            sample_ratio = ratio_global
            dim_ratio = np.sqrt(feature_dim + 1)

            # Dynamic range for length scale.
            length_scale_lower = max(0.1, feature_dim / (n_samples + 10))  # Avoid overfitting.
            length_scale_upper = min(10 * feature_dim, 50)  # Avoid excessive smoothing.
            # length_scales = np.array([trial.suggest_loguniform(f'length_scale_{i}', length_scale_lower, length_scale_upper) for i in range(feature_dim)])
            length_scales = np.array([trial.suggest_float(f"length_scale_{i}", length_scale_lower, length_scale_upper, log=True) for i in range(feature_dim)])

            # Dynamic range for noise level.
            noise_level_lower = 1e-3  # Ensure noise is not completely ignored.
            noise_level_upper = 0.5 * (1 - sample_ratio) + 1e-2  # Reduce noise when more samples are available.
            # noise_level = trial.suggest_loguniform('noise_level', noise_level_lower, noise_level_upper)
            noise_level = trial.suggest_float("noise_level", noise_level_lower, noise_level_upper, log=True)
                
            # Dynamic range for the constant kernel.
            constant_value_lower = 0.1
            constant_value_upper = 10 / dim_ratio
            # constant_val = trial.suggest_loguniform('constant_value', constant_value_lower, constant_value_upper)
            constant_val = trial.suggest_float("constant_value", constant_value_lower, constant_value_upper, log=True)
                
            # Dynamic range for alpha regularization.
            alpha_lower = 1e-6
            alpha_upper = min(1.0 / feature_dim, 1e-1)
            # alpha_val = trial.suggest_loguniform('alpha', alpha_lower, alpha_upper)
            alpha_val = trial.suggest_float("alpha", alpha_lower, alpha_upper, log=True)
        
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
            kernel = ConstantKernel(constant_value=constant_val, constant_value_bounds=(constant_value_lower, constant_value_upper)) * \
                     Matern(length_scale=length_scales, nu=2.5) + \
                     WhiteKernel(noise_level=noise_level, noise_level_bounds=(noise_level_lower, noise_level_upper))
        
            params['kernel'] = kernel
            params['alpha'] = alpha_val

        elif model_name == 'KAN':
            params["feature_dim"] = trial.suggest_int("feature_dim", feature_dim, feature_dim)
            params["target_dim"] = trial.suggest_int("target_dim", target_dim, target_dim)
            if n_samples < 500:
                kan_hidden_layer_options = [[2], [4], [2, 2]]
            else:
                kan_hidden_layer_options = [[2], [4], [8], [2, 2], [4, 4]]
            params['hidden_layers'] = trial.suggest_categorical('hidden_layers', kan_hidden_layer_options)
            params['grid'] = trial.suggest_int('grid', 3, 10)
            params['k'] = trial.suggest_int('k', 2, 3)
            # params['lamb'] = trial.suggest_loguniform('lamb', 1e-4, 5e-2)
            params["lamb"] = trial.suggest_float("lamb", 1e-4, 5e-2, log=True)

        elif model_name == 'FastKAN':
            params["feature_dim"] = trial.suggest_int("feature_dim", feature_dim, feature_dim)
            params["target_dim"] = trial.suggest_int("target_dim", target_dim, target_dim)
            if n_samples < 500:
                fastkan_hidden_layer_options = [[2], [4], [2, 2]]
            else:
                fastkan_hidden_layer_options = [[2], [4], [8], [2, 2], [4, 4]]
            params['hidden_layers'] = trial.suggest_categorical('hidden_layers', fastkan_hidden_layer_options)
            if n_samples < 500:
                num_grids_lower, num_grids_upper = (3, 6)
            else:
                num_grids_lower, num_grids_upper = (4, 8)
            params['num_grids'] = trial.suggest_int('num_grids', num_grids_lower, num_grids_upper)
            # params['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
            params["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        else:
            params = {}

        model = SurrogateModel(model_name, params)

        splitter = StratifiedKFold(n_splits=cv_n_splits, shuffle=True) if cls else KFold(n_splits=cv_n_splits, shuffle=True)
            
        cv_scores, cv_gaps = [], []
        metric_name = "logloss" if cls else "rmse"
        for tr_idx, val_idx in splitter.split(X_train, y_train if cls else None):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            if model_name == "CatBoost" and np.allclose(y_tr, y_tr[0]):
                trial.report(float("inf"), step=trial.number)
                raise optuna.TrialPruned()

            model.fit(X_tr, y_tr)
            
            if cls:
                # prefer log‑loss if proba available
                if hasattr(model, 'predict_proba'):
                    proba_val = model.predict_proba(X_val)
                    proba_tr  = model.predict_proba(X_tr)
                    val_err = log_loss(y_val, proba_val, labels=np.unique(y_train))
                    tr_err  = log_loss(y_tr,  proba_tr,  labels=np.unique(y_train))
                else:
                    pred_val = model.predict(X_val)
                    pred_tr  = model.predict(X_tr)
                    val_err = 1.0 - accuracy_score(y_val, pred_val)
                    tr_err  = 1.0 - accuracy_score(y_tr,  pred_tr)
            else:
                pred_val = model.predict(X_val)
                pred_tr  = model.predict(X_tr)
                val_err = mean_squared_error(y_val, pred_val)
                tr_err  = mean_squared_error(y_tr,  pred_tr)

            cv_scores.append(val_err)
            cv_gaps.append(val_err - tr_err)

        mean_val, mean_gap = float(np.mean(cv_scores)), float(np.mean(cv_gaps))
        objective_value = mean_val + alpha * max(0.0, mean_gap)
        
        trial.report(objective_value, step=_DEFAULT_EARLY_STOP_ROUNDS)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        return objective_value

    ### for all kind of Gaussian like model, no need to do hyperparameter_optimization since cost can be high
    if model_name == 'GP_gpu':
        return {}
    
    # ----------- Run Optuna -----------
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    logging.info("hyperparameter optimization complete for %s", model_name)
    return study.best_trial.params
