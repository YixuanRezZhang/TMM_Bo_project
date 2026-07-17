from __future__ import annotations
from typing import Optional, Dict, Literal, Tuple
from scipy.stats import norm, iqr, spearmanr, rankdata, entropy as kl_entropy
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim
from scipy.special import erf

import numpy as np
import inspect
import numbers
import os, pickle, torch, pywt, ray
from scipy.stats import norm
import pygmo as pg
import logging
from src.evaluation import ClusterBootstrapSampler
from src.model_uncertainty import (
    aggregate_bootstrap_model_variance,
    calibrate_oob_noise_variance,
    kalman_fusion_weights,
    predict_mean_and_internal_variance,
)
from scipy.signal import savgol_filter

try:
    import faiss
    import igraph as ig
    import leidenalg
    from scipy.sparse import coo_matrix
except ImportError:
    faiss = None
    ig = None
    leidenalg = None
    coo_matrix = None

if torch.cuda.is_available():
    device = torch.device('cuda')
    import cuml, cupy
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.cluster import KMeans as cuKMeans
    from cuml.neighbors import NearestNeighbors as cuKNN
else:
    try:
        import hdbscan
    except ImportError:
        hdbscan = None
    device = torch.device('cpu')

# All internal acquisition subproblems are formulated as maximization tasks.

@ray.remote
def model_predict(model, X_candidate_ref, sg_model, start, end):
    # Fetch data from the object reference and slice the requested batch.
    X_candidate_batch = X_candidate_ref[start:end]

    mean, internal_variance = predict_mean_and_internal_variance(
        model, X_candidate_batch, model_name=sg_model
    )
    if internal_variance is None:
        internal_variance = np.zeros_like(mean, dtype=float)
    return mean, internal_variance

@ray.remote
def compute_pareto_front_batch(points_ref, start, end):
    batch = points_ref[start:end]
    ud = pg.fast_non_dominated_sorting(batch)
    return ud[0][0]

def compute_hv_contributions(pareto_points, reference_point):
    
    P = np.asarray(pareto_points, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] == 0:
        return np.array([], dtype=np.float64)

    # Drop non-finite rows before hypervolume evaluation.
    P = P[np.all(np.isfinite(P), axis=1)]
    if P.shape[0] == 0:
        return np.array([], dtype=np.float64)

    rp = np.asarray(reference_point, dtype=np.float64)
    # Ensure the reference point strictly dominates the Pareto points in minimization space.
    rp = np.nextafter(rp, np.inf)

    hv = pg.hypervolume(P)
    return hv.contributions(rp)

# ----------------------------- helpers -----------------------------

def _dim_reduction(
    X: np.ndarray,
    max_dim: int = 50,
    method: Literal["jp", "pca"] = "jp",
    random_state: int = 0,
) -> np.ndarray:
    """Reduce dimension if `X.shape[1] > max_dim` to mitigate distance concentration."""
    d = X.shape[1]
    if d <= max_dim:
        return X  # no-op

    if method == "jp":
        k_opt = max(max_dim, johnson_lindenstrauss_min_dim(X.shape[0], eps=0.1))
        proj = GaussianRandomProjection(n_components=k_opt, random_state=random_state)
        return proj.fit_transform(X)
    else:  # PCA
        pca = PCA(n_components=max_dim, random_state=random_state)
        return pca.fit_transform(X)

def _downsample_1d_with_inverse(x, bins=None):
    """
    Downsample a path-ordered 1D sequence to `bins` bin means,
    and return an interpolation function that maps values back to the original length.
    """
    x = np.asarray(x, float)
    L = x.size
    if (bins is None) or (L <= bins):
        def inv_fn(y_ds):
            return np.asarray(y_ds, float)
        return x, L, inv_fn

    edges = np.linspace(0, L, bins + 1, dtype=int)
    counts = np.diff(edges)
    ds = (np.add.reduceat(x, edges[:-1]) / np.maximum(1, counts)).astype(float, copy=False)

    # Use bin centers as sample locations so interpolation can recover the original length.
    pos_ds   = 0.5 * (edges[:-1] + edges[1:])  # (bins,)
    pos_full = np.arange(L, dtype=float)

    def inv_fn(y_ds):
        y_ds = np.asarray(y_ds, float)
        return np.interp(pos_full, pos_ds, y_ds)

    return ds, bins, inv_fn


def _wav_trend_1d(y_seq, wavelet, min_points_per_level, energy_threshold):
    """
    Single path: wavelet decomposition -> keep low-frequency bands up to the energy threshold -> inverse transform to get the trend.
    Returns: trend, level, cut_idx.
    """
    y_seq = np.asarray(y_seq, float)
    w = pywt.Wavelet(wavelet)
    Lr = len(y_seq)
    max_level = pywt.dwt_max_level(Lr, w.dec_len)
    # Keep at least `min_points_per_level` points at the coarsest level.
    level = max(1, min(max_level, int(np.floor(np.log2(Lr / max(1, min_points_per_level))))))

    coeffs  = pywt.wavedec(y_seq, wavelet=wavelet, level=level)  # [cA_L, cD_L, ..., cD_1]
    bands_E = np.array([np.sum(c**2) for c in coeffs], dtype=float)
    E_total = float(bands_E.sum()) + 1e-12
    cum     = np.cumsum(bands_E)                                 # Cumulative energy from low-frequency to high-frequency bands.
    cut_idx = int(np.searchsorted(cum, float(energy_threshold) * E_total, side="left"))
    cut_idx = np.clip(cut_idx, 0, len(coeffs) - 1)

    keep = [c if i <= cut_idx else np.zeros_like(c) for i, c in enumerate(coeffs)]
    trend = pywt.waverec(keep, wavelet)[:Lr]
    return trend, int(level), int(cut_idx)


def _spectral_flatness(residual_seq, eps=1e-12):
    """One-dimensional spectral flatness of residuals after removing the DC component. Returns [0, 1]; larger values are closer to white noise."""
    r = np.asarray(residual_seq, float)
    if r.size < 8 or np.allclose(r.var(), 0.0):
        return 1.0
    R = np.fft.rfft(r - r.mean())
    P = (R.real**2 + R.imag**2).astype(float)
    if P.size <= 2:
        return 1.0
    P = P[1:]  # drop DC
    gm = np.exp(np.mean(np.log(P + eps)))
    am = np.mean(P + eps)
    return float(np.clip(gm / am, 0.0, 1.0))


def auto_select_energy_threshold(
    mean,
    Z,
    idx_pc1,
    idx_pc2,
    *,
    wavelet='db4',
    min_points_per_level=32,
    bins=1024,
    tau_grid=None,
    lambda_smooth=0.05,  # Small penalty for overly large thresholds; set to 0 to disable it.
):
    """
    Return (tau_star, diag).
      - tau_star: adaptive energy_threshold.
      - diag: R2/SF/rho/score diagnostics for each tau on the grid.
    Depends on _downsample_1d_with_inverse and _wav_trend_1d.
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.55, 0.85, 7)  # 0.55..0.85 step 0.05

    y = np.asarray(mean, float).reshape(-1)
    Z = np.asarray(Z, float)
    pc1, pc2 = Z[:, 0], Z[:, 1]
    idx_pc1 = np.asarray(idx_pc1)
    idx_pc2 = np.asarray(idx_pc2)
    N = y.size

    # Helper: path downsampling followed by inverse interpolation.
    def _prep(seq, idx):
        full = seq[idx]
        ds, _, inv = _downsample_1d_with_inverse(full, bins)
        return full, ds, inv

    y1_full, y1_ds, inv1 = _prep(y, idx_pc1)
    y2_full, y2_ds, inv2 = _prep(y, idx_pc2)

    # Base design matrix for linear fitting in the original sample order.
    X = np.c_[np.ones(N), pc1, pc2]

    def _r2_on_trend(trend):
        beta, *_ = np.linalg.lstsq(X, trend, rcond=None)
        y_hat = X @ beta
        sst = float(np.sum((trend - trend.mean())**2)) + 1e-12
        sse = float(np.sum((trend - y_hat)**2))
        return float(np.clip(1.0 - sse/sst, 0.0, 1.0))

    records = []
    for tau in tau_grid:
        # Reconstruct trends on both ordered paths.
        t1_ds, _, _ = _wav_trend_1d(y1_ds, wavelet, min_points_per_level, tau)
        t2_ds, _, _ = _wav_trend_1d(y2_ds, wavelet, min_points_per_level, tau)
        t1 = inv1(t1_ds); t2 = inv2(t2_ds)

        # Scatter path trends back to the original order and average them.
        trend1 = np.empty_like(y); trend1[idx_pc1] = t1
        trend2 = np.empty_like(y); trend2[idx_pc2] = t2
        trend  = 0.5 * (trend1 + trend2)

        # A) Trend linearity.
        R2 = _r2_on_trend(trend)

        # B) Residual whitening on both paths.
        r1 = y1_full - t1
        r2 = y2_full - t2
        SF = 0.5 * (_spectral_flatness(r1) + _spectral_flatness(r2))

        # C) Two-path consistency measured as correlation in the original order.
        if np.std(trend1) > 1e-12 and np.std(trend2) > 1e-12:
            rho = float(np.corrcoef(trend1, trend2)[0, 1])
        else:
            rho = 1.0
        rho = float(np.clip(rho, 0.0, 1.0))

        # Harmonic mean penalizes weak components; add a mild smoothness penalty.
        hmean = 3.0 / (1.0/(R2+1e-12) + 1.0/(SF+1e-12) + 1.0/(rho+1e-12))
        score = hmean - lambda_smooth * float(tau)
        records.append((float(tau), R2, SF, rho, hmean, score))

    rec = max(records, key=lambda x: x[-1])
    tau_star = float(rec[0])

    diag = {
        "tau_star": tau_star,
        "grid": [{"tau": t, "R2": R2, "SF": SF, "rho": rho, "hmean": hm, "score": sc}
                 for (t, R2, SF, rho, hm, sc) in records]
    }
    return tau_star, diag

def compute_structure_with_rigidity(
    mean,
    Z,
    idx_pc1,
    idx_pc2,
    *,
    wavelet='db4',
    energy_threshold='auto',       # float in (0,1] or 'auto': wavelet de-texturing threshold.
    min_points_per_level=32,
    bins=1024,                     # Bin ordered paths for speed; None disables downsampling.
    tau_grid=None,                 # Used when energy_threshold='auto'.
    lambda_smooth=0.05,            # Used only by auto_select_energy_threshold if that implementation needs it.
    # ---- Aggregation weights, applied as exponents to [0, 1] scores. ----
    alpha=0.5,     # Low-frequency energy ratio S_low.
    gamma=0.5,     # Curvature-to-gradient compression score S_curv.
    rho_pow=0.03,   # Light weight for path consistency rho.
    sf_pow=0.03,    # Light weight for residual spectral flatness SF.
    return_diagnostics=True,
):
    """
    Simple structural score without the S_smooth smoothness functional.
      It uses four quantities: S_low (low-frequency energy ratio), S_curv (curvature-to-gradient compression), SF (residual spectral flatness), and rho (two-path consistency).

    Subscores, all in [0, 1]:
      - S_low: low-frequency energy ratio averaged over both paths; low-frequency dominance indicates a simpler surface.
      - S_curv: curvature-to-gradient compression from a global quadratic approximation, kappa -> 1/(1+kappa).
      - SF and rho: residual spectral flatness and two-path trend consistency, used as lightly weighted robustness terms.

    External dependencies:
      - auto_select_energy_threshold(...)
      - _downsample_1d_with_inverse(...)
      - _wav_trend_1d(...)
      - _spectral_flatness(...)
    """
    import numpy as np

    y = np.asarray(mean, float).reshape(-1)
    Z = np.asarray(Z, float)
    idx_pc1 = np.asarray(idx_pc1)
    idx_pc2 = np.asarray(idx_pc2)

    N = y.size
    assert Z.shape == (N, 2), "Z must have shape [N, 2]"
    assert idx_pc1.shape == (N,) and idx_pc2.shape == (N,)

    eps = 1e-12

    # ===== Flat-surface guard =====
    if float(np.var(y)) < 1e-12:
        detail = dict(
            is_flat=True,
            struct_score=0.0,
            S_low=0.0, S_low_pc1=0.0, S_low_pc2=0.0,
            S_curv=0.0, kappa=0.0,
            SF=0.0, rho=0.0,
            # struct_score=1.0,
            # S_low=1.0, S_low_pc1=1.0, S_low_pc2=1.0,
            # S_curv=1.0, kappa=0.0,
            # SF=1.0, rho=1.0,
            energy_threshold=float(energy_threshold) if energy_threshold != 'auto' else None,
            min_points_per_level=int(min_points_per_level),
            bins_eff_pc1=None, bins_eff_pc2=None,
            tau_auto=None,
            Z_mean=None, Z_std=None,
            alpha=float(alpha), gamma=float(gamma),
            rho_pow=float(rho_pow), sf_pow=float(sf_pow),
        )
        return (0.0, detail) if return_diagnostics else (0.0, None)

    # ===== Select the wavelet threshold used for de-texturing. =====
    tau_auto_diag = None
    if isinstance(energy_threshold, str) and energy_threshold.lower() == 'auto':
        tau_star, tau_auto_diag = auto_select_energy_threshold(
            mean=y, Z=Z, idx_pc1=idx_pc1, idx_pc2=idx_pc2,
            wavelet=wavelet, min_points_per_level=min_points_per_level,
            bins=bins, tau_grid=tau_grid, lambda_smooth=lambda_smooth  # Remove this argument if the implementation does not need it.
        )
        tau = tau_star
    else:
        tau = float(energy_threshold)

    # ===== PC1 path: wavelet de-texturing plus low-frequency energy ratio. =====
    y1_full = y[idx_pc1]
    y1_full_c = y1_full - y1_full.mean()
    y1_ds, L1_ds, inv1 = _downsample_1d_with_inverse(y1_full, bins)
    t1_ds, lev1, cut1  = _wav_trend_1d(y1_ds, wavelet, min_points_per_level, tau)
    trend1 = inv1(t1_ds)
    r1 = y1_full - trend1
    E_tot1  = float(np.sum(y1_full_c**2)) + eps
    E_det1  = float(np.sum((r1 - r1.mean())**2))
    E_low1  = max(E_tot1 - E_det1, 0.0)
    S_low_pc1 = float(np.clip(E_low1 / E_tot1, 0.0, 1.0))

    # ===== PC2 path: wavelet de-texturing plus low-frequency energy ratio. =====
    y2_full = y[idx_pc2]
    y2_full_c = y2_full - y2_full.mean()
    y2_ds, L2_ds, inv2 = _downsample_1d_with_inverse(y2_full, bins)
    t2_ds, lev2, cut2  = _wav_trend_1d(y2_ds, wavelet, min_points_per_level, tau)
    trend2 = inv2(t2_ds)
    r2 = y2_full - trend2
    E_tot2  = float(np.sum(y2_full_c**2)) + eps
    E_det2  = float(np.sum((r2 - r2.mean())**2))
    E_low2  = max(E_tot2 - E_det2, 0.0)
    S_low_pc2 = float(np.clip(E_low2 / E_tot2, 0.0, 1.0))

    # ===== Robustness terms: path consistency and residual spectral flatness. =====
    trend_back1 = np.empty_like(y); trend_back1[idx_pc1] = trend1
    trend_back2 = np.empty_like(y); trend_back2[idx_pc2] = trend2
    rho = float(np.corrcoef(trend_back1, trend_back2)[0, 1]) if (np.std(trend_back1) > 1e-12 and np.std(trend_back2) > 1e-12) else 1.0
    rho = float(np.clip(rho, 0.0, 1.0))
    SF = 0.5 * (_spectral_flatness(r1) + _spectral_flatness(r2))
    SF = float(np.clip(SF, 0.0, 1.0))

    # ===== Merge the global trend by averaging both paths. =====
    y_trend = 0.5 * (trend_back1 + trend_back2)

    # ===== Standardize Z and center y_trend for curvature estimation. =====
    Z_mu = Z.mean(axis=0, keepdims=True)
    Z_sd = Z.std(axis=0, keepdims=True) + eps
    Z_std = (Z - Z_mu) / Z_sd
    y0 = y_trend - y_trend.mean()

    # ===== (1) S_low: low-frequency energy ratio. =====
    S_low = float(np.clip(0.5 * (S_low_pc1 + S_low_pc2), 0.0, 1.0))

    # ===== (2) S_curv: curvature-to-gradient compression from a global quadratic approximation. =====
    z1, z2 = Z_std[:, 0], Z_std[:, 1]
    Phi_quad = np.c_[np.ones(N), z1, z2, 0.5*z1*z1, z1*z2, 0.5*z2*z2]
    coef_quad, *_ = np.linalg.lstsq(Phi_quad, y0, rcond=None)
    b1, b2 = float(coef_quad[1]), float(coef_quad[2])
    c11, c12, c22 = float(coef_quad[3]), float(coef_quad[4]), float(coef_quad[5])
    g = np.array([b1, b2], float)
    H = np.array([[c11, c12],
                  [c12, c22]], float)
    kappa = float(np.linalg.norm(H, 'fro') / (np.linalg.norm(g) + eps))
    S_curv = float(1.0 / (1.0 + kappa))

    # ===== Aggregate the score. =====
    # struct_score = (
    #     (max(S_low,  eps) ** alpha) *
    #     (max(S_curv, eps) ** gamma) *
    #     (max(SF,     eps) ** sf_pow) *
    #     (max(rho,    eps) ** rho_pow)
    # )
    # struct_score = (
    #     ((max(S_low,  eps) * alpha) +
    #     (max(S_curv, eps) * gamma)) *
    #     (max(SF,     eps) ** sf_pow) *
    #     (max(rho,    eps) ** rho_pow)
    # )
    struct_score = max(S_low,  eps)
    struct_score = float(np.clip(struct_score, 0.0, 1.0))

    if not return_diagnostics:
        return struct_score, None

    detail = dict(
        struct_score = struct_score,
        # ---- Core subscores. ----
        S_low   = float(S_low),
        S_low_pc1 = float(S_low_pc1),
        S_low_pc2 = float(S_low_pc2),
        S_curv  = float(S_curv),
        kappa   = float(kappa),
        SF      = float(SF),
        rho     = float(rho),
        # ---- Wavelet and path diagnostics. ----
        level_pc1    = lev1,
        level_pc2    = lev2,
        trend_cut_pc1= cut1,
        trend_cut_pc2= cut2,
        energy_threshold      = float(tau),
        min_points_per_level  = int(min_points_per_level),
        bins_eff_pc1 = int(L1_ds),
        bins_eff_pc2 = int(L2_ds),
        tau_auto = tau_auto_diag,
        is_flat  = False,
        # ---- Normalization parameters and weights. ----
        Z_mean = Z_mu.reshape(-1).tolist(),
        Z_std  = Z_sd.reshape(-1).tolist(),
        alpha=float(alpha), gamma=float(gamma),
        rho_pow=float(rho_pow), sf_pow=float(sf_pow),
    )
    return struct_score, detail



def ranks_best(vals, ties='min', larger_value_larger_order=True, normalize=True):
    """
    Convert a one-dimensional score vector to ranks, where rank 1 is the largest value.
    ties: 'average'|'min'|'max'|'dense'|'ordinal'
    When normalize=True, map ranks linearly to [-1, 1], where 1 is the largest value.
    """
    order_res = []
    for row_v in vals:
        v = np.asarray(row_v)
        # Spearman/Kendall convention: larger values rank earlier, so rank -v.
        r = rankdata(v if larger_value_larger_order else -v, method=ties)  # 1..N
        if normalize:
            n = len(r)
            r = ((r - 1) / (n - 1)-0.5)*2 if n > 1 else np.ones_like(r, dtype=float)
        order_res.append(r)
    order_res = np.array(order_res)
    return order_res

def _read_sigma_data2(pack: dict) -> float | None:
    val = pack.get('oob_mad_var_mean', None)
    if val is None:
        val = pack.get('oob_var_mean', None)
    if val is None:
        _vars = pack.get('oob_var_list', None)
        _ns   = pack.get('oob_n_list', None)
        if _vars is not None and _ns is not None and len(_vars) == len(_ns) and len(_vars) > 0:
            _vars = np.asarray(_vars, float)
            _ns   = np.asarray(_ns, float)
            w = np.where(_ns > 0, _ns, 0.0)
            if w.sum() > 0:
                val = float(np.sum(_vars * w) / np.sum(w))
    if val is None or not np.isfinite(val) or val < 0:
        return None
    return float(val)

def auto_k_density(n_samples: int, d: int,
                   k_min=5, k_max=40, c=4.0) -> int:
    k_n = np.sqrt(n_samples)
    k_d = np.log(d + 1) * c
    k = int((k_n + k_d) / 2)
    return int(np.clip(k, k_min, k_max))

def estimate_density_and_spread(
    X_new: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_new: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    k_density: Union[int, Literal['auto']] = 'auto',
    max_dim: int = 50,
    random_state: int = 0,
) -> Dict[str, float]:
    """Compute three complementary indicators **simultaneously**.

    Returns a dict with keys:
        rho_new   – mean k‑th NN distance (inverse density proxy)
        LDR       – local‑density ratio w.r.t training (if `X_train` given)
        coverage  – grid coverage rate (if `X_train` given)
        KL        – 2D‑KDE KL divergence (if `X_train` given)
        spread_y  – IQR (or σ) of targets y_new
    """
    
    if k_density == 'auto':
        k_density = auto_k_density(len(X_new), X_new.shape[1])
    assert k_density >= 2, f"k_density must be ≥ 2, got {k_density}"

    # ------------- Dimensionality reduction (shared) -------------------
    if X_train is not None:
        X_stack = np.vstack([X_new, X_train])
        X_low = _dim_reduction(X_stack, max_dim=max_dim, random_state=random_state)
        X_new_low, X_train_low = X_low[: len(X_new)], X_low[len(X_new) :]
    else:
        X_new_low = _dim_reduction(X_new, max_dim=max_dim, random_state=random_state)
        X_train_low = None

    # ------------- (i) Local density of new points ---------------------
    nn_new = NearestNeighbors(n_neighbors=min(k_density, len(X_new)-1)).fit(X_new)   #X_new_low
    dist_new, _ = nn_new.kneighbors(X_new)   #X_new_low
    rho_new = dist_new[:, -1].mean()  # larger → sparser

    metrics: Dict[str, float] = {"rho_new": float(rho_new)}

    # ------------- (ii) Indicators involving training data -------------
    if X_train_low is not None:
        nn_train = NearestNeighbors(n_neighbors=min(k_density, len(X_train)-1)).fit(X_train)  #X_train_low
        dist_train, _ = nn_train.kneighbors(X_train)  #X_train_low
        rho_train_med = np.median(dist_train[:, -1])
        metrics["LDR"] = float((1.0 / rho_new) / (1.0 / rho_train_med))  # density ratio

        # Coverage in 2D PCA grid
        if X_new_low.shape[1] >= 2:
            pca2 = PCA(n_components=2, random_state=random_state)
            Z_all = pca2.fit_transform(X_low)   #X_train_low
            Z_train = pca2.transform(X_train_low)
            Z_new = pca2.transform(X_new_low)

            grid_bins = 20
            lo, hi = np.percentile(np.vstack([Z_new, Z_train]), [1, 99], axis=0)
            def _grid(z):
                return np.floor((z - lo) / (hi - lo + 1e-9) * grid_bins).astype(int)
            cover_new = {tuple(x) for x in _grid(Z_new)}
            cover_train = {tuple(x) for x in _grid(Z_train)}
            metrics["coverage"] = float(len(cover_new) / (len(cover_train) or 1))

        # KL divergence (histogram KDE on PCA‑2D)
        bins = 30
        H_train, _ = np.histogramdd(Z_train, bins=bins, density=True)
        H_new, _ = np.histogramdd(Z_new, bins=bins, density=True)
        p, q = H_train.flatten() + 1e-12, H_new.flatten() + 1e-12
        p, q = p / p.sum(), q / q.sum()
        metrics["KL"] = float(kl_entropy(p, q))

    # ------------- (iii) Target spread ---------------------------------
    if y_new is not None:
        y_flat = y_new.ravel()
        spread = iqr(y_flat) or np.std(y_flat)
    else:
        spread = 0.0
    metrics["spread_y"] = float(spread)

    return metrics

def select_history_aware_diverse_batch(
    X_candidates: np.ndarray,
    base_scores: np.ndarray,
    batch_size: int,
    X_history: Optional[np.ndarray] = None,
    novelty_floor: float = 0.05,
    query_batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedily rank candidates against history and the current batch.

    At step t the score is

        base_i * [floor + (1-floor) * normalized_distance_i],

    where distance_i is the nearest standardized Euclidean distance to either
    a historically sampled point or a candidate already selected in this batch.
    """
    X_candidates = np.asarray(X_candidates, dtype=float)
    base_scores = np.asarray(base_scores, dtype=float).reshape(-1)
    if X_candidates.ndim != 2:
        raise ValueError("X_candidates must be a 2-D array.")
    if len(X_candidates) != len(base_scores):
        raise ValueError("base_scores must match the candidate count.")
    if not 0.0 <= novelty_floor <= 1.0:
        raise ValueError("novelty_floor must be between 0 and 1.")
    if batch_size <= 0 or len(X_candidates) == 0:
        return np.empty(0, dtype=int), np.empty(len(X_candidates), dtype=float)

    if X_history is None:
        X_history = np.empty((0, X_candidates.shape[1]), dtype=float)
    else:
        X_history = np.asarray(X_history, dtype=float)
        if X_history.ndim != 2:
            raise ValueError("X_history must be a 2-D array.")
        if X_history.shape[1] != X_candidates.shape[1]:
            raise ValueError(
                "X_history and X_candidates must have the same feature count."
            )

    all_points = np.vstack([X_candidates, X_history])
    for feature_idx in range(all_points.shape[1]):
        column = all_points[:, feature_idx]
        finite = np.isfinite(column)
        fill_value = float(np.median(column[finite])) if finite.any() else 0.0
        column[~finite] = fill_value
        all_points[:, feature_idx] = column

    center = np.median(all_points, axis=0)
    scale = np.std(all_points, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    all_points = (all_points - center) / scale
    candidate_points = all_points[:len(X_candidates)]
    history_points = all_points[len(X_candidates):]

    if len(history_points):
        history_nn = NearestNeighbors(n_neighbors=1).fit(history_points)
        history_distances = np.empty(len(candidate_points), dtype=float)
        for start in range(0, len(candidate_points), query_batch_size):
            end = min(start + query_batch_size, len(candidate_points))
            history_distances[start:end] = history_nn.kneighbors(
                candidate_points[start:end],
                return_distance=True,
            )[0][:, 0]
    else:
        history_distances = np.full(len(candidate_points), np.inf, dtype=float)

    base_scores = np.nan_to_num(
        base_scores, nan=0.0, posinf=0.0, neginf=0.0
    )
    base_scores = base_scores - min(float(np.min(base_scores)), 0.0)
    max_base_score = float(np.max(base_scores))
    if max_base_score > 0.0:
        base_scores = base_scores / max_base_score
    else:
        base_scores = np.ones_like(base_scores)

    selected = []
    remaining = np.ones(len(candidate_points), dtype=bool)
    nearest_occupied_distance = history_distances.copy()
    initial_history_distances = history_distances.copy()

    for _ in range(min(int(batch_size), len(candidate_points))):
        remaining_idx = np.flatnonzero(remaining)
        distances = nearest_occupied_distance[remaining_idx]
        finite_distances = distances[np.isfinite(distances)]
        positive_distances = finite_distances[finite_distances > 0.0]
        if positive_distances.size:
            distance_scale = float(np.percentile(positive_distances, 90))
            normalized_distance = np.clip(
                distances / max(distance_scale, 1e-12),
                0.0,
                1.0,
            )
        elif finite_distances.size:
            normalized_distance = np.zeros_like(distances)
        else:
            # With no history, the first point is chosen by acquisition score.
            normalized_distance = np.ones_like(distances)

        novelty = novelty_floor + (1.0 - novelty_floor) * normalized_distance
        greedy_scores = base_scores[remaining_idx] * novelty
        best_local = int(np.argmax(greedy_scores))
        best_idx = int(remaining_idx[best_local])
        selected.append(best_idx)
        remaining[best_idx] = False

        distance_to_selected = np.linalg.norm(
            candidate_points - candidate_points[best_idx],
            axis=1,
        )
        nearest_occupied_distance = np.minimum(
            nearest_occupied_distance,
            distance_to_selected,
        )

    return np.asarray(selected, dtype=int), initial_history_distances

def safe_cv(std, mean, rel_floor=0.05, abs_floor=1e-12):
    """
    Safe coefficient of variation: CV = std / (abs(mean) + tau), where tau is an adaptive buffer:
    τ = max(abs_floor, rel_floor * median(|mean|[>0]))
    """
    std  = np.asarray(std, float)
    mean = np.asarray(mean, float)
    # Reference scale after removing zeros and NaNs.
    m = np.abs(mean[np.isfinite(mean) & (np.abs(mean) > 0)])
    ref = np.median(m) if m.size else 0.0
    tau = max(abs_floor, rel_floor * ref)
    denom = np.abs(mean) + tau
    return std / denom


def mix_ratio_from_scores(
    scores,
    *,
    method: str = 'gs',          # 'gs' or 'entropy' or 'std'
    invert: bool = False,        # True returns 1 - metric for cases where sharper distributions should score higher.
    weights=None,                # Candidate-axis weights with shape [N] or [N, T]; None means uniform averaging.
    prob: str = 'clip',          # 'clip' | 'softplus' | 'softmax': convert unnormalized scores to non-negative probability weights.
    temperature: float = 1.0,    # Softmax temperature.
    keepdims: bool = False,      # True returns [1, 1, T] for direct broadcasting with [M, N, T].
    eps: float = 1e-9
):
    """
    Input: scores may have shape [M], [M, T], or [M, N, T]; values are unnormalized and may be negative.
    Output: a ratio vector with shape [T], or [1, 1, T] when keepdims=True, clipped to [0, 1].

    method='gs'      -> normalized Gini-Simpson:  (1 - ∑ p^2) / (1 - 1/M)
    method='entropy' -> normalized Shannon entropy:   H / log(M)
    All-zero columns along M fall back to a uniform distribution and return metric value 1.
    """
    x = np.asarray(scores, dtype=float)

    # Normalize shapes to [M, N, T].
    if x.ndim == 1:      # [M]
        x = x[:, None, None]
    elif x.ndim == 2:    # [M,T]
        x = x[:, None, :]
    elif x.ndim == 3:    # [M,N,T]
        pass
    else:
        raise ValueError("scores must have shape [M], [M,T], or [M,N,T].")

    M, N, T = x.shape

    # Convert scores to non-negative probability drafts p_raw >= 0.
    if prob == 'softmax':
        z = x / max(temperature, eps)
        z = z - np.max(z, axis=0, keepdims=True)   # Numerical stabilization.
        p_raw = np.exp(z)
    elif prob == 'softplus':
        p_raw = np.log1p(np.exp(x))
    elif prob == 'clip':
        p_raw = np.maximum(x, 0.0)
    else:
        raise ValueError("prob must be 'clip', 'softplus', or 'softmax'.")

    # Normalize along the model axis; all-zero columns become uniform probabilities.
    S = p_raw.sum(axis=0, keepdims=True)            # [1,N,T]
    zero_mask = (S <= eps)                          # [1,N,T]
    p = np.where(zero_mask, 1.0 / max(M, 1), p_raw / (S + eps))  # [M,N,T]，∑_m p = 1

    # Compute the disagreement metric for each (n, t); all metrics are in [0, 1].
    if method == 'gs':
        if M == 1:
            metric = np.ones((N, T))
        else:
            sum_sq = (p ** 2).sum(axis=0)                  # [N,T]
            metric = (1.0 - sum_sq) / (1.0 - 1.0 / M)      # Normalized Gini-Simpson.
    elif method in ('entropy', 'shannon'):
        if M == 1:
            metric = np.ones((N, T))
        else:
            H = -(p * np.log(p + eps)).sum(axis=0)         # [N,T]
            metric = H / np.log(M)                         # Normalized entropy.
    elif method == 'std':
        metric = np.clip(np.std(x, axis=0)/np.mean(x+eps, axis=0), 0, 1)
    else:
        raise ValueError("method must be 'gs' or 'entropy'.")

    # Aggregate over candidate axis N to shape [T].
    if N == 1 or weights is None:
        out = metric.mean(axis=0)                           # [T]
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:                                     # [N] -> [N,1]
            w = w[:, None]
        if w.shape != (N, T):
            if w.shape == (N, 1):
                w = np.repeat(w, T, axis=1)
            else:
                raise ValueError("weights must have shape [N] or [N,T].")
        w = np.maximum(w, 0.0)
        w = w / (w.sum(axis=0, keepdims=True) + eps)        # Normalize separately for each target t.
        out = (metric * w).sum(axis=0)                      # [T]

    if invert:
        out = 1.0 - out

    out = np.clip(out, 0.0, 1.0)                            # Numerical safety.
    if keepdims:
        return out.reshape(1, 1, T)                         # Enables broadcasting with [M, N, T].
    return out                                              # [T]


def _array_to_single_line(arr, *, precision=4):
    return np.array2string(
        np.asarray(arr),
        precision=precision,
        suppress_small=True,
        separator=", ",
        max_line_width=10**6,
    ).replace("\n", " ")


def _matrix_to_single_line(arr, *, precision=4):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return _array_to_single_line(arr, precision=precision)
    rows = [_array_to_single_line(row, precision=precision) for row in arr]
    return "[" + "; ".join(rows) + "]"


def _high_dim_preview(arr, *, precision=4, max_slices=3):
    arr = np.asarray(arr)
    preview = arr[:max_slices]
    lines = []
    for idx, block in enumerate(preview):
        if block.ndim == 1:
            block_str = _array_to_single_line(block, precision=precision)
        elif block.ndim == 2:
            block_str = _matrix_to_single_line(block, precision=precision)
        else:
            block_str = _array_to_single_line(block.reshape(-1), precision=precision)
        lines.append(f"[{idx}] {block_str}")
    if arr.shape[0] > preview.shape[0]:
        lines.append(f"... ({arr.shape[0] - preview.shape[0]} more slices)")
    return "\n".join(lines)


def _format_array_for_log(name, arr, *, precision=4, max_items_per_dim=(3, 6, 6)):
    arr = np.asarray(arr)

    if arr.ndim == 0:
        return f"{name}: scalar={arr.item():.{precision}g}"

    finite = arr[np.isfinite(arr)]
    if finite.size:
        stats = (
            f"min={np.min(finite):.{precision}g}, "
            f"max={np.max(finite):.{precision}g}, "
            f"mean={np.mean(finite):.{precision}g}, "
            f"std={np.std(finite):.{precision}g}"
        )
    else:
        stats = "all values are non-finite"

    header = f"{name}: shape={arr.shape}, {stats}"

    if arr.ndim == 1:
        return f"{header}\n{_array_to_single_line(arr, precision=precision)}"

    if arr.ndim == 2 and arr.size < 20:
        return f"{header}\n{_matrix_to_single_line(arr, precision=precision)}"

    if arr.ndim >= 3 and os.environ.get("BO_LOG_HIGH_DIM_PREVIEW", "0") == "1":
        max_slices = max_items_per_dim[0] if max_items_per_dim else 3
        return f"{header}, preview=enabled\n{_high_dim_preview(arr, precision=precision, max_slices=max_slices)}"

    return header


def _format_scalar_stats_for_log(name, arr, *, precision=4):
    arr = np.asarray(arr, dtype=float)
    return (
        f"{name}: shape={arr.shape}, mean={np.mean(arr):.{precision}g}, "
        f"std={np.std(arr):.{precision}g}, min={np.min(arr):.{precision}g}, "
        f"max={np.max(arr):.{precision}g}"
    )


def _log_nan_summary(**named_arrays):
    warnings = []
    for name, arr in named_arrays.items():
        nan_count = int(np.isnan(np.asarray(arr)).sum())
        if nan_count:
            warnings.append(f"{name}={nan_count}")
    if warnings:
        logging.warning("NaN detected in arrays: %s", ", ".join(warnings))



class AcquisitionFunction:
    def __init__(self, hpar=0.1):
        self.hpar = hpar

    # ---------------------- basic scalar A/Fs -------------------------
    def ucb(self, mean, std, hpara=None):
        return mean + (self.hpar if hpara is None else hpara) * std

    def ei(self, mean, std, y_best, hpara=None):
        imp = mean - y_best - (self.hpar if hpara is None else hpara)
        Z = imp / (std + 1e-9)
        return np.where(std == 0, 0.0, imp * norm.cdf(Z) + std * norm.pdf(Z))

    def pi(self, mean, std, y_best, hpara=None):
        imp = mean - y_best - (self.hpar if hpara is None else hpara)
        Z = imp / (std + 1e-9)
        return np.where(std == 0, 0.0, norm.cdf(Z))

    # --- Helper method for Monte Carlo EHVI Calculation ---
    def _calculate_mc_hvi(self, means, variances, corr_matrix, pareto_front, ref_point, model_scores, n_samples=64, ucb=False, select_region=None):

        n_candidates, n_obj = means.shape
        HV_values = np.zeros(n_candidates)
        
        # Calculate the base hypervolume of the current Pareto front
        front_max_point = np.max(pareto_front, axis=0)
        final_ref_point = np.maximum(front_max_point+1e-4, ref_point)*model_scores
        pareto_front = pareto_front*model_scores
        hv = pg.hypervolume(pareto_front)
        base_hv = hv.compute(final_ref_point)

        for i in range(n_candidates):
            stds = np.sqrt(variances[i, :])
            # Construct the full covariance matrix for this candidate
            cov_matrix = np.diag(stds) @ corr_matrix @ np.diag(stds)
            # Add a small identity matrix for numerical stability before Cholesky
            stable_cov = cov_matrix + np.eye(n_obj) * 1e-6
            try:
                chol_cov = np.linalg.cholesky(stable_cov)
            except np.linalg.LinAlgError:
                # If matrix is not positive definite, skip this candidate
                continue 

            # Draw n_samples from the multivariate normal distribution
            random_samples = np.random.randn(n_samples, n_obj)
            correlated_samples = means[i, :] + random_samples @ chol_cov.T
            if select_region is not None:
                correlated_samples = correlated_samples-select_region
            correlated_samples = np.minimum(correlated_samples, np.maximum(front_max_point+1e-4, ref_point)-1e-6)*model_scores
            
            # Calculate the HV improvement for each MC sample
            improvements = np.zeros(n_samples)
            for j in range(n_samples):
                combined_front = np.vstack([pareto_front, correlated_samples[j, :]])
                new_hv = pg.hypervolume(combined_front).compute(final_ref_point)
                improvements[j] = max(0, new_hv - base_hv)
            
            # The EHVI is the average of these potential improvements
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            if ucb:
                HV_values[i] = mean_improvement + self.hpar * std_improvement
            else:
                HV_values[i] = np.mean(improvements)

        return HV_values

    def compute_entropy(self, ensemble_var):
        entropy = 0.5 * np.log(2 * np.pi * np.e * ensemble_var)
        return entropy

    def adaptive_kappa(self, entropy, kappa_base=0.1, kappa_scale=0.5):
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-9)
        kappa = kappa_base + kappa_scale * entropy_norm
        return kappa

    def adaptive_weights(self, entropy, scale_factor=1.0):

        entropy_median = np.median(entropy)
        entropy_std = np.std(entropy) + 1e-9  # Avoid division by zero.
        
        # Use more exploration when entropy is above the median.
        w_ucb = 1 / (1 + np.exp(-(entropy - entropy_median) / (entropy_std * scale_factor)))
        w_ei = 1 - w_ucb
        return w_ucb, w_ei

    def hypervolume(self, points, reference_point=None, batch_size=20000):
    
        points = -points.astype(np.float32)
        points_ref = ray.put(points)
        if reference_point is None:
            m = np.max(points, axis=0)
            reference_point = np.nextafter(m, np.inf)
        else:
            reference_point = -reference_point

        num_points = len(points)
        batches = [(i, min(i + batch_size, num_points)) for i in range(0, num_points, batch_size)]

        # Step 1: Use Ray actor pool to compute local Pareto fronts
        logging.info('Start local Pareto calculation')
        ray_tasks = []
        for start, end in batches:
            ray_tasks.append(compute_pareto_front_batch.remote(points_ref, start, end))

        pareto_batch_indices = ray.get(ray_tasks)
        
        # Map batch indices back to global indices
        logging.info('Start global Pareto calculation')
        global_pareto_indices = []
        for i, batch_indices in enumerate(pareto_batch_indices):
            global_pareto_indices.extend(batch_indices + i * batch_size)
        logging.info('Finding true global Pareto front from candidates')
        global_pareto_indices = list(set(global_pareto_indices))  # Deduplicate indices
        global_pareto_points = points[global_pareto_indices]
        
        # Step 2: Compute hypervolume contributions for global Pareto front
        logging.info('start hyper volume calculation')
        
        # 1) Remove extreme and non-finite values.
        assert np.all(np.isfinite(global_pareto_points)), "Non-finite point in HV input"
        # 2) Check that the reference point strictly dominates the Pareto points.
        logging.info(
            _format_array_for_log(
                "reference_point_minus_max",
                reference_point - np.max(global_pareto_points, axis=0),
            )
        )
        assert np.all(global_pareto_points < reference_point), "Some point >= reference point (violates strictness)"

        hv_contributions = compute_hv_contributions(global_pareto_points, reference_point)
        all_hv_contributions = np.zeros(num_points)
        all_hv_contributions[global_pareto_indices] = hv_contributions
    
        # Step 4: Calculate distances for all points
        logging.info('start norm calculation')
        distances = np.empty(points.shape[0], np.float32)
        for s in range(0, points.shape[0], batch_size):
            b = points[s:s+batch_size]
            distances[s:s+batch_size] = np.linalg.norm(b - reference_point, axis=1)
        # distances = np.linalg.norm(points - reference_point, axis=1)

        final_recommand = all_hv_contributions + 0.1*distances
        
        logging.info('finish hyper volume!')
        
        return final_recommand


    def select_next(self, method, X_candidate, model_name_list, num_of_targets, model_path, batch_size=10, X_train=None, y_value=None, model_result=None, stack=False, select_region=None, diversity_method=False, alpha=0.5, optimization_goal='maximize', use_correlation=False, use_model_correlation=True, train_clsuter_labels=None, data_level_control=False, two_step = False):

        assert method in ['ucb', 'ei', 'pi', 'ucb_auto', 'mix']
                
        n_candidates = X_candidate.shape[0]
        X_candidate_ref = ray.put(X_candidate)
        candidate_batch_size = 200000

        logging.info(f"--- start PCA ---")
        center = np.nanmedian(X_candidate, axis=0) if np.any(~np.isfinite(X_candidate)) else X_candidate.mean(axis=0)
        scale  = X_candidate.std(axis=0, ddof=0)
        X_scaled = (X_candidate - center) / scale
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        eps = 1e-9
            
        # ---------- 1. PCA(2) ----------
        pca = PCA(n_components=2).fit(X_scaled)
        cand_Z = pca.transform(X_scaled)                  # [N,2] -> PC1, PC2
        cand_Zidx_pc1 = np.argsort(cand_Z[:, 0])
        cand_Zidx_pc2 = np.argsort(cand_Z[:, 1])
        logging.info(f"--- finish PCA ---")
        
        # pca = PCA(n_components=2)
        # cand_Z = pca.fit_transform(X_candidate)
        
        if stack:
            stacking_models = []
            stacking_scores = []
            for target_i in range(num_of_targets):
                if model_result is None:
                    stack_file_path = f"{model_path}/stacking_models_{target_i}.pkl"
                    with open(stack_file_path, 'rb') as f:
                        data = pickle.load(f)
                    stacking_models.append(data['models'])
                    stacking_scores.append(data['errors'])
                else:
                    stacking_models.append(model_result[target_i]['stacking_models']['models'])
                    stacking_scores.append(model_result[target_i]['stacking_models']['errors'])
                    
            stacking_scores = stacking_scores/np.sum(np.array(stacking_scores), axis=0)
        
        acq_values = np.zeros([n_candidates, num_of_targets])
        
        if select_region is not None and y_value is not None:
            y_value = -np.abs(y_value-np.mean(select_region, axis=0))
        
        if y_value is None:
            acq_y_bests = None
        else:
            acq_y_bests = np.max(y_value, axis=0)
        
        if use_model_correlation and num_of_targets>1:
            target_y_num = num_of_targets-1
        else:
            target_y_num = num_of_targets
            
        if use_correlation and num_of_targets>1:
            if y_value is None:
                corr_acq_y_bests = None
            else:
                y_best_indexes = pg.fast_non_dominated_sorting(points = -y_value[:,:target_y_num])[0][0]
                # corr_acq_y_bests = y_value[:,:target_y_num][y_best_indexes]
                corr_acq_y_bests = y_value[y_best_indexes]
        
        corr_means, corr_stds, corr_modelstds, ori_scores, exp_ori_scores, struct_scores, clpss, flat_info_dict = {}, {}, {}, {}, {}, {}, {}, {}
        oob_residual_variances = {
            model_name: np.full(num_of_targets, np.nan, dtype=float)
            for model_name in model_name_list
        }
        # Load models
        for sg_model in model_name_list:
            logging.info(f"--- {sg_model}: Calculating the mean and std of candidates ---")
            ori_means = np.zeros((n_candidates, num_of_targets))
            ori_stds = np.zeros((n_candidates, num_of_targets))
            ori_modelstds = np.zeros((n_candidates, num_of_targets))
            ori_maxs = np.zeros((n_candidates, num_of_targets))
            ori_model_scores = np.zeros(num_of_targets)
            exp_ori_model_scores = np.zeros(num_of_targets)
            struct_model_scores = np.zeros(num_of_targets)
            flat_info_matrix = np.zeros(num_of_targets)
            model_clpss = np.zeros(num_of_targets)

            logging.info(f"start {sg_model}")
            for target_i in range(num_of_targets):
                if model_result is None:
                    logging.info(f'load models {sg_model}_{target_i}')
                    file_path = f"{model_path}/{sg_model}_{target_i}.pkl"
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    models = data['models']
                    model_errors = data['errors']
                    model_elpd_scores = np.clip(np.exp(data['elpd_per_point_mean'])/np.e, 0, 1)
                    model_clps = data['crps']
                    sigma_data2 = _read_sigma_data2(data)
                else:
                    pack = model_result[target_i][sg_model]
                    models = pack['models']
                    model_errors = pack['errors']
                    model_elpd_scores = np.clip(np.exp(pack['elpd_per_point_mean'])/np.e, 0, 1)
                    model_clps = pack['crps']
                    sigma_data2 = _read_sigma_data2(pack)

                if sigma_data2 is not None:
                    oob_residual_variances[sg_model][target_i] = sigma_data2

                if stack:
                    model_inter_score = stacking_scores[target_i][sg_model]
                    score_mu, score_std = np.mean(model_errors), np.std(model_errors)*len(model_errors)/(len(model_errors)-1)
                    model_extra_score = np.clip(score_mu - 0.01*score_std, 0, np.inf)
                    exp_score_mu, clps = model_elpd_scores, np.mean(model_clps)-0.1*np.std(model_clps)
                    model_extra_exp_score = exp_score_mu
                    model_score = (model_extra_score+model_inter_score)/2
                    exp_model_score = model_extra_exp_score
                else:
                    score_mu, score_std = np.mean(model_errors), np.std(model_errors)*len(model_errors)/(len(model_errors)-1)
                    exp_score_mu, clps = model_elpd_scores, np.mean(model_clps)-0.1*np.std(model_clps)
                    model_score = np.clip(score_mu - 0.01*score_std, 0, np.inf)
                    exp_model_score = exp_score_mu
                
                ## model scores
                model_score = np.nan_to_num(model_score, nan=0.0)
                exp_model_score = np.nan_to_num(exp_model_score, nan=0.0)
                clps_score = np.nan_to_num(clps, nan=0.0)
                
                if n_candidates <= candidate_batch_size:
                    tasks = []
                    for model in models:
                        tasks.append(model_predict.remote(model, X_candidate_ref, sg_model, start=0, end=n_candidates))
                    
                    results = ray.get(tasks)
                else:
                    tasks_by_model = [[] for _ in models]
                    results = []
                    
                    for model_idx, model in enumerate(models):
                        for i in range(0, n_candidates, candidate_batch_size):
                            start = i
                            end = min(i + candidate_batch_size, n_candidates)
                            tasks_by_model[model_idx].append(model_predict.remote(model, X_candidate_ref, sg_model, start, end))
    
                    for model_idx, tasks_for_one_model in enumerate(tasks_by_model):
                        # batch_results, e.g., [(preds_batch1, uncert_batch1), (preds_batch2, uncert_batch2), ...]
                        batch_results = ray.get(tasks_for_one_model)
                    
                        full_preds = np.concatenate([res[0] for res in batch_results])
                        full_internal_variances = np.concatenate([res[1] for res in batch_results])
                        results.append((full_preds, full_internal_variances))
                
                preds = np.asarray([res[0] for res in results], dtype=float)
                internal_variances = np.asarray([res[1] for res in results], dtype=float)
                mean, model_variance, within_variance, between_variance = aggregate_bootstrap_model_variance(
                    preds,
                    internal_variances,
                )
                mean = mean.reshape(-1)
                std_raw = np.sqrt(np.maximum(model_variance, 0.0)).reshape(-1)
                std = std_raw

                logging.info(
                    "model_uncertainty[%s][%s]: within_max=%.4g, between_max=%.4g, total_std_max=%.4g",
                    sg_model,
                    target_i,
                    float(np.max(within_variance)),
                    float(np.max(between_variance)),
                    float(np.max(std_raw)),
                )

                logging.info(f"--- start structure_score calculation ---")
                structure_score, s_detail = compute_structure_with_rigidity(mean, cand_Z, cand_Zidx_pc1, cand_Zidx_pc2)
                logging.info(f"{sg_model}_structure_score: {structure_score}, {s_detail['S_low']}, {s_detail['S_curv']}, {s_detail['SF']}, {s_detail['rho']}")
                logging.info(f"--- finish structure_score calculation ---")
                
                ## model post scores
                # post_model_score = model_score * np.nan_to_num(structure_score, nan=0.0) 
                # exp_post_model_score = exp_model_score * np.nan_to_num(structure_score, nan=0.0)

                model_structure_score = np.nan_to_num(structure_score, nan=0.0) 

                ori_means[:, target_i] = mean
                ori_stds[:, target_i] = std
                ori_modelstds[:, target_i] = std_raw
                ori_model_scores[target_i] = model_score
                exp_ori_model_scores[target_i] = exp_model_score
                struct_model_scores[target_i] = model_structure_score
                flat_info_matrix[target_i] = s_detail['is_flat']
                model_clpss[target_i] = clps_score
                    
                if select_region is not None:
                    logging.info(
                        "region selection active: mean_shape=%s, select_region_shape=%s",
                        mean.shape,
                        select_region.shape,
                    )
                    mean = -np.abs(mean-np.mean(select_region, axis=0)[target_i])
            
            corr_means[sg_model] = ori_means
            corr_stds[sg_model] = ori_stds
            corr_modelstds[sg_model] = ori_modelstds
            ori_scores[sg_model] = ori_model_scores
            exp_ori_scores[sg_model] = exp_ori_model_scores
            struct_scores[sg_model] = struct_model_scores
            flat_info_dict[sg_model] = flat_info_matrix
            clpss[sg_model] = model_clpss

        ### ----------- model predict values and std ----------- ### 
        all_sg_values = np.nan_to_num(np.array([corr_means[i] for i in model_name_list]), nan=0.0)                # [M,N,T]
        all_sg_value_sgstds = np.nan_to_num(np.array([corr_stds[i] for i in model_name_list]), nan=1e-8)

        ### ----------- model score values and normalize ----------- ### 
        all_sg_ori_scores = np.nan_to_num(np.array([ori_scores[i] for i in model_name_list]), nan=0.0)
        exp_all_sg_ori_scores = np.nan_to_num(np.array([exp_ori_scores[i] for i in model_name_list]), nan=0.0)
        all_sg_clpss_scores = np.nan_to_num(np.array([clpss[i] for i in model_name_list]), nan=0.0)
        
        # all_sg_ori_scores     /= all_sg_ori_scores.sum(axis=0, keepdims=True) + 1e-9
        # exp_all_sg_ori_scores /= exp_all_sg_ori_scores.sum(axis=0, keepdims=True) + 1e-9
        # all_sg_clpss_scores /= all_sg_clpss_scores.sum(axis=0, keepdims=True) + 1e-9

        ### ----------- acq needed means and stds ----------- ### 
        tar_mean = all_sg_values*np.expand_dims(all_sg_ori_scores, 1)                                 # [M,N,T]
        tar_std = all_sg_value_sgstds*np.expand_dims(all_sg_ori_scores, 1)
        exp_tar_mean = all_sg_values*np.expand_dims(exp_all_sg_ori_scores, 1)
        exp_tar_std = all_sg_value_sgstds*np.expand_dims(exp_all_sg_ori_scores, 1)

        ### ----------- cross model means and stds ----------- ### 
        all_sg_values_mean = np.mean(all_sg_values, axis=0)
        all_sg_values_std = np.std(all_sg_values, axis=0)

        ### ----------- KF needed means and stds ----------- ### 
        # all_sg_pred_stds = np.array([1/(corr_stds[i]**2) for i in model_name_list])  ### normal KF_std
        # reverse_all_sg_pred_stds = np.array([(corr_stds[i]**2) for i in model_name_list])  ### reverse KF_std
        reverse_all_sg_pred_stds = np.array([(corr_modelstds[i]**2) for i in model_name_list])
        oob_residual_variance_array = np.array(
            [oob_residual_variances[i] for i in model_name_list]
        )
        R_model, p_reference, oob_valid = calibrate_oob_noise_variance(
            reverse_all_sg_pred_stds,
            oob_residual_variance_array,
            eps=eps,
        )
        std_weights, reverse_std_weights, model_kalman_gain, total_error_variances = kalman_fusion_weights(
            reverse_all_sg_pred_stds,
            R_model,
            shrinkage=0.1,
            eps=eps,
        )
        KF_all_sg_values = all_sg_values*std_weights
        reverse_KF_all_sg_values = all_sg_values*reverse_std_weights

        # Compute improvement on sampled or preset candidates for SNR and D_m calculations.
        M, N, T = all_sg_values.shape

        if acq_y_bests is None:
            y_star = all_sg_values.max(axis=1, keepdims=True)       # [M,1,T] or mean
        else:
            y_star = acq_y_bests.reshape(1,1,-1)

        # -------- 1) Per-model intrinsic SNR: robust two-channel aggregation of peak top and peak shoulder.
        sigma_model = all_sg_value_sgstds
        sigma_calib = np.std(all_sg_values, axis=0, keepdims=True)
        sigma_eff = np.sqrt(sigma_model**2 + (0.5 * sigma_calib)**2 + 1e-6)
        
        kappa0      = 1.0
        kappa_max   = 3.0
        target_pos  = 0.10  # Target at least about 10% of (mu + kappa*sigma) values exceeding y_star.
        margin0     = all_sg_values + kappa0 * sigma_model - y_star
        p_pos       = np.mean(margin0 > 0.0)  # Current positive-improvement rate.
        kappa       = min(kappa_max, kappa0 + 2.0 * max(0.0, target_pos - p_pos))

        # --- 2) Soft improvement: softplus hinge avoids large zero-valued regions. ---
        tau         = 0.05  # Smoothing temperature; smaller values approach ReLU and can be kept constant.
        margin      = all_sg_values + kappa * sigma_model - y_star
        # softplus: tau * log(1 + exp(margin / tau))
        _imp_soft    = tau * np.log1p(np.exp(margin / tau))      # [M,N,T]
        imp_soft = _imp_soft / (np.abs(y_star) + 0.01)
        imp = imp_soft

        ### ----------- calculating SNR scores ----------- ### 
        
        # ------------ 3) SNR using soft improvement as the amplitude. ------------ #
        snr_full    = imp_soft / sigma_eff
        snr_full = np.nan_to_num(snr_full, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3a) Peak-top channel: weight by imp_soft, so points nearer the peak get more weight.
        w_head      = imp_soft
        den_head    = w_head.sum(axis=1, keepdims=True) + 1e-9             # [M,1,T]
        snr_head_m  = (snr_full * w_head).sum(axis=1, keepdims=False) / den_head.squeeze(1)  # [M,T]
        snr_head_m = np.nan_to_num(snr_head_m, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3b) Peak-shoulder channel: use only margin > 0 to avoid domination by a few high peaks.
        w_flat_bin  = (margin > 0).astype(np.float32)
        den_flat    = w_flat_bin.sum(axis=1, keepdims=True) + 1e-9
        snr_flat_m  = (snr_full * w_flat_bin).sum(axis=1, keepdims=False) / den_flat.squeeze(1)  # [M,T]
        snr_flat_m = np.nan_to_num(snr_flat_m, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3c) Fuse the two channels with equal weights and no new hyperparameters.
        snr_m       = 0.5 * snr_head_m + 0.5 * snr_flat_m                  # [M,T]
        snr_norm = snr_m
        
        # model structure score
        all_struct_sg_scores = np.nan_to_num(np.array([struct_scores[i] for i in model_name_list]), nan=0.0)
        flat_info = np.array([flat_info_dict[i] for i in model_name_list]).astype(bool)
        all_sg_ori_scores[flat_info] *= 0.01
        exp_all_sg_ori_scores[flat_info] *= 0.01
        logging.info(
            "\n".join([
                _format_array_for_log("flat_info", flat_info),
                _format_array_for_log("snr_norm", snr_norm),
                _format_array_for_log("snr_m", snr_m),
                _format_array_for_log("all_struct_sg_scores", all_struct_sg_scores),
            ])
        )

        ### ----------- acq/KF ratio ----------- ### 

        if data_level_control:
            # ----------- data level ratio ----------- #    
            P_pred_ens = reverse_all_sg_pred_stds
            R_eff = R_model
            K = model_kalman_gain

        else:
            # ----------- acq_KF top level ratio ----------- #    
            P_pred_ens = np.mean(reverse_all_sg_pred_stds, axis=(0,1))
            R_eff = np.mean(R_model, axis=(0,1))
            K = np.divide(
                P_pred_ens,
                P_pred_ens + R_eff,
                out=np.zeros_like(P_pred_ens),
                where=(P_pred_ens + R_eff) > 0.0,
            )
        logging.info(
            "\n".join([
                _format_array_for_log("oob_residual_variances", oob_residual_variance_array),
                _format_array_for_log("candidate_P_reference", p_reference),
                _format_array_for_log("oob_valid", oob_valid),
                _format_array_for_log("reverse_all_sg_pred_stds", reverse_all_sg_pred_stds),
                _format_array_for_log("P_pred_ens", P_pred_ens),
                _format_array_for_log("R_eff", R_eff),
                _format_array_for_log("P_plus_R", total_error_variances),
                _format_array_for_log("K", K),
            ])
        )
    
        # beta = R / (P + R): observation noise relative to model uncertainty.
        beta = np.clip(1-K, 0.0, 1.0)  # How much we trust the model consensus.

        logging.info(
            _format_scalar_stats_for_log("beta", beta) + "\n" +
            _format_array_for_log("beta_values", beta)
        )

        ### ----------- r2/elpd ratio ----------- ### 
        ee_ratio = np.expand_dims(np.sqrt(snr_norm * all_struct_sg_scores), 1)
        # ee_ratio = np.expand_dims((snr_norm + all_struct_sg_scores)/2, 1)
        
        # ee_ratio = np.clip(np.expand_dims(all_struct_sg_scores, 1), 0.01, 0.99)
        # ee_ratio = np.clip(np.expand_dims(np.sqrt(snr_norm * all_struct_sg_scores), 1), 0.05, 0.95)
        logging.info(_format_array_for_log("ee_ratio", ee_ratio))
        
        ### ----------- KF/rKF ratio ----------- ###
        norm_all_sg_ori_scores = all_sg_ori_scores/np.max(all_sg_ori_scores, axis=0)
        norm_exp_all_sg_ori_scores_mean = exp_all_sg_ori_scores/np.max(exp_all_sg_ori_scores, axis=0)
        
        ### R2 score results
        all_sg_ori_scores_mean = np.mean(norm_all_sg_ori_scores, axis=0)
        all_sg_ori_scores_std = np.std(norm_all_sg_ori_scores, axis=0)
        
        # entropy_r2 = mix_ratio_from_scores(all_sg_ori_scores, method='entropy', invert=False, keepdims=True)  "gs", "std"
        entropy_r2 = all_sg_ori_scores_mean*all_sg_ori_scores_std
        logging.info(_format_array_for_log("entropy_r2", entropy_r2))
        
        ### elpd score results
        exp_all_sg_ori_scores_mean = np.mean(norm_exp_all_sg_ori_scores_mean, axis=0)
        exp_all_sg_ori_scores_std = np.std(norm_exp_all_sg_ori_scores_mean, axis=0)
        
        # entropy_elpd = mix_ratio_from_scores(exp_all_sg_ori_scores, method='entropy', invert=False, keepdims=True)  "gs", "std"
        entropy_elpd = exp_all_sg_ori_scores_mean*exp_all_sg_ori_scores_std
        logging.info(_format_array_for_log("entropy_elpd", entropy_elpd))
        
        ### clpss deviation
        all_sg_clpss_scores_std = np.std(all_sg_clpss_scores, axis=0)
        
        logging.info(
            "\n".join([
                _format_array_for_log("all_sg_ori_scores", all_sg_ori_scores),
                _format_array_for_log("all_sg_ori_scores_mean", all_sg_ori_scores_mean),
                _format_array_for_log("all_sg_ori_scores_std", all_sg_ori_scores_std),
                _format_array_for_log("exp_all_sg_ori_scores", exp_all_sg_ori_scores),
                _format_array_for_log("exp_all_sg_ori_scores_mean", exp_all_sg_ori_scores_mean),
                _format_array_for_log("exp_all_sg_ori_scores_std", exp_all_sg_ori_scores_std),
                _format_array_for_log("all_sg_clpss_scores", all_sg_clpss_scores),
                _format_scalar_stats_for_log("all_sg_clpss_scores_stats", all_sg_clpss_scores),
            ])
        )
        
        dev_ratio_r2 = entropy_r2
        dev_ratio_elpd = entropy_elpd

        logging.info(_format_array_for_log("dev_ratio_r2", dev_ratio_r2))
        logging.info(_format_array_for_log("dev_ratio_elpd", dev_ratio_elpd))

        ### ----------- acq/KF ratio ----------- ### 
        # lambda_kf_final = beta / (1-D_m.std(axis=0, keepdims=True))
        # lambda_t = np.clip(lambda_kf_final, 0.0, 1.0)
        lambda_t = np.clip(beta*np.mean(all_struct_sg_scores, axis=0), 0.01, 1)
        logging.info(_format_array_for_log("lambda_t", lambda_t))
        
        ### -------------------------------------------- Conducting acq sections -------------------------------------------- ### 
        if method == 'ucb':
            acq_value = self.ucb(tar_mean, tar_std)
            exp_acq_value = self.ucb(exp_tar_mean, exp_tar_std)
        elif method == 'ucb_auto':
            entropy = self.compute_entropy(tar_std**2)
            hpara = self.adaptive_kappa(entropy, kappa_scale=0.5)
            acq_value = self.ucb(tar_mean, tar_std, hpara=hpara)
            exp_acq_value = self.ucb(exp_tar_mean, exp_tar_std, hpara=hpara)
        elif method == 'ei' or method == 'pi':
            if acq_y_bests is None:
                raise ValueError(f"Unknown current best target value y_best, add current best target value if you want to use EI or PI")
            if method == 'ei':
                acq_value = self.ei(tar_mean, tar_std, acq_y_bests)
                exp_acq_value = self.ei(exp_tar_mean, exp_tar_std, acq_y_bests)
                # print(f'{sg_model} acq_value(ei): {np.mean(acq_value)}')
            elif method == 'pi':
                acq_value = self.pi(tar_mean, tar_std, acq_y_bests)
                exp_acq_value = self.pi(exp_tar_mean, exp_tar_std, acq_y_bests)
                # print(f'{sg_model} acq_value(pi): {np.mean(acq_value)}')
        elif method == 'mix':
            entropy = self.compute_entropy(tar_std**2)
            hpara = self.adaptive_kappa(entropy, kappa_base=self.hpar, kappa_scale=0.5)
            w_ucb, w_ei = self.adaptive_weights(entropy, scale_factor=1.0)
            acq_ucb = self.ucb(tar_mean, tar_std, hpara=hpara)
            acq_ei = self.ei(tar_mean, tar_std, acq_y_bests)
            lambda_ei = (acq_ucb.max() - acq_ucb.min()) / (acq_ei.max() - acq_ei.min() + 1e-9)
            acq_value = w_ucb * acq_ucb + w_ei * lambda_ei * acq_ei
            exp_acq_ucb = self.ucb(exp_tar_mean, exp_tar_std, hpara=hpara)
            exp_acq_ei  = self.ei(exp_tar_mean,  exp_tar_std,  acq_y_bests)
            exp_acq_value = w_ucb * exp_acq_ucb + w_ei * lambda_ei * exp_acq_ei
        else:
            raise ValueError(f"Unknown acquisition method: {method}")

        ## get final acq values
        model_acq_values = np.mean(acq_value*ee_ratio, axis=0)
        model_exp_acq_values = np.mean(exp_acq_value*(1-ee_ratio), axis=0)

        _log_nan_summary(
            acq_value=acq_value,
            exp_acq_value=exp_acq_value,
            ee_ratio=ee_ratio,
            beta=beta,
            model_acq_values=model_acq_values,
            model_exp_acq_values=model_exp_acq_values,
        )
        
        acq_model_values = model_acq_values + model_exp_acq_values

        ### -------------------------------------------- Conducting model knowledge align (KF) sections -------------------------------------------- ###
        ## model knowledge align
        # all_sg_values = np.array([ranks_best(corr_means[i], ties='min', larger_value_larger_order=True, normalize=True) for i in model_name_list])
        
        # # R2
        # all_sg_values_KF_mean = np.sum(KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (1-dev_ratio_r2) * ee_ratio, axis=0)
        # all_sg_values_KF_std = np.sum(reverse_KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (dev_ratio_r2) * ee_ratio, axis=0)
        # all_sg_std_value = all_sg_values_KF_mean + all_sg_values_KF_std

        # # Elpd
        # exp_all_sg_values_KF_mean = np.sum(KF_all_sg_values * np.expand_dims(exp_all_sg_ori_scores, 1) * (1-dev_ratio_elpd) * (1-ee_ratio), axis=0)
        # exp_all_sg_values_KF_std = np.sum(reverse_KF_all_sg_values * np.expand_dims(exp_all_sg_ori_scores, 1) * (dev_ratio_elpd) * (1-ee_ratio), axis=0)
        # exp_all_sg_std_value = exp_all_sg_values_KF_mean + exp_all_sg_values_KF_std

        # # Weight add
        # KF_model_values = all_sg_std_value + exp_all_sg_std_value

        # only R2
        all_sg_values_KF_mean = np.sum(KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (1-dev_ratio_r2), axis=0)
        # all_sg_values_KF_std = np.sum(reverse_KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (dev_ratio_r2*(1+beta)), axis=0)  #lambda_t
        all_sg_values_KF_std = np.sum(reverse_KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (dev_ratio_r2*(1+lambda_t)), axis=0)  #lambda_t
        # all_sg_values_KF_std = np.sum(reverse_KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (dev_ratio_r2, axis=0)
        all_sg_std_value = all_sg_values_KF_mean + all_sg_values_KF_std

        KF_model_values = all_sg_std_value

        # ### ----------- calculating sorting consistency of the models ----------- ### 
        # # -------- Rank-based consistency and disagreement D.
        # acq_rank = rankdata(acq_model_values, axis=0, method='average')
        # acq_ranks01 = (acq_rank - 1) / (acq_model_values.shape[0] - 1 + 1e-9)
        # KF_rank = rankdata(KF_model_values, axis=0, method='average')
        # KF_ranks01 = (KF_rank - 1) / (KF_model_values.shape[0] - 1 + 1e-9)
        # mean_rank = (acq_ranks01+KF_ranks01)/2                               # [N,T]
        
        # ranks  = rankdata(all_sg_values, axis=1, method='average')           # [M,N,T]
        # ranks01 = (ranks - 1) / (all_sg_values.shape[1] - 1 + 1e-9)          # [M,N,T]
        
        # dev = np.abs(ranks01 - mean_rank[None, :, :])                        # [M,N,T]
        # # D_m = np.nan_to_num(dev.mean(axis=1), nan=0.0)
        
        # knum = int(min(np.ceil(0.10 * N), batch_size*10))
        # knum = min(max(knum, 10), N)  ## prevent zero num
        # print(f"knum: {knum}")
        
        # top_k_indices = np.argsort(mean_rank, axis=0)[-knum:, :]             # [K,T]
        # top_k_indices = np.flip(top_k_indices, axis=0)
        
        # dev_top_k = np.take_along_axis(dev, top_k_indices[None, :, :], axis=1)   # [M,K,T]
        # D_m = np.nan_to_num(dev_top_k.mean(axis=1), nan=0.0)
        # logging.info(f"D_m:\n{D_m}\nD_m shape: {D_m.shape}")
        
        # penalty_factor = 1 - np.expand_dims(D_m * beta, 1)
        # logging.info(f"penalty_factor: {penalty_factor}, {penalty_factor.shape}")

        # logging.info(f"acq_model_values and KF_model_values maxs: {acq_model_values.max()}, {KF_model_values.max()}")
        # logging.info(f"acq_model_values and KF_model_values mins: {acq_model_values.min()}, {KF_model_values.min()}")

        # # --- Compute strategy consistency. ---
        # correlations = []
        # acq_best_index = np.argsort(acq_model_values, axis=0)[-knum:, :]
        # kf_best_index = np.argsort(KF_model_values, axis=0)[-knum:, :]
        # all_best_index = np.vstack((acq_best_index,kf_best_index))
        # # print(f"index of acq and KF: {acq_best_index}, {kf_best_index}")
        # # acq_bests = np.take_along_axis(acq_model_values, all_best_index, axis=0)[-knum:, :]
        # # kf_bests = np.take_along_axis(KF_model_values, all_best_index, axis=0)[-knum:, :]
        # for t in range(T):
        #     use_index = all_best_index[:,t]
        #     unique_index = list(set(use_index))
        #     logging.info(f"{t} index len: {len(unique_index)}, {len(use_index)}")
        #     print(acq_model_values[:, t][unique_index])
        #     corr, _ = spearmanr(acq_model_values[:, t][unique_index], KF_model_values[:, t][unique_index])
        #     correlations.append(np.maximum(0.0, corr))
        
        # strategy_consistency = np.clip(np.nan_to_num(np.array(correlations), nan=0.0), 0.01, 1)   # Shape is [T]; each value lies in [-1, 1], and values closer to 1 mean the two strategies agree.
        # logging.info(f"strategy_consistency:\n{strategy_consistency}\nD_m shape: {strategy_consistency.shape}")

        ## two step samples:
        if two_step:
            cand_num = min(max(int(np.ceil(np.mean(lambda_t) * N)), batch_size*10), N)
            # top_k_acq = np.argsort(acq_model_values, axis=0)[-cand_num:, :]
            # print(top_k_acq)
        else:
            cand_num = N
            # top_k_acq = np.argsort(acq_model_values, axis=0)[-cand_num:, :]

        # ### 
        # final_model_acq_values = np.mean(acq_value*ee_ratio*penalty_factor, axis=0)
        # final_model_exp_acq_values = np.mean(exp_acq_value*(1-ee_ratio)*penalty_factor, axis=0)
        # final_acq_model_values = final_model_acq_values + final_model_exp_acq_values

        # # # R2
        # # final_all_sg_values_KF_mean = np.sum(KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (dev_ratio_r2) * ee_ratio*penalty_factor, axis=0)
        # # final_all_sg_values_KF_std = np.sum(reverse_KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (1-dev_ratio_r2) * ee_ratio*penalty_factor, axis=0)
        # # final_all_sg_std_value = final_all_sg_values_KF_mean + final_all_sg_values_KF_std
        # # # Elpd
        # # final_exp_all_sg_values_KF_mean = np.sum(KF_all_sg_values * np.expand_dims(exp_all_sg_ori_scores, 1) * (dev_ratio_elpd) * (1-ee_ratio)*penalty_factor, axis=0)
        # # final_exp_all_sg_values_KF_std = np.sum(reverse_KF_all_sg_values * np.expand_dims(exp_all_sg_ori_scores, 1) * (1-dev_ratio_elpd) * (1-ee_ratio)*penalty_factor, axis=0)
        # # final_exp_all_sg_std_value = final_exp_all_sg_values_KF_mean + final_exp_all_sg_values_KF_std
        # # # final KF values
        # # final_KF_model_values = final_all_sg_std_value + final_exp_all_sg_std_value
        
        # # R2
        # final_all_sg_values_KF_mean = np.sum(KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (dev_ratio_r2) * penalty_factor, axis=0)
        # final_all_sg_values_KF_std = np.sum(reverse_KF_all_sg_values * np.expand_dims(all_sg_ori_scores, 1) * (1-dev_ratio_r2) * penalty_factor, axis=0)
        # final_all_sg_std_value = final_all_sg_values_KF_mean + final_all_sg_values_KF_std
        # # final KF values
        # final_KF_model_values = final_all_sg_std_value

        
        final_acq_model_values = acq_model_values
        final_KF_model_values = KF_model_values

        final_acq_best_index = np.array(list(set(np.argsort(final_acq_model_values, axis=0)[-cand_num:, :].flatten())))
        
        logging.info(f"final_acq_model_values and final_KF_model_values maxs: {final_acq_model_values.max()}, {final_KF_model_values.max()}")
        logging.info(f"final_acq_model_values and final_KF_model_values mins: {final_acq_model_values.min()}, {final_KF_model_values.min()}")
        
        ### ----------- Final values ----------- ### 
        acq_values += np.sqrt(np.power(np.clip(final_acq_model_values, 0, np.inf), 2) + np.power(np.clip(final_KF_model_values, 0, np.inf), 2))
        # acq_values += final_acq_model_values*(1-lambda_t) + final_KF_model_values*lambda_t
        # acq_values += np.log(np.exp(final_acq_model_values) + np.exp(final_KF_model_values))
        
        if acq_values.shape[1]>1:
            acq_min = np.min(acq_values, axis=0)-1e-6
            acq_max = np.max(acq_values, axis=0)+1e-6
            den = acq_max - acq_min
            den[den <= 0] = 1.0
            norm_acq = (acq_values-acq_min)/den
            
            _log_nan_summary(norm_acq=norm_acq, acq_values=acq_values)

            if use_model_correlation and num_of_targets>1:
                combined_acq = self.hypervolume(norm_acq[:,:target_y_num]) + acq_values[:,target_y_num]
                # combined_acq = self.hypervolume(norm_acq[:,:target_y_num]) + self.hypervolume(acq_values[:,:target_y_num]) + acq_values[:,target_y_num]
                # combined_acq = self.hypervolume(norm_acq[:,:target_y_num]) + norm_acq[:,target_y_num]
            else:
                combined_acq = self.hypervolume(norm_acq)# + self.hypervolume(acq_values)
                # combined_acq = self.hypervolume(norm_acq)
        else:
            combined_acq = acq_values.reshape(-1)
        
        # dist_penalty, inverse power law
        # knn = NearestNeighbors(n_neighbors=3).fit(X_train)
        # dist, _ = knn.kneighbors(X_candidate)
        # dist_penalty = dist[:, -1]
        
        # # inverse power law parameters
        # alpha = 1.0
        # eps   = 1e-9
        # combined_acq -= self.hpar / np.power(dist_penalty + eps, alpha)
        
        pre_next_indexes = final_acq_best_index[np.argsort(combined_acq[final_acq_best_index])[::-1][:min(20000, n_candidates)]]
        
        if use_correlation and num_of_targets>1:
            corr_acq_values = np.zeros(len(pre_next_indexes))
            logging.info(_format_array_for_log("corr_acq_y_bests", corr_acq_y_bests))
            for sg_model in model_name_list:
                logging.info(f"--- {sg_model}: Loading residual correlation matrix for multi-objects tasks ---")
                _means, _stds, _score = corr_means[sg_model][pre_next_indexes][:,:target_y_num], corr_stds[sg_model][pre_next_indexes][:,:target_y_num], exp_ori_scores[sg_model][:target_y_num]
                # _means, _stds, _score = corr_means[sg_model][pre_next_indexes], corr_stds[sg_model][pre_next_indexes], exp_ori_scores[sg_model]
                try:
                    with open(f'{model_path}/residual_correlation.pkl', 'rb') as f:
                        corr_matrix_dict = pickle.load(f)
                    # corr_matrix = corr_matrix_dict['stacking_models'] if stack else corr_matrix_dict[sg_model]
                    corr_matrix = corr_matrix_dict[sg_model]
                except FileNotFoundError:
                    logging.warning("residual_correlation.pkl not found. Assuming independence (identity matrix).")
                    corr_matrix = np.eye(num_of_targets)
                
                logging.info(f"--- {sg_model}: Calculating Hypervolume Improvement ---")
                
                if corr_acq_y_bests is None or not isinstance(corr_acq_y_bests, np.ndarray) or corr_acq_y_bests.ndim != 2:
                     raise ValueError("For HV, `y_value` must be provided to calculate the observed Pareto front (an array of points).")
                
                pareto_front_obs = corr_acq_y_bests[:,:target_y_num]
                # ref_point = np.min(all_means, axis=0) - 3.0
                ref_point = np.ones(target_y_num)*3
                ucb_method = True if method in ['ucb', 'ucb_auto', 'mix'] else False
                
                ### all_means, y_best and ref_point need to be negated as pg.hypervolume is designed for minimization
                if select_region is not None:
                    _region = np.mean(select_region, axis=0)[:target_y_num]
                else:
                    _region = None
                corr_model_acq_values = self._calculate_mc_hvi(-_means, _stds**2, corr_matrix, -pareto_front_obs, ref_point, _score, ucb=ucb_method, select_region=_region)
                corr_acq_values += corr_model_acq_values
                
            combined_acq[pre_next_indexes] += corr_acq_values
            screen_next_indexes = np.argsort(combined_acq[pre_next_indexes])[::-1]
            _next_indexes = pre_next_indexes[screen_next_indexes]
        else:
            _next_indexes = pre_next_indexes

        if train_clsuter_labels is None:
            metreics_Xtrain = X_train
        else:
            logging.info("train_Clustersampler")
            train_Clustersampler = ClusterBootstrapSampler(batch_size=batch_size, enable_umap=False)
            _train_probs = train_Clustersampler.compute_bootstrap_probabilities_prob(train_clsuter_labels, np.sum(y_value, axis=1))
            # weights_baseline = np.ones_like(_train_probs) / len(_train_probs)
            # final_weights = weights_baseline + _train_probs
            # train_probs = final_weights/np.sum(final_weights)
            train_probs = _train_probs/np.sum(_train_probs)
            train_indices = np.arange(len(X_train))
            _bootstrap_train_indices = np.random.choice(train_indices, size=int(len(X_train)*1.2), replace=True, p=train_probs)
            bootstrap_train_indices = np.array(list(set(_bootstrap_train_indices)))
            logging.info(
                "cluster bootstrap sizes: unique=%s, sampled=%s",
                len(bootstrap_train_indices),
                len(_bootstrap_train_indices),
            )
            metreics_Xtrain = X_train[bootstrap_train_indices]
            
        metrics = estimate_density_and_spread(X_new=X_candidate[_next_indexes[:min(max(batch_size*2, 20), len(_next_indexes))]], X_train=metreics_Xtrain, y_new=None)
        logging.info(
            "metrics: LDR=%.4g, coverage=%.4g, KL=%.4g",
            metrics.get("LDR", -np.inf),
            metrics.get("coverage", 1.0),
            metrics.get("KL", 1.0),
        )
        if any([metrics.get("LDR", -np.inf) > 1.5, metrics.get("coverage", 1.0) < 0.2, metrics.get("KL", 1.0) < 0.05]):
            activate_diversity = True
        else:
            activate_diversity = False
        
        screen_X_candidate = X_candidate[_next_indexes]
        logging.info("start final sort")
        if diversity_method and activate_diversity:
            logging.info(
                "need diversity, start clustering/history-aware sort, "
                "screen_X_candidate_shape=%s",
                screen_X_candidate.shape,
            )
            Clustersampler = ClusterBootstrapSampler(
                batch_size=batch_size,
                enable_umap=False,
            )
            labels = Clustersampler.compute_bootstrap_probabilities_clustering(
                screen_X_candidate,
                enable_refine=True,
            )
            diversity_probs = Clustersampler.compute_bootstrap_probabilities_prob(
                labels,
                combined_acq[_next_indexes],
            )
            base_diversity_scores = (
                combined_acq[_next_indexes] * diversity_probs
            )
            selected_local, history_distances = select_history_aware_diverse_batch(
                screen_X_candidate,
                base_diversity_scores,
                batch_size=batch_size,
                X_history=X_train,
            )
            finite_history_distances = history_distances[
                np.isfinite(history_distances)
            ]
            if finite_history_distances.size:
                logging.info(
                    "history novelty distance: min=%.4g, median=%.4g, max=%.4g",
                    float(np.min(finite_history_distances)),
                    float(np.median(finite_history_distances)),
                    float(np.max(finite_history_distances)),
                )
            next_indexes = _next_indexes[selected_local]
        else:
            logging.info("no diversity needed, skip clustering/history-aware sort")
            sort_result = combined_acq[_next_indexes]
            next_indexes = _next_indexes[
                np.argsort(sort_result)[::-1][:batch_size]
            ]
        
        return next_indexes
        

    def MT_select_next(self, method, X_candidate, Mainmodel_train, batch_size=5, y_value=None):

        if y_value is None:
            y_best = None
        else:
            y_best = np.max(y_value, axis=0)
            
        tasks = []                     
        acq_value = np.zeros(X_candidate.shape[0])
        X_candidate_ref = ray.put(X_candidate)
        for model in Mainmodel_train:
            tasks.append(model_predict.remote(model, X_candidate_ref, sg_model=None, start=0, end=len(X_candidate)))

        # Aggregate internal posterior variance and between-bootstrap variance.
        results = ray.get(tasks)
        preds = np.asarray([res[0] for res in results], dtype=float)
        internal_variances = np.asarray([res[1] for res in results], dtype=float)
        mean, model_variance, _, _ = aggregate_bootstrap_model_variance(
            preds,
            internal_variances,
        )
        mean = mean.reshape(-1)
        std = np.sqrt(np.maximum(model_variance, 0.0)).reshape(-1)

        if method == 'ucb':
            acq_value = self.ucb(mean, std)
        elif method == 'ei' or method == 'pi':
            if y_best is None:
                raise ValueError(f"Unknown current best target value y_best, add current best target value if you want to use EI or PI")
            if method == 'ei':
                acq_value = self.ei(mean, std, y_best)
            elif method == 'pi':
                acq_value = self.pi(mean, std, y_best)
        else:
            raise ValueError(f"Unknown acquisition method: {method}")
        sort_result = acq_value.reshape(-1)
        
        next_indexes = np.argsort(sort_result)[::-1][:batch_size]
        
        return next_indexes


    def MF_predres(self, X_candidates, model_name_list, model_path, model_result=None, stack=False):

        cmean=[]
        cstd=[]       
        # Load models
        if stack:
            if model_result is None:
                stack_file_path = f"{model_path}/stacking_models_0.pkl"
                with open(stack_file_path, 'rb') as f:
                    data = pickle.load(f)
                stacking_model = data['stacking_model']
                stacking_score = data['stacking_error']
            else:
                stacking_model = model_result['stacking_model']
                stacking_score = model_result['stacking_error']
                
        
        for sg_model in model_name_list:

            if model_result is None:
                logging.info(f'load models {sg_model}_0')
                file_path = f"{model_path}/{sg_model}_0.pkl"
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                # model_name = data['model_name']
                # optimized_params = data['optimized_params']
                models = data['models']
                model_errors = [x for x in data['errors'] if np.isnan(x) == False]
            else:
                models = model_result[sg_model]['models']
                model_errors = [x for x in model_result[sg_model]['errors'] if np.isnan(x) == False]

            if stack:
                model_score = stacking_score[sg_model]
            else:
                score_mu, score_std = np.mean(model_errors), np.std(model_errors)
                model_score = np.clip(score_mu-0.01*score_std, 0, np.inf)

            # Aggregate internal posterior variance and between-bootstrap variance.
            preds = []
            internal_variances = []
            for model in models:
                pred, internal_variance = predict_mean_and_internal_variance(
                    model,
                    X_candidates,
                    model_name=sg_model,
                )
                preds.append(pred)
                if internal_variance is None:
                    internal_variance = np.zeros_like(pred, dtype=float)
                internal_variances.append(internal_variance)
            mean, model_variance, _, _ = aggregate_bootstrap_model_variance(
                np.asarray(preds, dtype=float),
                np.asarray(internal_variances, dtype=float),
            )
            mean = mean.reshape(-1)
            std = np.sqrt(np.maximum(model_variance, 0.0)).reshape(-1)
            weighted_mean = mean*model_score
            weighted_std = std*model_score
            cmean.append(weighted_mean)
            cstd.append(weighted_std)
        cmean_array = np.array(cmean)
        cstd_array = np.array(cstd)

        cmean = np.mean(cmean_array, axis=0)
        cstd = np.mean(cstd_array, axis=0)

        return (cmean, cstd)


    def BOfusion_select_next(self, method,HFidx_candidate,mean_tuple, std_tuple,cost,batch_size=10, y_value=None):

        if y_value is None:
            y_best = None
        else:
            y_best = np.max(y_value, axis=0)
        
        std_tuple = [np.array(std) for std in std_tuple]

        variance_tuple = [std**2 for std in std_tuple]
        
        weights = [1 / (var + 1e-10) for var in variance_tuple]  
        fused_mean = np.sum([mean * weight for mean, weight in zip(mean_tuple, weights)], axis=0) / np.sum(weights, axis=0)

        fused_variance = 1 / np.sum(weights, axis=0)
        fused_std = np.sqrt(fused_variance)

        if method == 'ucb':
            acq_value = self.ucb(fused_mean, fused_std)
        elif method == 'ei' or method == 'pi':
            if y_best is None:
                raise ValueError(f"Unknown current best target value y_best, add current best target value if you want to use EI or PI")
            if method == 'ei':
                acq_value = self.ei(fused_mean, fused_std, y_best)
            elif method == 'pi':
                acq_value = self.pi(fused_mean, fused_std, y_best)
        else:
            raise ValueError(f"Unknown acquisition method: {method}")
        
        sort_result = acq_value.reshape(-1)
        next_indexes = np.argsort(sort_result)[::-1][:batch_size]
        original_indexes = HFidx_candidate[next_indexes]  
        cost_std_tuple = tuple(arr / num for arr, num in zip(std_tuple, cost))
        preferredlevelforall = np.argmax(cost_std_tuple, axis=0)

        preferredlevel = preferredlevelforall[next_indexes]

        return original_indexes, preferredlevel
