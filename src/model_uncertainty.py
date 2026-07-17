"""Model-only predictive uncertainty utilities.

The aggregation follows the law of total variance over outer bootstrap models:

    Var(Y | x) = E_b[Var(Y | x, model_b)] + Var_b(E[Y | x, model_b])

Only models with a meaningful internal predictive distribution contribute the
first term. Boosting stages are deliberately not treated as ensemble samples.
"""

from __future__ import annotations

import numbers

import numpy as np
import torch
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)


_BAGGING_ENSEMBLE_TYPES = (
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
_BAGGING_MODEL_NAMES = {
    'BaggingClassifier',
    'BaggingRegressor',
    'ExtraTreesClassifier',
    'ExtraTreesRegressor',
    'RandomForest',
    'RandomForestClassifier',
    'RandomForestRegressor',
}
_GP_MODEL_NAMES = {'GP_cpu', 'GP_gpu'}


def _resolve_model_name(model, model_name=None):
    if model_name is not None:
        return model_name
    return getattr(model, 'model_name', model.__class__.__name__)


def _unwrap_estimator(model):
    underlying = getattr(model, 'model', None)
    return underlying if underlying is not None else model


def _predict_gp(model, X, model_name):
    estimator = _unwrap_estimator(model)
    if model_name == 'GP_gpu':
        try:
            parameter = next(estimator.parameters())
            tensor_device = parameter.device
            tensor_dtype = parameter.dtype
        except (AttributeError, StopIteration, TypeError):
            tensor_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tensor_dtype = torch.float32

        X_tensor = torch.as_tensor(
            X, dtype=tensor_dtype, device=tensor_device
        )
        with torch.no_grad():
            posterior = estimator.posterior(X_tensor)
        mean = posterior.mean.detach().cpu().numpy().reshape(-1)
        variance = posterior.variance.detach().cpu().numpy().reshape(-1)
        return mean, np.clip(variance, 0.0, np.inf)

    # sklearn GaussianProcessRegressor, either raw or inside SurrogateModel.
    mean, std = estimator.predict(X, return_std=True)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    variance = np.square(np.asarray(std, dtype=float).reshape(-1))
    return mean, np.clip(variance, 0.0, np.inf)


def _predict_bagging_ensemble(model, X):
    estimator = _unwrap_estimator(model)
    members = getattr(estimator, 'estimators_', None)
    if members is None or len(members) == 0:
        return np.asarray(model.predict(X), dtype=float).reshape(-1), None

    member_predictions = np.asarray(
        [member.predict(X) for member in members],
        dtype=float,
    )
    if member_predictions.ndim != 2:
        # The current BO framework is single-target at this point. Avoid silently
        # flattening a multi-output/member axis into an invalid uncertainty.
        return np.asarray(model.predict(X), dtype=float).reshape(-1), None

    mean = np.asarray(model.predict(X), dtype=float).reshape(-1)
    if len(member_predictions) == 1:
        variance = np.zeros_like(mean)
    else:
        # Treat members as samples from an internal model distribution. Do not
        # divide by their count: that would estimate numerical error of the mean
        # and would collapse as more trees are added.
        variance = np.var(member_predictions, axis=0, ddof=1)
    return mean, np.clip(variance, 0.0, np.inf)


def predict_mean_and_internal_variance(model, X, model_name=None):
    """Return predictive mean and meaningful internal variance, if available.

    The variance is returned as None for deterministic models and boosting
    models, even when they expose an estimators_ attribute: boosting stages are
    additive residual corrections, not exchangeable posterior samples.
    """
    resolved_name = _resolve_model_name(model, model_name)
    if resolved_name in _GP_MODEL_NAMES:
        return _predict_gp(model, X, resolved_name)

    estimator = _unwrap_estimator(model)
    if (
        resolved_name in _BAGGING_MODEL_NAMES
        or isinstance(estimator, _BAGGING_ENSEMBLE_TYPES)
    ):
        return _predict_bagging_ensemble(model, X)

    mean = np.asarray(model.predict(X), dtype=float).reshape(-1)
    return mean, None


def aggregate_bootstrap_model_variance(means, internal_variances=None):
    """Aggregate bootstrap predictions using the law of total variance."""
    means = np.asarray(means, dtype=float)
    if means.ndim != 2 or means.shape[0] == 0:
        raise ValueError(
            "means must have shape (n_bootstrap_models, n_candidates)."
        )

    ensemble_mean = np.mean(means, axis=0)
    if means.shape[0] > 1:
        between_variance = np.var(means, axis=0, ddof=1)
    else:
        between_variance = np.zeros_like(ensemble_mean)

    if internal_variances is None:
        within_variance = np.zeros_like(ensemble_mean)
    else:
        internal_variances = np.asarray(internal_variances, dtype=float)
        if internal_variances.shape != means.shape:
            raise ValueError(
                "internal_variances must have the same shape as means."
            )
        internal_variances = np.nan_to_num(
            internal_variances, nan=0.0, posinf=0.0, neginf=0.0
        )
        within_variance = np.mean(
            np.clip(internal_variances, 0.0, np.inf), axis=0
        )

    between_variance = np.nan_to_num(
        between_variance, nan=0.0, posinf=0.0, neginf=0.0
    )
    total_variance = np.clip(
        within_variance + between_variance, 0.0, np.inf
    )
    return ensemble_mean, total_variance, within_variance, between_variance


def calibrate_oob_noise_variance(
    model_variances,
    oob_residual_variances,
    eps=1e-12,
):
    """Estimate observation noise R without recounting candidate model variance.

    OOB residual variance is a total prediction-error estimate. We subtract a
    robust candidate-level reference for P. Missing OOB values fall back to the
    median calibrated R from other models for the same target; if all are
    missing, the target-level median P is used as a conservative fallback.
    """
    model_variances = np.asarray(model_variances, dtype=float)
    oob_residual_variances = np.asarray(oob_residual_variances, dtype=float)
    if model_variances.ndim < 2:
        raise ValueError(
            "model_variances must have model and candidate axes."
        )
    expected_oob_shape = (
        model_variances.shape[0],
        *model_variances.shape[2:],
    )
    if oob_residual_variances.shape != expected_oob_shape:
        raise ValueError(
            "oob_residual_variances must have shape "
            "(n_models, *target_axes)."
        )

    clean_p = np.nan_to_num(
        model_variances, nan=0.0, posinf=0.0, neginf=0.0
    )
    clean_p = np.clip(clean_p, 0.0, np.inf)
    p_reference = np.median(clean_p, axis=1)
    valid_oob = np.isfinite(oob_residual_variances) & (
        oob_residual_variances >= 0.0
    )
    calibrated = np.where(
        valid_oob,
        np.maximum(oob_residual_variances - p_reference, eps),
        np.nan,
    )

    fallback = np.empty(calibrated.shape[1:], dtype=float)
    for index in np.ndindex(fallback.shape):
        model_values = calibrated[(slice(None),) + index]
        finite_values = model_values[np.isfinite(model_values)]
        if finite_values.size:
            fallback[index] = float(np.median(finite_values))
        else:
            p_values = p_reference[(slice(None),) + index]
            fallback[index] = max(float(np.median(p_values)), eps)

    calibrated = np.where(
        np.isfinite(calibrated),
        calibrated,
        np.expand_dims(fallback, axis=0),
    )
    return np.expand_dims(calibrated, axis=1), p_reference, valid_oob


def regularized_precision_weights(
    model_variances,
    shrinkage=0.1,
    relative_floor=1e-6,
    absolute_floor=1e-12,
):
    """Return inverse-variance weights protected from zero-variance monopoly."""
    variances = np.asarray(model_variances, dtype=float)
    if variances.ndim < 1 or variances.shape[0] == 0:
        raise ValueError("model_variances must have a non-empty model axis.")
    if not isinstance(shrinkage, numbers.Real) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1.")

    variances = np.nan_to_num(
        variances, nan=np.inf, posinf=np.inf, neginf=np.inf
    )
    finite_positive = np.where(
        np.isfinite(variances) & (variances > 0.0), variances, np.nan
    )
    with np.errstate(all='ignore'):
        scale = np.nanmedian(finite_positive, axis=0, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    floor = np.maximum(absolute_floor, relative_floor * scale)
    stabilized = np.maximum(variances, floor)

    precision = np.where(np.isfinite(stabilized), 1.0 / stabilized, 0.0)
    denominator = np.sum(precision, axis=0, keepdims=True)
    uniform = np.full_like(precision, 1.0 / variances.shape[0])
    normalized = np.divide(
        precision,
        denominator,
        out=uniform.copy(),
        where=denominator > 0.0,
    )
    return (1.0 - shrinkage) * normalized + shrinkage * uniform


def regularized_variance_weights(model_variances, shrinkage=0.1):
    """Return variance-proportional weights with the same uniform shrinkage."""
    variances = np.asarray(model_variances, dtype=float)
    if variances.ndim < 1 or variances.shape[0] == 0:
        raise ValueError("model_variances must have a non-empty model axis.")
    if not isinstance(shrinkage, numbers.Real) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between 0 and 1.")

    variances = np.nan_to_num(
        variances, nan=0.0, posinf=0.0, neginf=0.0
    )
    variances = np.clip(variances, 0.0, np.inf)
    denominator = np.sum(variances, axis=0, keepdims=True)
    uniform = np.full_like(variances, 1.0 / variances.shape[0])
    normalized = np.divide(
        variances,
        denominator,
        out=uniform.copy(),
        where=denominator > 0.0,
    )
    return (1.0 - shrinkage) * normalized + shrinkage * uniform

def kalman_fusion_weights(
    model_variances,
    observation_noise_variances,
    shrinkage=0.1,
    eps=1e-12,
):
    """Return KF precision weights, rKF gain weights, and Kalman gain.

    KF uses the total error P + R. The reverse/exploration branch normalizes
    K = P / (P + R), so irreducible OOB noise cannot be rewarded as if it were
    reducible model uncertainty.
    """
    model_variances = np.asarray(model_variances, dtype=float)
    observation_noise_variances = np.asarray(
        observation_noise_variances, dtype=float
    )
    try:
        model_variances, observation_noise_variances = np.broadcast_arrays(
            model_variances,
            observation_noise_variances,
        )
    except ValueError as exc:
        raise ValueError(
            "observation_noise_variances must broadcast to model_variances."
        ) from exc

    model_variances = np.clip(
        np.nan_to_num(
            model_variances, nan=0.0, posinf=0.0, neginf=0.0
        ),
        0.0,
        np.inf,
    )
    observation_noise_variances = np.clip(
        np.nan_to_num(
            observation_noise_variances,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
        np.inf,
    )
    total_error_variances = np.maximum(
        model_variances + observation_noise_variances,
        eps,
    )
    kalman_gain = np.divide(
        model_variances,
        total_error_variances,
        out=np.zeros_like(model_variances),
        where=total_error_variances > 0.0,
    )
    kf_weights = regularized_precision_weights(
        total_error_variances,
        shrinkage=shrinkage,
    )
    rkf_weights = regularized_variance_weights(
        kalman_gain,
        shrinkage=shrinkage,
    )
    return kf_weights, rkf_weights, kalman_gain, total_error_variances
