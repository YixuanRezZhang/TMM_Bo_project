import os
import time
import random
import numpy as np
import pandas as pd
import torch

from scipy.stats import qmc
from src.bayesian_optimization import BayesianOptimization


# ============================================================
# 0. Reproducibility utilities
# ============================================================

def set_global_seed(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Note:
    Full determinism may still not be guaranteed when using Ray,
    LightGBM, XGBoost, Optuna, or other parallel backends.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_random_rotation(dim: int, seed: int = 42) -> np.ndarray:
    """
    Generate a reproducible random orthogonal rotation matrix.

    The QR decomposition has sign ambiguity, so we explicitly fix
    the sign of each column to make the result stable for a given seed.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)

    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs

    return Q


def make_shift_vector(
    dim: int,
    lb: np.ndarray,
    ub: np.ndarray,
    seed: int = 42,
    shift_ratio: float = 0.3,
) -> np.ndarray:
    """
    Generate a shifted optimum location inside the search domain.

    shift_ratio controls how far the new optimum is moved away from
    the domain center:

    - 0.0: no shift, optimum remains at the center
    - 0.3: moderate shift
    - 0.8: strong shift, but still inside the domain
    """
    rng = np.random.default_rng(seed)

    center = 0.5 * (lb + ub)
    half_width = 0.5 * (ub - lb)

    direction = rng.uniform(-1.0, 1.0, size=dim)
    x_opt = center + shift_ratio * half_width * direction

    return x_opt


def transform_x(
    x: np.ndarray,
    shift: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply shift and rotation to the input vector.

    The transformed coordinate is

        z = R.T @ (x - shift)

    For Ackley and Rastrigin, whose original optimum is at z = 0,
    this makes the new optimum located at x = shift.

    For Schwefel, this transformation is usually not recommended,
    because the original optimum is not at the origin.
    """
    z = np.asarray(x, dtype=float)

    if shift is not None:
        z = z - shift

    if rotation is not None:
        z = rotation.T @ z

    return z


def make_benchmark_func(
    base_func,
    dim: int,
    shift: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
):
    """
    Wrap a benchmark function with optional shift and rotation.
    """
    def wrapped_func(x, dim=dim):
        z = transform_x(x, shift=shift, rotation=rotation)
        return base_func(z, dim=dim)

    return wrapped_func


# ============================================================
# 1. Benchmark functions
# ============================================================

def ackley_np(x=None, dim=10, a=20, b=0.2, c=2 * np.pi):
    """
    Ackley function in NumPy.

    Original global minimum:
        x = [0, ..., 0], f(x) = 0

    This implementation returns the negative value, so the task becomes
    a maximization problem with the maximum value equal to 0.
    """
    if x is None:
        x = np.zeros(dim)

    x = np.asarray(x, dtype=float)
    d = x.shape[0]

    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(c * x))

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)

    return -(term1 + term2 + a + np.e)


def rastrigin_np(x=None, dim=10):
    """
    Rastrigin function in NumPy.

    Original global minimum:
        x = [0, ..., 0], f(x) = 0

    This implementation returns the negative value, so the task becomes
    a maximization problem with the maximum value equal to 0.
    """
    if x is None:
        x = np.zeros(dim)

    x = np.asarray(x, dtype=float)
    d = x.shape[0]

    return -(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def schwefel_np(x=None, dim=10):
    """
    Schwefel function in NumPy.

    Standard global minimum in the conventional domain [-500, 500]^d:
        x_i ≈ 420.968746, f(x) ≈ 0

    This implementation returns the negative value, so the task becomes
    a maximization problem with the maximum value approximately equal to 0.

    Important:
    Schwefel is usually not shifted or rotated in this script, because
    its optimum is not located at the origin and because the function may
    show better-than-standard values outside the conventional domain.
    """
    if x is None:
        x = np.full(dim, 420.968746)

    x = np.asarray(x, dtype=float)
    d = x.shape[0]

    return -(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


# ============================================================
# 2. Benchmark configuration
# ============================================================

SEED = 42
set_global_seed(SEED)

root = os.getcwd()
data_dir = os.path.join(root, "Data")
data_file = os.path.join(data_dir, "bm_data.csv")

force_regenerate_data = True

d = 20
batch_size = 10
n_samples = 10
test_iteration = 100

# ------------------------------------------------------------
# Select benchmark
# ------------------------------------------------------------

benchmark_name = "ackley"
# benchmark_name = "rastrigin"
# benchmark_name = "schwefel"

if benchmark_name == "ackley":
    base_func = ackley_np
    bounds = [-32.768, 32.768]

    use_shift = True
    use_rotation = True

elif benchmark_name == "rastrigin":
    base_func = rastrigin_np
    bounds = [-5.12, 5.12]

    use_shift = True
    use_rotation = True

elif benchmark_name == "schwefel":
    base_func = schwefel_np
    bounds = [-500, 500]

    # For Schwefel, shift and rotation are disabled by default.
    # This avoids moving the true optimum outside the search domain.
    use_shift = False
    use_rotation = False

else:
    raise ValueError(f"Unknown benchmark_name: {benchmark_name}")

lb = np.full(d, bounds[0], dtype=float)
ub = np.full(d, bounds[1], dtype=float)

# ------------------------------------------------------------
# Build shift and rotation
# ------------------------------------------------------------

shift = make_shift_vector(
    dim=d,
    lb=lb,
    ub=ub,
    seed=SEED + 1,
    shift_ratio=0.3,
) if use_shift else None

rotation = make_random_rotation(
    dim=d,
    seed=SEED + 2,
) if use_rotation else None

func = make_benchmark_func(
    base_func=base_func,
    dim=d,
    shift=shift,
    rotation=rotation,
)

print("=" * 80)
print("Benchmark setting")
print("=" * 80)
print(f"benchmark_name = {benchmark_name}")
print(f"dim            = {d}")
print(f"bounds         = {bounds}")
print(f"use_shift      = {use_shift}")
print(f"use_rotation   = {use_rotation}")

if shift is not None:
    print(f"shift      = {shift}")
    print(f"f(shift)       = {func(shift, dim=d)}")

if benchmark_name == "schwefel":
    schwefel_opt = np.full(d, 420.968746)
    print(f"Schwefel original optimum value = {func(schwefel_opt, dim=d)}")

print("=" * 80)


# ============================================================
# 3. Generate initial CSV data
# ============================================================

if force_regenerate_data and os.path.exists(data_file):
    os.remove(data_file)
    print(f"Old data file removed: {data_file}")

if not os.path.exists(data_file):
    os.makedirs(data_dir, exist_ok=True)

    sampler = qmc.LatinHypercube(d=d, seed=SEED)
    X = sampler.random(n=n_samples)
    X = qmc.scale(X, lb, ub)

    bm_target_value = np.array([func(x, dim=d) for x in X])

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
    df["bm_target"] = bm_target_value
    df.to_csv(data_file, index=False)

    print(f"Initial data file generated: {data_file}")
    print(f"Initial best bm_target: {df['bm_target'].max()}")

else:
    print(f"Data file already exists: {data_file}")


# ============================================================
# 4. Bayesian optimization
# ============================================================

target_props = ["bm_target"]

# Large trainset speed:
# 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor',
# 'AdaBoostRegressor', 'ExtraTreesRegressor', 'RandomForest',
# 'XGBoost', 'LightGBM', 'MLPRegressor', 'GradientBoostingRegressor',
# 'SVR', 'KNeighborsRegressor', 'FastKAN'

# Small trainset speed:
# 'Lasso', 'Ridge', 'ElasticNet', 'DecisionTreeRegressor',
# 'KNeighborsRegressor', 'SVR', 'MLPRegressor', 'ExtraTreesRegressor',
# 'XGBoost', 'LightGBM', 'AdaBoostRegressor', 'RandomForest',
# 'GradientBoostingRegressor', 'FastKAN'

model_list = [
    "Lasso",
    "Ridge",
    "ElasticNet",
    "MLPRegressor",
    "LightGBM",
    "XGBoost",
    "KNeighborsRegressor",
    "DecisionTreeRegressor",
]

best_value = -np.inf

for r in range(test_iteration):

    # Reset the seed at each BO iteration for better reproducibility.
    # This may not fully control randomness from parallel backends.
    set_global_seed(SEED + r)

    BO = BayesianOptimization(
        target_props,
        data_file,
        model_list=model_list,
        scaler_method="minmax",
        optimization_goal="maximize",
        stacking=False,
        acq_method="ucb",
        candidate_file=None,
        feature_lb=lb,
        feature_ub=ub,
        close_pool=False,
        uni_hyperparameter=True,
    )

    samples_next, next_indexes = BO.optimize(
        batch_size=batch_size,
        n_bootstrap_sample_nums=10,
        sampling_method="differential_evolution",
        num_candidate=10000,
        n_samples=100,
        iterations=500,
        hpar=0.1,
        if_train=True,
        candidate_sampling=False,
        n_random_models=2,
        seperate=True,
        diversity_method=True,
        use_data_correlation=True,
        use_model_correlation=True,
    )

    # Convert recommended samples to a DataFrame.
    df_new = pd.DataFrame(samples_next, columns=[f"x{i}" for i in range(d)])

    # Evaluate the true benchmark value for the recommended samples.
    bm_value = np.array([func(x, dim=d) for x in samples_next])
    df_new["bm_target"] = bm_value

    # Load old data and append new observations.
    df_old = pd.read_csv(data_file)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)

    best_value = df_combined["bm_target"].max()
    best_index = df_combined["bm_target"].idxmax()

    # Save updated data.
    df_combined.to_csv(data_file, index=False)

    print(
        f"iter_{r}: data saved to {data_file}, "
        f"best_value: {best_value}, best_index: {best_index}"
    )

print("Optimization finished.")