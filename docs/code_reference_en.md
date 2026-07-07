# TMM_BO Usage and Developer Reference

This document is the English companion to `code_reference_zh.md`. It explains how to run the project, how the Bayesian optimization pipeline is organized, what each major module does, and what the main functions accept and return.

## Audience

Use this document if you want to:

- run the provided benchmark or closed-pool workflows;
- call `BayesianOptimization` from an external script;
- understand the data flow from CSV input to recommended candidates;
- extend surrogate models, samplers, or acquisition functions.

The main source code lives under `src/`. The unified command-line entry point is `main.py`. The files `math_test.py`, `data_test_closedpool.py`, and `data_test_closedpool_targetwindow.py` are runnable workflow examples.

## Quick Start

Activate the project environment:

```bash
conda activate Bo_project
```

If `conda activate` is not available in your current shell, initialize conda with your local shell hook first.

Inspect the CLI:

```bash
python main.py --help
```

Run the synthetic benchmark workflow:

```bash
python main.py math
```

Run the generic closed-pool workflow:

```bash
python main.py data-closedpool
```

Run the closed-pool workflow with target-window constraints:

```bash
python main.py data-target-window
```

Preview the routing without starting a long job:

```bash
python main.py math --dry-run
python main.py data-closedpool --dry-run
python main.py data-target-window --dry-run
```

Call `BayesianOptimization` directly from the CLI:

```bash
python main.py bo \
  --data-file Data/bm_data.csv \
  --targets bm_target \
  --model-names Lasso Ridge ElasticNet MLPRegressor LightGBM XGBoost \
  --optimization-goal maximize \
  --scaler-method minmax \
  --sampling-method differential_evolution \
  --num-candidate 10000 \
  --n-samples 100 \
  --iterations 500 \
  --batch-size 10
```

## Repository Layout

```text
TMM_Bo_project/
├── main.py                                  # Unified CLI entry point
├── math_test.py                             # Open-pool synthetic benchmark workflow
├── data_test_closedpool.py                  # Generic closed-pool workflow
├── data_test_closedpool_targetwindow.py     # Closed-pool workflow with target windows
├── src/
│   ├── bayesian_optimization.py             # Main BO orchestrator
│   ├── io.py                                # CSV loading, cleaning, scaling, inverse scaling
│   ├── surrogate_model.py                   # Surrogate model wrappers and hyperparameter search
│   ├── evaluation.py                        # Bootstrap training, evaluation, persistence
│   ├── sampling.py                          # Candidate generation and optimization samplers
│   ├── acquisition_function.py              # Acquisition scoring and batch selection
│   ├── multi_task.py                        # Multi-task BO prototype
│   └── multi_fidelity.py                    # Multi-fidelity BO prototype
├── docs/
│   ├── usage.md                             # CLI usage and equation notes
│   ├── code_reference_en.md                 # This document
│   └── code_reference_zh.md                 # Chinese version
├── Data/                                    # Example/test data
└── model_weights/                           # Runtime model/scaler outputs
```

## Input Data

### Training CSV

Training data is loaded by `IOManager.read_data()`.

Requirements:

- each row is one sample;
- target columns are passed through `target_props` or `--targets`;
- feature columns should be numeric;
- if `feature_props` / `--features` is omitted, all non-target numeric columns are used;
- non-numeric feature columns can be dropped automatically;
- binary non-numeric target columns can be encoded as 0/1.

Example:

```csv
x0,x1,x2,bm_target
0.1,0.2,0.3,-1.23
0.4,0.5,0.6,-0.87
```

### Candidate-Pool CSV

Closed-pool or candidate-pool runs can use `candidate_file`. Candidate files may omit target columns; `read_candidate_data()` prints a message and continues with feature loading.

### Optimization Direction

If `optimization_goal='maximize'`, targets are used as-is.

If `optimization_goal='minimize'`, `BayesianOptimization.__init__()` negates `y` internally so the rest of the pipeline can still maximize.

## Runtime Outputs

Common output files are:

- `model_weights/*.pkl`: bootstrap models, optimized parameters, residuals, ELPD/CRPS statistics, and OOB diagnostics;
- `model_weights/scalerNone.pkl`: saved feature and target scalers;
- `suggested_samples.csv`: recommended samples in original feature scale;
- `suggested_samples_indexes.csv`: recommended indexes;
- `suggested_samples_original.csv`: original candidate rows when `candidate_file` is used;
- `Data/bm_data.csv`: the synthetic benchmark workflow appends evaluated samples here;
- `bo.log`: run log;
- `performance_record.txt`: closed-pool iteration log.

## Main Pipeline

### Open-Pool or Candidate-Pool Optimization: `BayesianOptimization.optimize()`

Typical caller: `math_test.py` or `python main.py bo ...`.

High-level flow:

1. `IOManager.read_data()` reads the training CSV into `X` and `y`.
2. `IOManager.standardize_data(..., if_train=True)` fits scalers and scales training/candidate arrays.
3. `ModelEvaluator.evaluate()` trains bootstrap ensembles for each target and each model.
4. Individual estimators are wrapped by `SurrogateModel`; hyperparameters can be searched with `hyperparameter_optimization()`.
5. `Sampler.generate_candidates_parallel()` proposes candidate points in the scaled feature space.
6. `AcquisitionFunction.select_next()` predicts candidate performance and selects the next batch.
7. `optimize()` inverse-transforms selected samples and returns them.

Return value:

```python
samples_next, next_indexes = BO.optimize(...)
```

- `samples_next`: `np.ndarray`, usually `[batch_size, n_features]`, in original feature scale.
- `next_indexes`: selected candidate indexes.

### Closed-Pool Simulation: `BayesianOptimization.close_pooling_test()`

Typical caller: `data_test_closedpool.py`.

High-level flow:

1. Compute a normalized product score from target values, or from distance to `select_region` target windows.
2. Draw an initial training set from lower-scoring samples.
3. Treat the remaining rows as a candidate pool with hidden target values.
4. Train models, compute acquisition scores, and select a batch.
5. Move selected candidates from the pool into the training set.
6. Stop when the threshold is reached or `n_iter` is exhausted.

Main outputs:

- `performance_record.txt`;
- `model_weights/{iteration}/data.pickle`;
- `model_weights/final_train_data.csv` when the target threshold is reached.

## Command-Line Entry Point: `main.py`

### `python main.py math`

Runs `math_test.py`.

Inputs: none required from the CLI; benchmark dimension, target function, batch size, and model list are defined inside the script.

Outputs: updated benchmark CSV, model weights, suggested samples, suggested indexes, and logs.

### `python main.py data-closedpool`

Runs `data_test_closedpool.py`.

Inputs: the script reads the first `.csv` file in the current directory and uses columns whose names start with `target_` as targets.

Outputs: closed-pool logs, model weights, and performance records.

### `python main.py data-target-window`

Runs `data_test_closedpool_targetwindow.py`.

Inputs: similar to `data-closedpool`, but the script includes a `select_region` dictionary defining desired target windows.

Outputs: closed-pool logs, model weights, and performance records.

### `python main.py bo ...`

Calls `BayesianOptimization` directly from CLI arguments.

Important inputs:

- `--data-file`: training CSV path;
- `--targets`: one or more target columns;
- `--features`: optional explicit feature columns;
- `--model-names`: surrogate model names;
- `--optimization-goal`: `maximize` or `minimize`;
- `--scaler-method`: `standard` or `minmax`;
- `--close-pool`: enable closed-pool behavior;
- `--close-pooling-test`: run `close_pooling_test()` instead of `optimize()`.

Outputs:

- regular `bo`: prints recommended samples and indexes;
- `bo --close-pooling-test`: writes closed-pool records and model artifacts.

## Module Reference

## `src.bayesian_optimization`

### `initialize_ray()`

Initializes a local Ray runtime.

Inputs: no explicit arguments. The function reads system memory, `/dev/shm`, and optional SLURM environment variables.

Output: path to the per-run Ray temp directory.

Side effects:

- starts Ray;
- creates `tmp/ray_<pid>` under the current project directory when possible;
- falls back to the system temp directory if the project-local tmp directory cannot be created.

### `cleanup_ray_runtime(ray_temp_dir=None)`

Stops Ray and removes the per-run Ray temporary directory.

Inputs:

- `ray_temp_dir`: path returned by `initialize_ray()`.

Output: none.

### `class BayesianOptimization`

The main orchestrator connecting IO, model evaluation, candidate generation, and acquisition selection.

#### `__init__(target_props, data_file=None, feature_props=None, drop_columns=None, optimization_goal='maximize', scaler_method='standard', model_list=None, model_path='.../model_weights', stacking=False, cross_val=False, acq_method='ucb', feature_lb=None, feature_ub=None, candidate_file=None, close_pool=False, close_pool_initial_samples=10, close_pool_threshold=None, select_region=None, uni_hyperparameter=False)`

Inputs:

- `target_props`: list of target column names;
- `data_file`: training CSV path;
- `feature_props`: optional feature column list;
- `drop_columns`: columns to remove before training;
- `optimization_goal`: `maximize` or `minimize`;
- `scaler_method`: `standard` or `minmax`;
- `model_list`: model names used by `SurrogateModel`;
- `model_path`: directory for model/scaler artifacts;
- `stacking`: enable stacking ensemble;
- `cross_val`: use cross-validation inside bootstrap evaluation;
- `acq_method`: acquisition method name;
- `feature_lb`, `feature_ub`: open-pool feature bounds;
- `candidate_file`: optional candidate-pool CSV;
- `close_pool`: initialize closed-pool thresholds and split metadata;
- `close_pool_initial_samples`: initial training size for closed-pool simulation;
- `close_pool_threshold`: stopping threshold;
- `select_region`: target-window dictionary, e.g. `{target: [low, high]}`;
- `uni_hyperparameter`: use a unified hyperparameter strategy.

Output: object instance.

Important attributes:

- `self.X`, `self.y`: training features and targets;
- `self.X_cand`: candidate-pool features or `None`;
- `self.io_manager`: `IOManager` instance;
- `self.model_list`: model names;
- `self.select_region`: target window array or `None`;
- `self.ray_temp_dir`: per-run Ray temp directory.

#### `cleanup_runtime()`

Stops Ray and removes the temporary Ray directory created by the instance.

Inputs: none.

Output: none.

#### `compute_normalized_product(y_values)`

Normalizes each target column with stored min/max values and multiplies columns row-wise.

Input:

- `y_values`: array `[n_samples, n_targets]`.

Output:

- `product`: array `[n_samples]`; larger is better.

#### `compute_normalized_product_region(y_values)`

Scores samples by closeness to the center of `select_region`.

Input:

- `y_values`: array `[n_samples, n_targets]`.

Output:

- `product`: array `[n_samples]`; larger means closer to the desired target window.

#### `custom_train_test_split(random_state=None)`

Builds the initial closed-pool train/candidate split.

Input:

- `random_state`: optional random seed.

Output:

```python
X_train, X_candidate, y_train, y_candidate
```

#### `close_pooling_test(...)`

Runs a closed-pool Bayesian optimization simulation.

Key inputs:

- `n_bootstrap_sample_nums`: bootstrap ensemble size;
- `n_iter`: maximum BO iterations;
- `batch_size`: samples selected per iteration;
- `hpar`: acquisition hyperparameter;
- `save_all_info`: save per-iteration model/data snapshots;
- `sampling_method`: candidate generation method;
- `num_candidate`: number of generated/screened candidates;
- `n_samples`: sampler population/sample count;
- `iterations`: sampler iteration count;
- `candidate_sampling`: whether to pre-screen candidates with `Sampler`;
- `diversity_method`: enable diversity-aware selection;
- `use_data_correlation`: use target residual correlations;
- `use_model_correlation`: add a product/consensus auxiliary target.

Output: no direct return. Writes performance records, model files, and final training data when the threshold is reached. Cleans Ray tmp at normal completion.

#### `optimize(...)`

Runs open-pool or candidate-pool BO and returns a recommended batch.

Key inputs:

- `batch_size`: number of recommendations;
- `n_bootstrap_sample_nums`: bootstrap ensemble size;
- `sampling_method`: sampler method;
- `num_candidate`: generated/screened candidate count;
- `n_samples`: sampler population/sample count;
- `iterations`: sampler iteration count;
- `hpar`: acquisition hyperparameter;
- `if_train`: retrain models or reuse existing artifacts;
- `candidate_sampling`: pre-screen candidate pool;
- `n_random_models`: bootstrap models sampled for ensemble prediction;
- `seperate`: generate candidates separately per model;
- `diversity_method`: diversity-aware selection;
- `use_data_correlation`: use residual correlations;
- `use_model_correlation`: add a product/consensus auxiliary target.

Output:

```python
samples_next, next_indexes
```

- `samples_next`: selected samples in original feature scale;
- `next_indexes`: selected candidate indexes.

## `src.io`

### `class IOManager`

Handles CSV loading, cleaning, feature/target scaling, inverse scaling, and scaler persistence.

#### `__init__(root=None, method='standard', file_path=None)`

Inputs:

- `root`: data root directory, default current working directory;
- `method`: `standard` or `minmax`;
- `file_path`: scaler/model artifact directory.

Output: object instance.

#### `read_data(file_name, target_props, feature_props=None, drop_columns=None, descriptor_type='magpie', handle_null=True, drop_non_numeric=True)`

Reads training data.

Inputs:

- `file_name`: CSV path;
- `target_props`: target columns;
- `feature_props`: optional feature columns;
- `drop_columns`: optional columns to remove;
- `handle_null`: handle nulls automatically;
- `drop_non_numeric`: drop non-numeric features and encode binary categorical targets.

Output:

```python
X, y
```

- `X`: `[n_samples, n_features]`;
- `y`: `[n_samples, n_targets]`.

#### `read_candidate_data(...)`

Reads candidate-pool features.

Output:

- `X`: `[n_candidates, n_features]`.

#### `handle_null_values(data, target_props, drop_non_numeric, if_train_data=True)`

Cleans nulls and non-numeric columns.

Output:

```python
cleaned_data, non_numeric_columns
```

#### `standardize_data(X=None, y=None, cand_X=None, cand_y=None, minmax_feature_range=(0, 1), if_train=False, data_id=None)`

Fits or loads scalers and transforms arrays.

Output: returns transformed arrays in the same logical order as non-`None` inputs. A single output is returned directly; multiple outputs are returned as a tuple.

#### `inverse_transform_X(X_scaled)` / `inverse_transform_y(y_scaled)`

Converts scaled arrays back to original scale.

#### `save_predictions(predictions, file_name)`

Writes a prediction vector to CSV.

## `src.surrogate_model`

### `class SurrogateModel`

A unified wrapper around sklearn models, XGBoost, LightGBM, CatBoost, BoTorch GP, KAN, and FastKAN.

#### `__init__(model_name=None, params=None)`

Inputs:

- `model_name`: model identifier, such as `Ridge`, `Lasso`, `ElasticNet`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForest`, `SVR`, `MLPRegressor`, `ExtraTreesRegressor`, `XGBoost`, `LightGBM`, `CatBoost`, `GP_cpu`, `GP_gpu`, `KAN`, or `FastKAN`;
- `params`: model constructor parameters.

Output: object instance with `self.model` initialized.

#### `fit(X, y)`

Fits the underlying model.

Inputs:

- `X`: `[n_samples, n_features]`;
- `y`: `[n_samples]` or `[n_samples, n_targets]`.

Output: no direct return.

#### `predict(X)`

Returns model predictions.

Input:

- `X`: `[n_samples, n_features]`.

Output:

- prediction array, usually `[n_samples]`.

#### `predict_proba(X)`

Returns class probabilities when available. If the underlying model has no `predict_proba`, regression outputs are clipped into `[0, 1]` and converted into a binary probability matrix.

Output: `[n_samples, n_classes]`.

#### `predict_std(X)` / `predict_var(X)`

Returns predictive uncertainty for models that support it, mainly `GP_cpu` and `GP_gpu`.

Output:

- `predict_std`: `[n_samples]` standard deviations;
- `predict_var`: `[n_samples]` variances.

Unsupported models raise `AttributeError`.

### `hyperparameter_optimization(...)`

Uses Optuna to search model hyperparameters.

Inputs:

- `model_name`, `X_train`, `y_train`;
- `cls`: classification flag;
- `n_trials`: number of Optuna trials;
- `cv_n_splits`: cross-validation folds;
- `alpha`: generalization-gap penalty;
- `expect_N`, `eta`: resource-budget controls.

Output:

- `best_params`: dictionary. `GP_gpu` returns `{}`.

### `estimate_budget(...)`

Estimates the resource budget from unsupervised dimension, supervised dimension, and intrinsic dimension.

Output:

```python
budget, d_eff, d_all
```

## `src.evaluation`

### `class ModelEvaluator`

Trains bootstrap ensembles, computes OOB metrics, stores model artifacts, trains stacking models, and computes residual correlations.

#### `__init__(X_train, y_train, file_path=None, bs_sample_number=None, optimization_goal='maximize')`

Inputs:

- scaled training features and targets;
- model artifact directory;
- optional bootstrap sample size;
- optimization direction.

Output: object instance. Also computes clustering labels for bootstrap weighting.

#### `evaluate(model_names, num_target, n_bootstrap_sample_nums, cls=False, use_full_eval=False, cross_val=False, uni_hyper=False)`

Trains and evaluates all requested models for one target column.

Output:

```python
model_results: dict
```

Each model entry contains models, errors, residuals, ELPD values, OOB sizes, ELPD-per-point scores, CRPS values, and OOB variance diagnostics.

#### `bootstrap_evaluation(...)`

Trains a bootstrap ensemble for one model and one target.

Output:

```python
[models, errors, residuals, elpds, oob_sizes, elpd_scores,
 elpd_per_point_mean, crps, oob_var_list, oob_mad_var_list,
 oob_n_list, oob_var_mean, oob_mad_var_mean]
```

#### `_train_model(...)`

Ray remote function that trains one bootstrap model and computes OOB metrics.

Output dictionary keys include:

- `model`;
- `error`;
- `residual`;
- `elpd`;
- `crps`;
- `oob_size`;
- `oob_var`;
- `oob_mad_var`;
- optionally `nu` for Student-t regression.

#### `train_stacking_model(...)`

Builds a `StackingRegressor` or `StackingClassifier` over model ensembles.

Output:

```python
stacking_model, base_model_errors, residuals
```

#### `evaluate_with_stacking(...)`

Runs `evaluate()`, trains stacking, saves `stacking_models_{num_target}.pkl`, and returns the augmented model result dictionary.

#### `calculate_and_save_residual_correlation(model_names, stacking=False)`

Loads saved residuals, computes residual correlation matrices, and stores them for acquisition use.

### `class ClusterBootstrapSampler`

Computes clustering-based bootstrap sampling probabilities.

Main methods:

- `compute_bootstrap_probabilities_clustering(X, enable_refine=False)` returns cluster labels;
- `compute_bootstrap_probabilities_prob(labels, y)` returns per-sample bootstrap probabilities.

### `class AdaptiveClusterer`

Compares clustering configurations and returns the best labels for bootstrap sampling.

Main method:

- `find_best_clustering(X)` returns clustering labels or clustering metadata.

## `src.sampling`

### `class Sampler`

Generates candidate points in scaled feature space.

#### `__init__(scaler, feature_bounds=None)`

Inputs:

- fitted scaler;
- optional scaled feature bounds `[2, n_features]`.

#### Basic samplers

- `sobol_sampling(feature_dim, num_candidate=50)` returns Sobol points;
- `gaussian_sampling(feature_dim, mean=None, std_dev=None, num_candidate=10)` returns Gaussian samples;
- `bernoulli_sampling(feature_dim, p=0.5, n_test=1, num_candidate=10)` returns binomial samples.

Output shape: `[num_candidate, feature_dim]`.

#### Optimization samplers

The following methods use `model.predict(X)` as a score and return the best candidate points:

- `monte_carlo_sampling(...)`;
- `genetic_algorithm_sampling(...)`;
- `particle_swarm_sampling(...)`;
- `simulated_annealing_sampling(...)`;
- `differential_evolution_sampling(...)`;
- `artificial_fish_swarm_sampling(...)`.

Common inputs:

- `model`: object with `predict(X)`;
- `feature_dim`: number of features;
- `num_candidate`: number of returned candidates;
- `candidate_list`: optional real candidate pool for nearest-neighbor mapping.

Common output:

- candidate matrix `[num_candidate, feature_dim]`.

#### `generate_candidates(...)`

Serial candidate-generation dispatcher.

Input `method` can be `gaussian`, `bernoulli`, `monte_carlo`, `genetic_algorithm`, `particle_swarm`, `differential_evolution`, `artificial_fish_swarm`, `simulated_annealing`, `ant_colony`, or `immune_algorithm`.

Output: candidate matrix.

#### `generate_candidates_parallel(...)`

Parallel candidate generation using Ray. It can generate candidates per model and add Sobol points for broader coverage.

Output:

- `candidate_X_scaled`: scaled candidate matrix.

### `RandomizedAbstractSurrogateModel`

Samples bootstrap models from trained model results and exposes a unified `predict(X)` method for samplers.

Output of `predict(X)`: a scalar score per candidate.

### `map_to_candidate_list*`

Maps continuous candidate samples back to the nearest unused rows in a real candidate pool.

Output:

```python
mapped_samples, selected_indices
```

## `src.acquisition_function`

### `class AcquisitionFunction`

Computes acquisition scores and selects candidate batches.

#### `__init__(hpar=0.1)`

Input:

- `hpar`: acquisition hyperparameter, used for exploration pressure.

#### `ucb(mean, std, hpara=None)`

Formula:

```text
UCB = mean + hpar * std
```

Output: UCB score array.

#### `ei(mean, std, y_best, hpara=None)`

Computes expected improvement.

Output: EI score array.

#### `pi(mean, std, y_best, hpara=None)`

Computes probability of improvement.

Output: PI score array.

#### `hypervolume(points, reference_point=None, batch_size=20000)`

Computes Pareto/hypervolume-related scores for multi-objective selection.

Output: hypervolume contribution scores.

#### `select_next(...)`

Core batch-selection function.

Inputs include:

- `method`: acquisition method;
- `X_candidate`: candidate matrix `[n_candidates, n_features]`;
- `model_name_list`: model names;
- `num_of_targets`: target count;
- `model_path`: model artifact directory;
- `batch_size`: returned batch size;
- `X_train`, `y_value`: current training data;
- `model_result`: output from `ModelEvaluator.evaluate()`;
- `stack`: use stacking models;
- `select_region`: scaled target window;
- `diversity_method`: diversity control;
- `optimization_goal`: maximize/minimize;
- `use_correlation`, `use_model_correlation`: correlation controls;
- `train_clsuter_labels`: clustering labels.

Output:

- `next_indexes`: selected candidate indexes, length `batch_size`.

### Acquisition helper functions

- `compute_hv_contributions(pareto_points, reference_point)` returns hypervolume contributions.
- `estimate_density_and_spread(...)` returns density/spread diagnostics for diversity.
- `safe_cv(std, mean, rel_floor=0.05, abs_floor=1e-12)` returns a stable coefficient of variation.
- `mix_ratio_from_scores(scores, method='gs', ...)` returns disagreement/mix ratios using Gini-Simpson or entropy.
- `compute_structure_with_rigidity(...)` returns structural smoothness/rigidity scores and diagnostics.

## `src.multi_task`

### `class MultiTaskBayesianOptimization`

Prototype for multi-task BO. It models a main target and correlated auxiliary targets, then uses multi-task acquisition selection.

Main methods:

- `__init__(data_file, Main_props, correlated_props, ...)` initializes data, targets, models, and sampling settings;
- `close_pooling_test(...)` runs a closed-pool multi-task simulation.

## `src.multi_fidelity`

### `class MultiFidelityBayesianOptimization`

Prototype for multi-fidelity BO. It models high-fidelity and low-fidelity targets and considers evaluation costs.

Main methods:

- `__init__(data_file, HF_props, LF_props, ..., HFcost=10, LFcost=[1], ...)` initializes fidelity-specific targets and costs;
- `close_pooling_test(...)` runs a closed-pool multi-fidelity simulation.

## Extension Guide

### Add a New Surrogate Model

1. Add the model name to `SurrogateModel._initialize_model()`.
2. Add a hyperparameter search branch in `hyperparameter_optimization()` if needed.
3. Ensure the model supports `fit(X, y)` and `predict(X)`.
4. Add `predict_std()` support if the model can provide uncertainty.

### Add a New Sampling Method

1. Add `xxx_sampling()` to `Sampler`.
2. Add a dispatcher branch in `generate_candidates()`.
3. Add a Ray dispatcher branch in `generate_candidates_ray()`.
4. Return a NumPy array with shape `[num_candidate, feature_dim]`.
5. Support `candidate_list` if the method should work with closed pools.

### Add a New Acquisition Method

1. Add a scoring function to `AcquisitionFunction`.
2. Add a method branch or blending logic in `select_next()`.
3. Define required inputs: mean, std, best value, ensemble variance, residual correlations, target windows, etc.
4. Return sortable candidate scores and final candidate indexes.

## FAQ

### Why is `math_test.py` slow?

The default benchmark is 20-dimensional, uses multiple surrogate models, and can run many BO iterations. Each iteration may train bootstrap ensembles and run candidate optimization.

### Why does Ray start multiple times?

Each `BayesianOptimization` instance initializes Ray. Some workflows build a new BO object per outer iteration.

### Which model list is a reasonable default?

A practical default used by the examples is:

```python
['Lasso', 'Ridge', 'ElasticNet', 'MLPRegressor', 'LightGBM', 'XGBoost', 'KNeighborsRegressor', 'DecisionTreeRegressor']
```

### When should `candidate_sampling=True` be used?

Use it when the candidate space is large or continuous. The sampler first reduces the candidate set, then acquisition scoring chooses the final batch. For small closed pools, direct acquisition scoring with `candidate_sampling=False` is often simpler.

### What is `select_region`?

`select_region` defines target windows rather than pure maximization. Example:

```python
select_region = {
    'target_gap': [1.45, 1.55],
    'target_magnetization': [19.5, 20.5],
}
```

The code scores candidates by closeness to the center of each target window.
