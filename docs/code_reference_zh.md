# TMM_BO 代码使用与开发参考

本文档面向两类使用者：

- 只想运行优化任务的人：重点看「快速开始」「输入/输出文件」「典型调用流程」。
- 想改代码或扩展模型/采样/acquisition 的人：重点看「核心流程」「模块与函数参考」。

当前代码主线在 `src/` 中。`main.py` 是外部一键调用入口，`math_test.py`、`data_test_closedpool.py`、`data_test_closedpool_targetwindow.py` 是示例/任务脚本。

## 快速开始

先激活环境：

```bash
# If your shell is not initialized for conda, first run your local conda shell hook.
conda activate Bo_project
```

查看一键入口：

```bash
python main.py --help
```

运行合成 benchmark：

```bash
python main.py math
```

运行闭池数据任务：

```bash
python main.py data-closedpool
```

运行带目标窗口的闭池任务：

```bash
python main.py data-target-window
```

只检查入口，不真正启动长任务：

```bash
python main.py math --dry-run
python main.py data-closedpool --dry-run
python main.py data-target-window --dry-run
```

直接用 CLI 参数调用 `BayesianOptimization`：

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

## 项目结构

```text
TMM_Bo_project/
├── main.py                                  # 统一 CLI 入口
├── math_test.py                             # 开池 synthetic benchmark 示例
├── data_test_closedpool.py                  # 通用闭池任务脚本
├── data_test_closedpool_targetwindow.py     # 带目标窗口的闭池任务脚本
├── src/
│   ├── bayesian_optimization.py             # 主 BO 编排器
│   ├── io.py                                # CSV 读取、清洗、缩放、反缩放
│   ├── surrogate_model.py                   # 单个 surrogate 模型封装与超参优化
│   ├── evaluation.py                        # bootstrap 训练、模型评估、模型持久化
│   ├── sampling.py                          # 候选点生成/采样器
│   ├── acquisition_function.py              # acquisition 计算与 batch 选择
│   ├── multi_task.py                        # 多任务 BO 原型
│   └── multi_fidelity.py                    # 多保真 BO 原型
├── docs/
│   ├── usage.md                             # 英文使用说明和方程解释
│   └── code_reference_zh.md                 # 本文档
├── Data/                                    # 示例/测试数据
└── model_weights/                           # 运行时模型权重与 scaler 输出
```

## 输入数据格式

### 训练数据 CSV

训练数据由 `IOManager.read_data()` 读取。基本要求：

- 每一行是一个样本。
- 特征列必须是数值列，除非你显式传入 `drop_columns` 删除它们。
- 目标列由 `target_props` / `--targets` 指定。
- 如果不传 `feature_props` / `--features`，代码默认使用所有非目标、非非数值列作为特征。

示例：

```csv
x0,x1,x2,bm_target
0.1,0.2,0.3,-1.23
0.4,0.5,0.6,-0.87
```

### 候选池 CSV

闭池或候选池任务可以传 `candidate_file`。候选池可以没有目标列；如果没有目标列，`read_candidate_data()` 会打印提示但继续读取特征。

### 目标方向

`optimization_goal='maximize'` 时直接优化目标值。

`optimization_goal='minimize'` 时，`BayesianOptimization.__init__()` 会把 `y` 变成 `-y`，使内部逻辑仍然按最大化处理。

## 输出文件

常见运行输出：

- `model_weights/*.pkl`：每个模型、每个目标的 bootstrap 模型集、误差、残差、ELPD/CRPS 等。
- `model_weights/scalerNone.pkl`：特征和目标 scaler。
- `suggested_samples.csv`：推荐样本特征。
- `suggested_samples_indexes.csv`：推荐样本在候选集中的索引。
- `Data/bm_data.csv`：`math_test.py` 会持续追加新评估样本。
- `bo.log`：运行日志。
- `performance_record.txt`：闭池测试的迭代记录。

## 整体代码流程

### 开池优化流程：`BayesianOptimization.optimize()`

典型调用来自 `math_test.py` 或 `python main.py bo ...`。

流程：

1. `IOManager.read_data()` 读取 CSV，得到 `X` 和 `y`。
2. `IOManager.standardize_data(..., if_train=True)` 拟合 scaler，并把训练数据缩放。
3. `ModelEvaluator.evaluate()` 针对每个目标、每个模型做 bootstrap 训练。
4. 每个模型由 `SurrogateModel` 封装，必要时通过 `hyperparameter_optimization()` 用 Optuna 搜索超参。
5. `Sampler.generate_candidates_parallel()` 在连续空间或候选池中生成候选点。
6. `AcquisitionFunction.select_next()` 读取模型预测、评估 acquisition 分数，并选出 batch。
7. `optimize()` 返回推荐样本和索引。

返回：

```python
samples_next, next_indexes = BO.optimize(...)
```

- `samples_next`: `np.ndarray`，形状通常为 `[batch_size, n_features]`，为原始尺度下的推荐样本。
- `next_indexes`: `np.ndarray` 或 list，推荐样本在候选集合中的索引；开池任务可能是候选生成结果中的索引。

### 闭池测试流程：`BayesianOptimization.close_pooling_test()`

典型调用来自 `data_test_closedpool.py`。

流程：

1. 根据目标值或目标窗口计算 normalized product。
2. 用 `custom_train_test_split()` 从低分区域抽初始训练集，其余作为候选池。
3. 每轮训练模型、计算 acquisition、从候选池选 batch。
4. 将选中的候选点移入训练集。
5. 如果达到 `close_pool_threshold`，保存最终训练集并停止。

主要输出：

- `performance_record.txt`
- `model_weights/{iteration}/data.pickle`
- `model_weights/final_train_data.csv`，当达到阈值时生成。

## 主要脚本

### `main.py`

统一命令行入口。

#### `python main.py math`

运行 `math_test.py`。

输入：无必需 CLI 参数；脚本内部定义 benchmark、维度、batch size、模型列表。

输出：更新 `Data/bm_data.csv`，写入 `model_weights/`、`suggested_samples.csv`、`suggested_samples_indexes.csv`。

#### `python main.py data-closedpool`

运行 `data_test_closedpool.py`。

输入：当前目录下第一个 `.csv` 文件；目标列自动取所有以 `target_` 开头的列。

输出：闭池测试记录和模型权重。

#### `python main.py data-target-window`

运行 `data_test_closedpool_targetwindow.py`。

输入：当前目录下第一个 `.csv` 文件；脚本内部有 `select_region`，即目标窗口。

输出：闭池测试记录和模型权重。

#### `python main.py bo ...`

直接通过 CLI 参数实例化 `BayesianOptimization`。

关键输入：

- `--data-file`: 训练 CSV。
- `--targets`: 一个或多个目标列。
- `--features`: 可选，指定特征列。
- `--model-names`: surrogate 模型列表。
- `--optimization-goal`: `maximize` 或 `minimize`。
- `--scaler-method`: `standard` 或 `minmax`。
- `--close-pool`: 使用闭池模式。
- `--close-pooling-test`: 调用 `close_pooling_test()` 而不是 `optimize()`。

输出：

- 普通 `bo` 调用会打印推荐样本和索引。
- 如果加 `--close-pooling-test`，输出为闭池测试文件。

## 模块与函数参考

下面按模块列出主要类和函数。私有 helper 会简要说明；外部最常用的是 `BayesianOptimization`、`IOManager`、`ModelEvaluator`、`SurrogateModel`、`Sampler`、`AcquisitionFunction`。

## `src.bayesian_optimization`

### `initialize_ray()`

作用：初始化本地 Ray，用于并行训练模型和并行预测。

输入：无显式参数；函数会读取系统内存、`/dev/shm`、SLURM 环境变量和 scratch 路径。

输出：无返回值。副作用是启动或连接 Ray runtime。

### `class BayesianOptimization`

主编排器，负责把数据读取、模型训练、采样和 acquisition 选择串起来。

#### `__init__(target_props, data_file=None, feature_props=None, drop_columns=None, optimization_goal='maximize', scaler_method='standard', model_list=None, model_path='.../model_weights', stacking=False, cross_val=False, acq_method='ucb', feature_lb=None, feature_ub=None, candidate_file=None, close_pool=False, close_pool_initial_samples=10, close_pool_threshold=None, select_region=None, uni_hyperparameter=False)`

输入：

- `target_props`: list[str]，目标列名。
- `data_file`: str，训练 CSV 路径。
- `feature_props`: list[str] 或 None，特征列名；None 表示自动使用非目标数值列。
- `drop_columns`: list[str] 或 None，读取时删除的列。
- `optimization_goal`: `'maximize'` 或 `'minimize'`。
- `scaler_method`: `'standard'` 或 `'minmax'`。
- `model_list`: list[str]，模型名称，例如 `Lasso`, `Ridge`, `ElasticNet`, `MLPRegressor`, `LightGBM`, `XGBoost`。
- `model_path`: str，模型和 scaler 保存目录。
- `stacking`: bool，是否训练 stacking 模型。
- `cross_val`: bool，bootstrap 训练时是否使用交叉验证。
- `acq_method`: str，acquisition 名称，例如 `'ucb'`。
- `feature_lb`, `feature_ub`: array-like 或 None，开池连续优化的特征下/上界。
- `candidate_file`: str 或 None，候选池 CSV。
- `close_pool`: bool，是否初始化闭池测试相关阈值。
- `close_pool_initial_samples`: int，闭池初始训练样本数。
- `close_pool_threshold`: float 或 None，闭池停止阈值。
- `select_region`: dict 或 None，目标窗口，例如 `{target_name: [low, high]}`。
- `uni_hyperparameter`: bool，是否用统一超参策略。

输出：构造对象，无显式返回值。

重要属性：

- `self.X`, `self.y`: 训练特征和目标，numpy 数组。
- `self.X_cand`: 候选池特征，若没有 `candidate_file` 则为 None。
- `self.io_manager`: `IOManager` 实例。
- `self.model_list`: 模型名称列表。
- `self.select_region`: 缩放前的目标窗口数组或 None。

#### `compute_normalized_product(y_values)`

作用：把多目标值按初始化时记录的 min/max 归一化，然后逐行相乘，用于闭池任务里的综合目标分数。

输入：

- `y_values`: `np.ndarray`，形状 `[n_samples, n_targets]`。

输出：

- `product`: `np.ndarray`，形状 `[n_samples]`。值越大表示综合表现越好。

#### `compute_normalized_product_region(y_values)`

作用：针对 `select_region` 目标窗口计算综合分数。距离窗口中心越近，分数越高。

输入：

- `y_values`: `np.ndarray`，形状 `[n_samples, n_targets]`。

输出：

- `product`: `np.ndarray`，形状 `[n_samples]`。

#### `custom_train_test_split(random_state=None)`

作用：闭池任务中从候选数据中抽取初始训练集，并把剩余样本作为候选池。

输入：

- `random_state`: int 或 None，随机种子。

输出：

```python
X_train, X_candidate, y_train, y_candidate
```

- `X_train`: 初始训练特征。
- `X_candidate`: 剩余候选特征。
- `y_train`: 初始训练目标。
- `y_candidate`: 剩余候选真实目标，仅用于闭池模拟评估。

#### `close_pooling_test(...)`

作用：闭池 BO 模拟。每轮从已知全集中隐藏目标值，训练模型后选择候选，再把真实目标加入训练集。

常用输入：

- `n_bootstrap_sample_nums`: int，每个模型 bootstrap 训练次数。
- `n_iter`: int，最大 BO 轮数。
- `batch_size`: int，每轮选择几个样本。
- `hpar`: float，acquisition 超参数，例如 UCB 中的探索权重。
- `save_all_info`: bool，是否保存每轮模型和数据。
- `sampling_method`: str，候选生成方法，如 `genetic_algorithm`, `differential_evolution`, `monte_carlo`。
- `num_candidate`: int，候选生成数量。
- `n_samples`: int，采样算法内部 population 或样本数。
- `iterations`: int，采样算法内部迭代数。
- `candidate_sampling`: bool，True 时先用 sampler 生成/筛选候选；False 时直接在闭池候选中 acquisition。
- `diversity_method`: bool 或 str，是否启用 diversity 修正。
- `use_data_correlation`: bool，是否使用目标残差相关性。
- `use_model_correlation`: bool，是否训练模型相关性的辅助目标。

输出：无显式返回值。主要副作用：

- 更新 `performance_record.txt`。
- 写入 `model_weights/{iteration}/`。
- 达到阈值时写入 `model_weights/final_train_data.csv`。

#### `optimize(...)`

作用：开池或候选池 BO 的主入口。训练 surrogate，生成候选，选择下一批推荐样本。

常用输入：

- `batch_size`: int，最终推荐样本数。
- `n_bootstrap_sample_nums`: int，每个模型 bootstrap 数。
- `sampling_method`: str，候选生成方法。
- `num_candidate`: int，候选池大小或生成候选数。
- `n_samples`: int，采样算法内部样本数/population size。
- `iterations`: int，采样算法内部迭代数。
- `hpar`: float，acquisition 超参数。
- `if_train`: bool，是否重新训练模型。
- `candidate_sampling`: bool，是否先用 sampler 生成候选。
- `n_random_models`: int，从 bootstrap 模型集中随机抽几个模型做 ensemble 预测。
- `seperate`: bool，是否对每个模型单独生成候选。
- `diversity_method`: bool 或 str，是否加入 diversity 修正。
- `use_data_correlation`: bool，是否利用目标残差相关性。
- `use_model_correlation`: bool，是否加入模型相关性的辅助目标。

输出：

```python
samples_next, next_indexes
```

- `samples_next`: 推荐样本，通常为原始特征尺度，形状 `[batch_size, n_features]`。
- `next_indexes`: 推荐样本索引。

## `src.io`

### `class IOManager`

负责数据读取、清洗、缩放、保存/加载 scaler。

#### `__init__(root=None, method='standard', file_path=None)`

输入：

- `root`: 数据根目录，默认当前工作目录。
- `method`: `'standard'` 使用 `StandardScaler`；`'minmax'` 使用 `MinMaxScaler`。
- `file_path`: scaler 保存目录，默认 `model_weights/`。

输出：构造对象。

#### `read_data(file_name, target_props, feature_props=None, drop_columns=None, descriptor_type='magpie', handle_null=True, drop_non_numeric=True)`

作用：读取训练 CSV，清洗空值和非数值列，返回特征和目标数组。

输入：

- `file_name`: str，CSV 路径。
- `target_props`: list[str]，目标列。
- `feature_props`: list[str] 或 None，特征列。
- `drop_columns`: list[str] 或 None，先删除的列。
- `handle_null`: bool，是否自动处理空值。
- `drop_non_numeric`: bool，是否删除非数值特征；二分类非数值目标会编码成 0/1。

输出：

```python
X, y
```

- `X`: `np.ndarray`，形状 `[n_samples, n_features]`。
- `y`: `np.ndarray`，形状 `[n_samples, n_targets]`。

#### `read_candidate_data(file_name, target_props, feature_props=None, drop_columns=None, descriptor_type='magpie', drop_non_numeric=True)`

作用：读取候选池 CSV，只返回候选特征。

输入类似 `read_data()`。

输出：

- `X`: `np.ndarray`，候选池特征，形状 `[n_candidates, n_features]`。

#### `handle_null_values(data, target_props, drop_non_numeric, if_train_data=True)`

作用：处理空值和非数值列。

输入：

- `data`: `pd.DataFrame`。
- `target_props`: list[str]。
- `drop_non_numeric`: bool。
- `if_train_data`: bool，训练数据会检查目标列；候选数据不会要求目标列存在。

输出：

```python
cleaned_data, non_numeric_columns
```

#### `standardize_data(X=None, y=None, cand_X=None, cand_y=None, minmax_feature_range=(0, 1), if_train=False, data_id=None)`

作用：训练模式下拟合 scaler 并保存；预测模式下加载 scaler 并 transform。

输入：

- `X`: 训练特征。
- `y`: 训练目标。
- `cand_X`: 候选特征。
- `cand_y`: 候选目标。
- `minmax_feature_range`: MinMaxScaler 范围。
- `if_train`: True 表示 fit scaler；False 表示 load scaler。
- `data_id`: scaler 文件后缀，例如 `scaler{data_id}.pkl`。

输出：按输入顺序返回非 None 的缩放结果。例：

```python
X_scaled, y_scaled = standardize_data(X=X, y=y, if_train=True)
X_scaled, y_scaled, cand_X_scaled = standardize_data(X=X, y=y, cand_X=Xc, if_train=True)
```

#### `inverse_transform_X(X_scaled)` / `inverse_transform_y(y_scaled)`

输入：缩放后的 `X` 或 `y`。

输出：反缩放到原始尺度的数组。

#### `save_predictions(predictions, file_name)`

输入：预测值和输出 CSV 路径。

输出：无返回值；写入 CSV。

## `src.surrogate_model`

### `class SurrogateModel`

统一封装 sklearn、XGBoost、LightGBM、CatBoost、BoTorch GP、KAN/FastKAN 等模型。

#### `__init__(model_name=None, params=None)`

输入：

- `model_name`: str，模型名。常见值：`Ridge`, `Lasso`, `ElasticNet`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForest`, `SVR`, `MLPRegressor`, `ExtraTreesRegressor`, `XGBoost`, `LightGBM`, `CatBoost`, `GP_cpu`, `GP_gpu`, `KAN`, `FastKAN`。
- `params`: dict 或 None，模型初始化参数。

输出：构造对象，内部 `self.model` 是实际 estimator。

#### `fit(X, y)`

输入：

- `X`: `np.ndarray`，形状 `[n_samples, n_features]`。
- `y`: `np.ndarray`，形状 `[n_samples]` 或 `[n_samples, n_targets]`。

输出：无显式返回值；训练内部模型。

#### `predict(X)`

输入：

- `X`: `np.ndarray`，形状 `[n_samples, n_features]`。

输出：

- `pred`: `np.ndarray`，形状通常为 `[n_samples]`。

#### `predict_proba(X)`

作用：分类模型返回概率；如果底层模型没有 `predict_proba`，则把回归输出裁剪到 `[0,1]` 并构造二分类概率。

输入：`X`。

输出：`np.ndarray`，形状 `[n_samples, n_classes]`。

#### `predict_std(X)` / `predict_var(X)`

作用：返回模型预测不确定度。当前主要支持 `GP_cpu` 和 `GP_gpu`。

输入：`X`。

输出：

- `predict_std`: 标准差，形状 `[n_samples]`。
- `predict_var`: 方差，形状 `[n_samples]`。

如果模型不支持，会抛出 `AttributeError`。

### `hyperparameter_optimization(model_name, X_train, y_train, cls=False, n_trials=20, cv_n_splits=5, alpha=0.2, expect_N=3, eta=3)`

作用：用 Optuna 搜索模型超参数。

输入：

- `model_name`: str。
- `X_train`: 训练特征。
- `y_train`: 训练目标。
- `cls`: bool，是否分类任务。
- `n_trials`: int，Optuna trial 数。
- `cv_n_splits`: int，交叉验证折数。
- `alpha`: float，泛化 gap 惩罚权重。
- `expect_N`, `eta`: 资源预算相关参数。

输出：

- `best_params`: dict，最优超参数。`GP_gpu` 直接返回 `{}`。

### `estimate_budget(X_ori, y_ori, base=32, max_cap=1000, lambda_unsup=1.0, growth='loglinear')`

作用：根据无监督维度、监督维度和 intrinsic dimension 估计超参搜索资源预算。

输出：

```python
budget, d_eff, d_all
```

## `src.evaluation`

### `class ModelEvaluator`

负责 bootstrap 训练、OOB 评估、模型保存、stacking 和残差相关性。

#### `__init__(X_train, y_train, file_path=None, bs_sample_number=None, optimization_goal='maximize')`

输入：

- `X_train`: 缩放后的训练特征。
- `y_train`: 缩放后的训练目标。
- `file_path`: 模型保存目录。
- `bs_sample_number`: 每个 bootstrap 样本大小；None 时自动决定。
- `optimization_goal`: 目标方向。

输出：构造对象。会初始化 `ClusterBootstrapSampler` 并计算 `global_labels`。

#### `evaluate(model_names, num_target, n_bootstrap_sample_nums, cls=False, use_full_eval=False, cross_val=False, uni_hyper=False)`

作用：对一个目标列训练多个模型的 bootstrap ensemble，并保存每个模型结果。

输入：

- `model_names`: list[str]。
- `num_target`: int，目标列索引。
- `n_bootstrap_sample_nums`: int，bootstrap 次数。
- `cls`: bool，是否分类。
- `use_full_eval`: bool，是否用全训练集评估；False 时用 OOB。
- `cross_val`: bool，是否使用 KFold 方式训练/评估。
- `uni_hyper`: bool，是否先用统一采样权重优化一次超参。

输出：

```python
model_results: dict
```

结构大致为：

```python
{
  model_name: {
    'models': [...],
    'errors': [...],
    'residuals': [...],
    'elpd': [...],
    'oob_sizes': [...],
    'elpd_scores': np.ndarray,
    'elpd_per_point_mean': float,
    'crps': [...],
    'oob_var_mean': float,
    'oob_mad_var_mean': float,
  }
}
```

#### `bootstrap_evaluation(...)`

作用：对单个模型、单个目标做 bootstrap ensemble 训练。

输出：list，包含：

```python
[models, errors, residuals, elpds, oob_sizes, elpd_scores,
 elpd_per_point_mean, crps, oob_var_list, oob_mad_var_list,
 oob_n_list, oob_var_mean, oob_mad_var_mean]
```

#### `_train_model(...)`

Ray remote 函数。训练一个 bootstrap 模型并计算 OOB 指标。

输入：模型名、超参、训练数组引用、bootstrap 索引、任务类型等。

输出：dict，包含：

- `model`: 训练好的 `SurrogateModel`。
- `error`: 回归为 R2 参考值，分类为 accuracy。
- `residual`: 全训练集残差。
- `elpd`: OOB expected log predictive density。
- `crps`: OOB CRPS 或分类 Brier score。
- `oob_size`: OOB 样本数。
- `oob_var`, `oob_mad_var`: OOB 残差方差估计。

#### `save_models(...)` / `load_models(file_name)`

作用：保存/读取模型结果 pickle。

输出：

- `save_models`: 无返回值。
- `load_models`: dict，保存时的 payload。

#### `train_stacking_model(...)`

作用：把多个模型 ensemble 包装成 sklearn `StackingRegressor` 或 `StackingClassifier`。

输出：

```python
stacking_model, base_model_errors, residuals
```

#### `evaluate_with_stacking(...)`

作用：先调用 `evaluate()`，再训练 stacking，并保存 `stacking_models_{num_target}.pkl`。

输出：`model_results`，额外包含 `stacking_models`。

#### `calculate_and_save_residual_correlation(model_names, stacking=False)`

作用：读取各模型/目标保存的 residual，计算目标残差相关性矩阵并保存。

输出：残差相关性字典；同时有文件输出副作用。

### `class ClusterBootstrapSampler`

负责按聚类结果给 bootstrap 抽样赋权，使数据稀疏区、噪声点或目标高价值区得到不同采样概率。

重要函数：

- `compute_bootstrap_probabilities_clustering(X, enable_refine=False)`
  - 输入：特征矩阵 `X`。
  - 输出：`labels`，每个样本的聚类标签。
- `compute_bootstrap_probabilities_prob(labels, y)`
  - 输入：聚类标签和目标值。
  - 输出：每个样本的 bootstrap 采样概率，形状 `[n_samples]`。

### `class AdaptiveClusterer`

作用：自动比较 HDBSCAN/GMM/KMeans 等聚类配置，并返回最合适的聚类标签。主要供 bootstrap sampler 使用。

核心函数：

- `find_best_clustering(X)`
  - 输入：特征矩阵。
  - 输出：`labels` 或包含模型信息的聚类结果。

## `src.sampling`

### `class Sampler`

负责生成候选点。输入/输出都通常在缩放后的特征空间中。

#### `__init__(scaler, feature_bounds=None)`

输入：

- `scaler`: `StandardScaler` 或 `MinMaxScaler`，用于决定默认搜索范围。
- `feature_bounds`: `np.ndarray` 或 None，形状 `[2, n_features]`，缩放后的下/上界。

输出：构造对象。

#### 基础采样函数

- `sobol_sampling(feature_dim, num_candidate=50)`
  - 输入：特征维度和候选数。
  - 输出：Sobol 候选点，形状 `[num_candidate, feature_dim]`。
- `gaussian_sampling(feature_dim, mean=None, std_dev=None, num_candidate=10)`
  - 输出：高斯随机候选点。
- `bernoulli_sampling(feature_dim, p=0.5, n_test=1, num_candidate=10)`
  - 输出：二项分布候选点。

#### 优化式采样函数

这些函数都接受 `model`，要求 `model.predict(X)` 返回候选点得分；内部把最大化转成 `-model.predict` 以适配部分最小化优化器。

- `monte_carlo_sampling(model, feature_dim, n_samples=100, iterations=20, perturbation_scale=0.3, Lambda=1.5, num_candidate=100, candidate_list=None)`
- `genetic_algorithm_sampling(model, feature_dim, population_size=1000, generations=20, num_candidate=10, candidate_list=None)`
- `particle_swarm_sampling(model, feature_dim, population_size=50, iterations=20, num_candidate=10, candidate_list=None)`
- `simulated_annealing_sampling(model, feature_dim, iterations=200, num_candidate=10, candidate_list=None)`
- `differential_evolution_sampling(model, feature_dim, population_size=50, generations=300, num_candidate=10, candidate_list=None)`
- `artificial_fish_swarm_sampling(model, feature_dim, population_size=50, iterations=20, num_candidate=10, candidate_list=None)`

输入共同点：

- `model`: 有 `predict()` 的 surrogate 或 wrapper。
- `feature_dim`: int。
- `num_candidate`: int，返回候选数。
- `candidate_list`: `np.ndarray` 或 None；若提供，则连续采样点会映射回最近的真实候选池样本。

输出：

- 通常为 `np.ndarray`，形状 `[num_candidate, feature_dim]`。

#### `generate_candidates(method, model, feature_dim, num_candidate=100, n_samples=100, iterations=500, candidate_list=None)`

作用：串行候选生成统一入口。

输入：

- `method`: `gaussian`, `bernoulli`, `monte_carlo`, `genetic_algorithm`, `particle_swarm`, `differential_evolution`, `artificial_fish_swarm`, `simulated_annealing`, `ant_colony`, `immune_algorithm`。
- 其他参数传给对应采样函数。

输出：候选矩阵 `[num_candidate, feature_dim]`。

#### `generate_candidates_parallel(...)`

作用：并行候选生成。可按模型分别生成候选，再合并 Sobol 点以增加覆盖。

输入：

- `method`: 采样方法。
- `feature_dim`: 特征维度。
- `model_results`: `ModelEvaluator.evaluate()` 输出。
- `model_list`: 模型名列表。
- `num_of_targets`: 目标数量。
- `model_path`: 模型保存路径。
- `num_candidate`, `n_samples`, `iterations`: 采样规模。
- `candidate_list`: 候选池或 None。
- `n_random_models`: 每个模型随机抽取多少 bootstrap 模型。
- `Seperate`: 是否按模型分别生成。
- `rand_all`: 是否额外用所有模型混合生成一批候选。
- `select_region`: 目标窗口。

输出：

- `candidate_X_scaled`: 缩放空间中的候选矩阵。

### `RandomizedAbstractSurrogateModel`

作用：从已训练的 bootstrap 模型集中随机抽模型，形成一个可被 sampler 调用的 ensemble surrogate。

输入：模型列表、`model_results`、目标数、随机模型数等。

输出方法：

- `predict(X)`: 返回综合目标分数，供采样器排序。

### `map_to_candidate_list*`

作用：把连续空间采样点映射到候选池中最近的真实样本，并避免重复索引。

输入：

- `samples`: 连续采样点。
- `candidate_list`: 真实候选池。
- `used_indices`: 已选索引集合。
- `metric`: 距离度量。

输出：

```python
mapped_samples, selected_indices
```

## `src.acquisition_function`

### `class AcquisitionFunction`

负责把模型预测转成 acquisition 分数，并选择下一批样本。

#### `__init__(hpar=0.1)`

输入：

- `hpar`: acquisition 超参数。对 UCB 来说相当于探索权重。

输出：构造对象。

#### `ucb(mean, std, hpara=None)`

公式：

```text
UCB = mean + hpar * std
```

输入：

- `mean`: 预测均值。
- `std`: 预测标准差。
- `hpara`: 可选，覆盖对象中的 `hpar`。

输出：UCB 分数，形状与 `mean` 一致。

#### `ei(mean, std, y_best, hpara=None)`

作用：Expected Improvement。

输入：预测均值、标准差、当前最优值。

输出：EI 分数。

#### `pi(mean, std, y_best, hpara=None)`

作用：Probability of Improvement。

输入：预测均值、标准差、当前最优值。

输出：PI 分数。

#### `hypervolume(points, reference_point=None, batch_size=20000)`

作用：计算 Pareto front 和 hypervolume contribution，支持多目标选择。

输入：

- `points`: 候选点的目标空间坐标。
- `reference_point`: hypervolume 参考点。
- `batch_size`: 分批处理大小。

输出：与候选点相关的 hypervolume 分数/贡献。

#### `select_next(method, X_candidate, model_name_list, num_of_targets, model_path, batch_size=10, X_train=None, y_value=None, model_result=None, stack=False, select_region=None, diversity_method=False, alpha=0.5, optimization_goal='maximize', use_correlation=False, use_model_correlation=True, train_clsuter_labels=None, data_level_control=False, two_step=False)`

作用：BO 的核心选择函数。它读取模型 ensemble，预测候选点的均值/方差，计算 acquisition、diversity、相关性修正等，最终返回最值得评估的候选索引。

输入：

- `method`: acquisition 方法，例如 `'ucb'`。
- `X_candidate`: 候选特征，通常是缩放后的数组 `[n_candidates, n_features]`。
- `model_name_list`: 模型名列表。
- `num_of_targets`: 目标数量。
- `model_path`: 模型 pickle 目录。
- `batch_size`: 返回多少个候选。
- `X_train`, `y_value`: 当前训练数据，用于 diversity/相关性/最佳值等计算。
- `model_result`: `ModelEvaluator.evaluate()` 的结果，可避免重复读文件。
- `stack`: 是否使用 stacking 模型。
- `select_region`: 目标窗口，缩放后的 `[2, n_targets]`。
- `diversity_method`: 是否启用多样性修正。
- `alpha`: diversity/acquisition 混合权重。
- `optimization_goal`: `'maximize'` 或 `'minimize'`。
- `use_correlation`: 是否使用数据残差相关性。
- `use_model_correlation`: 是否使用模型相关辅助目标。
- `train_clsuter_labels`: 训练集聚类标签。
- `data_level_control`, `two_step`: 实验性控制项。

输出：

- `next_indexes`: 被选中的候选索引，长度为 `batch_size`。

#### `MF_predres(...)`, `BOfusion_select_next(...)`, `MT_select_next(...)`

这些函数服务于多保真、多任务或 fusion BO 原型。

输出一般是候选预测结果或被选中的候选索引。

### acquisition helper 函数

- `compute_hv_contributions(pareto_points, reference_point)`
  - 输入：Pareto 点和参考点。
  - 输出：每个 Pareto 点的 hypervolume contribution。
- `estimate_density_and_spread(...)`
  - 输入：候选点、训练点、可选目标值。
  - 输出：密度和 spread 相关诊断，用于 diversity。
- `safe_cv(std, mean, rel_floor=0.05, abs_floor=1e-12)`
  - 输入：标准差和均值。
  - 输出：稳定的 coefficient of variation。
- `mix_ratio_from_scores(scores, method='gs', ...)`
  - 输入：模型分数张量。
  - 输出：基于 Gini-Simpson 或 entropy 的 disagreement/mix ratio。
- `compute_structure_with_rigidity(...)`
  - 输入：候选上的预测均值、降维坐标、两条路径索引等。
  - 输出：结构简单性/刚性分数及 diagnostics。

## `src.multi_task`

### `class MultiTaskBayesianOptimization`

多任务 BO 原型。主任务和相关任务分别抽初始样本，训练相关模型后用 `AcquisitionFunction.MT_select_next()` 做选择。

主要函数：

- `__init__(data_file, Main_props, correlated_props, ...)`
  - 输入：数据文件、主目标列、相关目标列、模型列表、初始样本数等。
  - 输出：构造对象。
- `close_pooling_test(...)`
  - 输入：bootstrap 数、迭代数、batch size、hpar。
  - 输出：运行闭池多任务模拟，主要通过文件和日志输出。

## `src.multi_fidelity`

### `class MultiFidelityBayesianOptimization`

多保真 BO 原型。高保真和低保真目标分别建模，并考虑成本。

主要函数：

- `__init__(data_file, HF_props, LF_props, ..., HFcost=10, LFcost=[1], ...)`
  - 输入：高保真目标、低保真目标、成本、初始样本等。
  - 输出：构造对象。
- `close_pooling_test(...)`
  - 输入：bootstrap 数、迭代数、batch size、hpar。
  - 输出：运行闭池多保真模拟。

## 如何扩展

### 添加新的 surrogate 模型

1. 在 `src/surrogate_model.py` 的 `SurrogateModel._initialize_model()` 中加入模型名到类/构造器的映射。
2. 如果需要超参搜索，在 `hyperparameter_optimization()` 中为该 `model_name` 增加搜索空间。
3. 确保模型至少有 `fit(X, y)` 和 `predict(X)`。
4. 如果模型能给不确定度，增加 `predict_std()` 支持。

### 添加新的采样方法

1. 在 `Sampler` 中新增 `xxx_sampling()`。
2. 在 `generate_candidates()` 和 `generate_candidates_ray()` 中加入 `method == 'xxx'` 分支。
3. 返回值必须是 `[num_candidate, feature_dim]` 的 numpy 数组。
4. 如果支持闭池，处理 `candidate_list` 映射。

### 添加新的 acquisition 方法

1. 在 `AcquisitionFunction` 中新增 scoring 函数。
2. 在 `select_next()` 中加入 method 分支或组合逻辑。
3. 明确该方法需要哪些输入：均值、标准差、当前最优值、模型间方差、目标相关性等。
4. 输出必须能被排序为候选分数，并最终返回候选索引。

## 常见问题

### 为什么 `math_test.py` 很慢？

默认配置是 20 维 Ackley benchmark，`test_iteration=100`，每轮 `iterations=500`，还会训练多种模型并用 Ray 并行。单轮可能需要数分钟到数十分钟。

### 为什么会反复启动 Ray？

`BayesianOptimization` 初始化会调用 `initialize_ray()`。某些流程会在迭代或重新构造对象时重新初始化 Ray。

### 哪些模型最稳？

当前示例里较常用的一组是：

```python
['Lasso', 'Ridge', 'ElasticNet', 'MLPRegressor', 'LightGBM', 'XGBoost', 'KNeighborsRegressor', 'DecisionTreeRegressor']
```

如果数据很少，线性模型、KNN、树模型通常更快；MLP、Boosting 和 GP 类模型更重。

### 什么时候使用 `candidate_sampling=True`？

当候选空间很大或是开池连续空间时，可以先用 sampler 生成较小候选集合，再由 acquisition 精选 batch。闭池候选本身不大时，可以设为 False，直接对候选池打分。

### `select_region` 是什么？

`select_region` 用于目标窗口任务。它不是简单最大化某个目标，而是希望目标落在指定区间附近。例如：

```python
select_region = {
    'target_gap': [1.45, 1.55],
    'target_magnetization': [19.5, 20.5],
}
```

代码会把目标值转换为到窗口中心的负距离，越接近窗口中心得分越高。
