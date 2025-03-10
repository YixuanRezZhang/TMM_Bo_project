from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
import optuna
from optuna.pruners import MedianPruner
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from botorch.models.transforms import Standardize
from fastkan import FastKAN as FastKAN
from kan import KAN
from kan.utils import create_dataset_from_data

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ## CUDA not applicatble yet
device = torch.device('cpu')

import warnings
import logging

# 关闭警告信息
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

class RegressionToClassificationWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return (self.regressor.predict(X) > 0.5).astype(int)

    def predict_proba(self, X):
        preds = self.regressor.predict(X)
        return np.vstack((1 - preds, preds)).T

def filter_gp_params(params):
    allowed_keys = {"alpha", "optimizer", "n_restarts_optimizer", "copy_X_train", "random_state"}
    return {k: v for k, v in params.items() if k in allowed_keys}

class SurrogateModel:
    def __init__(self, model_name=None, params=None):
        self.model_name = model_name
        self.params = params if params else {}
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

        hidden_layers = []
        for key in sorted(self.params.keys()):  # 确保层的顺序
            if key.startswith('neurons_layer_'):
                hidden_layers.append(self.params[key])

        if not hidden_layers:
            raise ValueError("MLPRegressor 需要至少一个 hidden_layer_sizes 参数")

        # 过滤掉 `neurons_layer_*`，确保 `MLPRegressor` 只接收有效参数
        clean_params = {k: v for k, v in self.params.items() if not k.startswith('neurons_layer_')}

        # 传递 `hidden_layer_sizes`
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
            print(f"KAN params: {self.params}")
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
        preds = self.model.predict(X)
        return np.vstack((1 - preds, preds)).T

def hyperparameter_optimization(
    model_name, 
    X_train, 
    y_train, 
    cls=False, 
    n_trials=20, 
    max_sample_for_tuning=None,  # 用于平滑动态区间
    base_value=200,
    cv_n_splits=5,              # KFold 折数
    alpha=0.2                   # 偏置惩罚系数
):

    n_samples, feature_dim = X_train.shape

    if max_sample_for_tuning is None:
        n_components = min(n_samples, feature_dim)
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        explained_variance = pca.explained_variance_ratio_
        valid = explained_variance > 1e-9
        if not np.any(valid):
            return 0
        p_i = explained_variance[valid]
        p_i = p_i / np.sum(p_i)
        r_eff = np.exp(-np.sum(p_i * np.log(p_i)))
        max_sample_for_tuning = int(base_value * r_eff)
        print(f'estimate_max_sample: {max_sample_for_tuning}')

    target_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    ratio = min(n_samples / max_sample_for_tuning, 1.0)

    def objective(trial):
        params = {}

        if model_name == 'RandomForest':
            n_estimators_upper = int(50 + (1000 - 50) * ratio)
            max_depth_upper = int(1 + (64 - 1) * ratio)
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, n_estimators_upper)
            params['max_depth'] = trial.suggest_int('max_depth', 1, max_depth_upper)

        elif model_name in ['XGBoost', 'LightGBM']:
            n_estimators_upper = int(10 + (1000 - 10) * ratio)
            max_depth_upper = int(1 + (64 - 1) * ratio)
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, n_estimators_upper)
            params['max_depth'] = trial.suggest_int('max_depth', 1, max_depth_upper)
            lr_lower, lr_upper_base = 1e-4, 1.0
            lr_upper = lr_lower + (lr_upper_base - lr_lower) * ratio
            params['learning_rate'] = trial.suggest_loguniform('learning_rate', lr_lower, lr_upper)
            if model_name == 'LightGBM':
                min_child_samples_upper = int(20 + (200 - 20) * ratio)
                params['min_child_samples'] = trial.suggest_int('min_child_samples', 5, min_child_samples_upper)

        elif model_name == 'CatBoost':
            # 动态调整学习率和迭代次数
            lr_lower, lr_upper = 1e-4, 0.5  # 归一化数据时，学习率较小
            iterations_lower, iterations_upper = int(100 + 400 * ratio), int(500 + 1000 * ratio)
            
            params['iterations'] = trial.suggest_int('iterations', iterations_lower, iterations_upper)
            params['learning_rate'] = trial.suggest_loguniform('learning_rate', lr_lower, lr_upper)
            params['depth'] = trial.suggest_int('depth', 4, min(12, feature_dim))  # 深度不超过特征维度
            params['l2_leaf_reg'] = trial.suggest_loguniform('l2_leaf_reg', 1e-2, 10.0)  # L2 正则项
            params['bagging_temperature'] = trial.suggest_uniform('bagging_temperature', 0.0, 1.0)  # 采样温度
            params['random_strength'] = trial.suggest_uniform('random_strength', 0.0, 1.0)  # 随机性控制
            params['boosting_type'] = trial.suggest_categorical('boosting_type', ['Ordered', 'Plain'])  # CatBoost 两种 boosting 方式

        elif model_name == 'SVR':
            # 当数据较少时，使用较窄的范围；数据充足时允许更大范围
            C_lower, C_upper = 1e-2, 1e3
            epsilon_lower, epsilon_upper = 1e-3, 1e1
            effective_C_upper = C_lower + (C_upper - C_lower) * ratio
            effective_epsilon_upper = epsilon_lower + (epsilon_upper - epsilon_lower) * ratio
            params['C'] = trial.suggest_loguniform('C', C_lower, effective_C_upper)
            params['epsilon'] = trial.suggest_loguniform('epsilon', epsilon_lower, effective_epsilon_upper)
    
        elif model_name == 'MLPRegressor':
 
            # 动态计算层数：当 ratio=0 时采用1层，当 ratio=1 时采用3层，线性插值得到中间值
            n_layers = int(round(1 + 3 * ratio))
            f = 1 + feature_dim / 50.0
            
            # 动态计算每层神经元数范围：
            neurons_lower = int(round((50 + (100 - 50) * ratio) * f))
            neurons_upper = int(round((100 + (300 - 100) * ratio) * f))
            
            # hidden_layers = tuple(trial.suggest_int(f'neurons_layer_{i}', neurons_lower, neurons_upper) for i in range(n_layers))
            # params['hidden_layer_sizes'] = hidden_layers
            for i in range(n_layers):
                params[f'neurons_layer_{i}'] = trial.suggest_int(f'neurons_layer_{i}', neurons_lower, neurons_upper)

            params['activation'] = trial.suggest_categorical('activation', ['tanh', 'relu', 'logistic'])
            params['solver'] = trial.suggest_categorical('solver', ['adam', 'sgd'])
            
            # 对正则化参数 alpha 同样设定动态范围（此处可保持固定范围或做简单插值）
            alpha_lower, alpha_upper = 1e-5, 1e0
            params['alpha'] = trial.suggest_loguniform('alpha', alpha_lower, alpha_upper)
            params['early_stopping'] = trial.suggest_categorical('early_stopping', [True])
            # params['validation_fraction'] = trial.suggest_uniform('validation_fraction', 0.1, 0.3)
            # params['n_iter_no_change'] = trial.suggest_int('n_iter_no_change', 5, 20)

        elif model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e3)
            if model_name == 'ElasticNet':
                params['l1_ratio'] = trial.suggest_uniform('l1_ratio', 0, 1)

        elif model_name == 'KNeighborsRegressor':
            n_neighbors_upper = min(n_samples // 2, 100)
            params['n_neighbors'] = trial.suggest_int('n_neighbors', 1, n_neighbors_upper)

        elif model_name == 'DecisionTreeRegressor':
            max_depth_upper = int(1 + (64 - 1) * ratio)
            params['max_depth'] = trial.suggest_int('max_depth', 1, max_depth_upper)

        elif model_name in ['GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor']:
            n_estimators_upper = int(10 + (1000 - 10) * ratio)
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, n_estimators_upper)
            if model_name in ['GradientBoostingRegressor', 'AdaBoostRegressor']:
                lr_lower, lr_upper_base = 1e-4, 1.0
                lr_upper = lr_lower + (lr_upper_base - lr_lower) * ratio
                params['learning_rate'] = trial.suggest_loguniform('learning_rate', lr_lower, lr_upper)
            if model_name in ['GradientBoostingRegressor', 'ExtraTreesRegressor']:
                max_depth_upper = int(1 + (64 - 1) * ratio)
                params['max_depth'] = trial.suggest_int('max_depth', 1, max_depth_upper)

        elif model_name == 'KRR':
            params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e3)
            params['gamma'] = trial.suggest_loguniform('gamma', 1e-5, 1e1)

        # elif model_name == 'GP_cpu':
        #     # 根据样本数调整上界，同时引入特征维度影响
        #     if n_samples < 500:
        #         ls_upper = 10.0
        #         gp_alpha_upper = 1e-1
        #     else:
        #         # 例如让上界不超过 feature_dim * 10
        #         ls_upper = min(100.0, feature_dim * 10)
        #         # 在高维下可适当降低 alpha 的上界（加强正则化）
        #         gp_alpha_upper = 1.0 / max(1, feature_dim * 0.1)
        #     # 常数项
        #     constant_val = trial.suggest_loguniform('constant_value', 1e-2, 10)
        #     # 使用 ARD：为每个特征采样一个 length_scale 参数
        #     length_scales = []
        #     for i in range(feature_dim):
        #         ls = trial.suggest_loguniform(f'length_scale_{i}', 1e-2, ls_upper)
        #         length_scales.append(ls)
        #     length_scales = np.array(length_scales)
        #     # 噪声水平
        #     noise_level = trial.suggest_loguniform('noise_level', 1e-2, 3)
        #     # 正则化参数 alpha
        #     alpha_val = trial.suggest_loguniform('alpha', 1e-6, gp_alpha_upper)
            
        #     from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        #     # 构建核函数，其中 Matern 核采用 ARD 参数（length_scale 为向量）
        #     kernel = ConstantKernel(constant_value=constant_val, constant_value_bounds=(1e-3, 1e2)) * \
        #              Matern(length_scale=length_scales, nu=2.5) + \
        #              WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6, 1e-1))
        #     params['kernel'] = kernel
        #     params['alpha'] = alpha_val

        elif model_name == 'GP_cpu':
            # 计算动态因子
            sample_ratio = ratio
            dim_ratio = np.sqrt(feature_dim + 1)

            # **Length Scale 动态范围**
            length_scale_lower = max(0.1, feature_dim / (n_samples + 10))  # 防止过拟合
            length_scale_upper = min(10 * feature_dim, 50)  # 避免过度平滑
            length_scales = np.array([trial.suggest_loguniform(f'length_scale_{i}', length_scale_lower, length_scale_upper) for i in range(feature_dim)])

            # **Noise Level 动态范围**
            noise_level_lower = 1e-3  # 确保不会完全忽略噪声
            noise_level_upper = 0.5 * (1 - sample_ratio) + 1e-2  # 样本多时减少噪声
            noise_level = trial.suggest_loguniform('noise_level', noise_level_lower, noise_level_upper)
                
            # **Constant Kernel 动态范围**
            constant_value_lower = 0.1
            constant_value_upper = 10 / dim_ratio
            constant_val = trial.suggest_loguniform('constant_value', constant_value_lower, constant_value_upper)
        
            # **Alpha (正则化) 动态范围**
            alpha_lower = 1e-6
            alpha_upper = min(1.0 / feature_dim, 1e-1)
            alpha_val = trial.suggest_loguniform('alpha', alpha_lower, alpha_upper)
        
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
            params['lamb'] = trial.suggest_loguniform('lamb', 1e-4, 5e-2)

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
            params['lr'] = trial.suggest_loguniform('lr', 1e-4, 1e-1)
        else:
            params = {}

        model = SurrogateModel(model_name, params)

        # ----------- 使用KFold 做交叉验证 -----------
        kf = KFold(n_splits=cv_n_splits, shuffle=True, random_state=42)
        cv_scores = []
        cv_gaps = []  # 用于记录(验证误差 - 训练误差), 评估过拟合

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)

            pred_val = model.predict(X_val)
            pred_tr = model.predict(X_tr)

            if cls:
                fold_val_error = 1.0 - accuracy_score(y_val, pred_val)
                fold_tr_error = 1.0 - accuracy_score(y_tr, pred_tr)
            else:
                fold_val_error = mean_squared_error(y_val, pred_val)
                fold_tr_error = mean_squared_error(y_tr, pred_tr)

            cv_scores.append(fold_val_error)

            # 计算过拟合差距(验证 - 训练), 如果>0 说明验证集更差
            gap = fold_val_error - fold_tr_error
            cv_gaps.append(gap)

        mean_val_error = np.mean(cv_scores)
        mean_gap = np.mean(cv_gaps)  # 越大说明越过拟合

        # 偏置惩罚: 如果 mean_gap>0 (验证损失>训练损失), 则视情况加惩罚
        # alpha越大, 惩罚越强
        penalty = alpha * max(0.0, mean_gap)

        final_score = mean_val_error + penalty

        return final_score

    ### for all kind of Gaussian like model, no need to do hyperparameter_optimization since cost can be high
    if model_name == 'GP_gpu':
        return {}

    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    study = optuna.create_study(direction='minimize', pruner=pruner, sampler=optuna.samplers.TPESampler(n_startup_trials=3, multivariate=True))
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    return study.best_params
