from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from botorch.models.transforms import Standardize

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
            'MLPRegressor': MLPRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'AdaBoostRegressor': AdaBoostRegressor,
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'XGBoost': xgb.XGBRegressor,
            'LightGBM': lambda **kwargs: lgb.LGBMRegressor(verbose=-1, **kwargs),
            'GaussianProcess': self._initialize_gp_model
        }
        if self.model_name is None or self.model_name not in models.keys():
            raise ValueError(f"Unknown model name: {self.model_name}")
        model_class = models.get(self.model_name)
        return model_class(**self.params)

    def _initialize_gp_model(self):
        return None  # Placeholder for the Gaussian process model

    def fit(self, X, y):
        ### MaternKernel as an example, people can add more in the future follow this way
        if self.model_name == 'GaussianProcess':
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[1], lengthscale_constraint=Interval(0.1, 4.0)))
            outcome_transform = Standardize(m=1)
            self.model = SingleTaskGP(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(-1), outcome_transform=outcome_transform, covar_module=covar_module)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)
        else:
            self.model.fit(X, y)

    def predict(self, X):
        ### MaternKernel as an example, people can add more in the future follow this way
        if self.model_name == 'GaussianProcess':
            model_pred = self.model.posterior(torch.tensor(X, dtype=torch.float32))
            mean = model_pred.mean.detach().cpu().numpy().reshape(-1)
            # std = torch.sqrt(model_pred.variance).detach().cpu().numpy().reshape(-1)
            return mean
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        preds = self.model.predict(X)
        return np.vstack((1 - preds, preds)).T

def hyperparameter_optimization(model_name, X_train, y_train, cls=False, n_trials=20):    
    def objective(trial):
        params = {}
        if model_name == 'RandomForest':
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 100)
            params['max_depth'] = trial.suggest_int('max_depth', 1, 32)
        elif model_name == 'SVR':
            params['C'] = trial.suggest_loguniform('C', 1e-3, 1e3)
            params['epsilon'] = trial.suggest_loguniform('epsilon', 1e-3, 1e1)
        elif model_name == 'MLPRegressor':
            params['hidden_layer_sizes'] = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50,50)])
            params['activation'] = trial.suggest_categorical('activation', ['tanh', 'relu'])
            params['solver'] = trial.suggest_categorical('solver', ['adam', 'sgd'])
            params['alpha'] = trial.suggest_loguniform('alpha', 1e-4, 1e-1)
        elif model_name in ['XGBoost', 'LightGBM']:
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 100)
            params['max_depth'] = trial.suggest_int('max_depth', 1, 32)
            params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-3, 1.0)
            if model_name == 'LightGBM':
                params['min_child_samples'] = trial.suggest_int('min_child_samples', 5, 100)
        elif model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            params['alpha'] = trial.suggest_loguniform('alpha', 1e-3, 1e2)
            if model_name == 'ElasticNet':
                params['l1_ratio'] = trial.suggest_uniform('l1_ratio', 0, 1)
        elif model_name == 'KNeighborsRegressor':
            params['n_neighbors'] = trial.suggest_int('n_neighbors', 1, min(len(X_train), 20))
        elif model_name == 'DecisionTreeRegressor':
            params['max_depth'] = trial.suggest_int('max_depth', 1, 32)
        elif model_name in ['GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor']:
            params['n_estimators'] = trial.suggest_int('n_estimators', 10, 100)
            if model_name == 'GradientBoostingRegressor' or model_name == 'AdaBoostRegressor':
                params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-3, 1.0)
            if model_name == 'GradientBoostingRegressor' or model_name == 'ExtraTreesRegressor':
                params['max_depth'] = trial.suggest_int('max_depth', 1, 32)

        model = SurrogateModel(model_name, params)
        
        if cls:
            base_model = model
            model = RegressionToClassificationWrapper(base_model)
            
        model.fit(X_train, y_train)
        preds = model.predict(X_train)

        if cls:
            return accuracy_score(y_train, preds)
        else:
            return mean_squared_error(y_train, preds)

    ### for all kind of Gaussian like model, no need to do hyperparameter_optimization since cost can be high
    if model_name == 'GaussianProcess':
        return {}

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params
