import numpy as np
import os, pickle
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from src.surrogate_model import SurrogateModel, hyperparameter_optimization

class ModelEvaluator:
    def __init__(self, X_train, y_train, file_path=None):
        self.X_train = X_train
        self.y_train = y_train
        self.file_path = file_path if file_path is not None else f'{os.getcwd()}/model_weights'

    def save_models(self, model_name, optimized_params, models, model_errors, file_name):
        if not os.path.exists(f'{self.file_path}'):
            os.mkdir(f'{self.file_path}')
        with open(f'{self.file_path}/{file_name}', 'wb') as f:
            pickle.dump({'model_name': model_name, 'optimized_params': optimized_params, 'models': models, 'model_errors':model_errors}, f)

    def load_models(self, file_name):
        with open(f'{self.file_path}/{file_name}', 'rb') as f:
            data = pickle.load(f)
        
        # model_name = data['model_name']
        # optimized_params = data['optimized_params']
        # models = data['models']
        # model_errors = data['model_errors']

        return data

    def bootstrap_evaluation(self, model_name, optimized_params, num_target, n_bootstrap_sample_nums=20, cls=False, use_full_eval=False, cross_val=False, cv_n_splits=5):
        n_samples = len(self.X_train)
        errors = []
        models = []
        X_bs = self.X_train
        y_bs = self.y_train[:,num_target]
        if n_bootstrap_sample_nums < 2:
            n_bootstrap_sample_nums = 2
        cv_n_splits = min(n_bootstrap_sample_nums, cv_n_splits)

        if cross_val:
            kf = KFold(n_splits=cv_n_splits)
            for train_idx, val_idx in kf.split(X_bs):
                X_train_cv, X_val_cv = X_bs[train_idx], X_bs[val_idx]
                y_train_cv, y_val_cv = y_bs[train_idx], y_bs[val_idx]

                model = SurrogateModel(model_name, optimized_params)
                model.fit(X_train_cv, y_train_cv)
                preds = model.predict(X_val_cv)

                if cls:
                    error = accuracy_score(y_val_cv, preds)
                else:
                    r2_error = np.clip(r2_score(y_val_cv, preds), 0, 1)
                    error = r2_error

                errors.append(error)
                models.append(model)
                
        else:
            if model_name == 'GaussianProcess':
                X_tr, X_te, y_tr, y_te = train_test_split(X_bs, y_bs, test_size=0.8)
                model = SurrogateModel(model_name, optimized_params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                
                if cls:
                    error = accuracy_score(y_te, preds)
                else:
                    r2_error = np.clip(r2_score(y_te, preds), 0, 1)
                    error = r2_error

                errors.append(error)
                models.append(model)
                
            else:
                for i in range(n_bootstrap_sample_nums):
                    bootstrap_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
                    X_sample = X_bs[bootstrap_indices]
                    y_sample = y_bs[bootstrap_indices]
        
                    model = SurrogateModel(model_name, optimized_params)
                    model.fit(X_sample, y_sample)
        
                    if use_full_eval:
                        X_eval = X_bs
                        y_eval = y_bs
                    else:
                        eval_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
                        X_eval = X_bs[eval_indices] if len(eval_indices) != 0 else X_bs
                        y_eval = y_bs[eval_indices] if len(eval_indices) != 0 else y_bs
        
                    preds = model.predict(X_eval)
        
                    if cls:
                        error = accuracy_score(y_eval, preds)
                    else:
                        r2_error = np.clip(r2_score(y_eval, preds), 0, 1)
                        error = r2_error
        
                    errors.append(error)
                    models.append(model)

        # 保存每个模型名称的所有 bootstrap 结果
        self.save_models(model_name, optimized_params, models, errors, f"{model_name}_{num_target}_bootstrap.pkl")

        return [models, errors]

    def evaluate(self, model_names, num_target, n_bootstrap_sample_nums, cls=False, use_full_eval=False, cross_val = False):
        model_results = {}

        for model_name in model_names:
            
            if model_name == 'GaussianProcess' and len(self.X_train) <= 500:
                cross_val = True
            elif model_name == 'GaussianProcess' and len(self.X_train) > 500:
                cross_val = False
            
            # 优化超参数
            optimized_params = hyperparameter_optimization(model_name, self.X_train, self.y_train[:,num_target], cls=cls)
            # 评估模型
            models, errors = self.bootstrap_evaluation(model_name, optimized_params, num_target, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=cls, use_full_eval=use_full_eval, cross_val=cross_val)
            model_results[model_name] = {'model': models, 'error': errors}

        return model_results

