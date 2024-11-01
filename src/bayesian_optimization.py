import os, ray, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.io import IOManager
from src.surrogate_model import SurrogateModel, hyperparameter_optimization
from src.evaluation import ModelEvaluator
from src.acquisition_function import AcquisitionFunction
from src.sampling import Sampler
from src.fast_fit.bwo import BWO
from src.fast_fit.turbo import TurboM
from src.fast_fit.turbo import TurboM_bwo

import torch
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import normalize, unnormalize

if not ray.is_initialized():
    ray.init(_temp_dir='/home/zl76saso/zl76saso/Zhiyuan/temp')


class BayesianOptimization:
    def __init__(self, data_file, target_props, feature_props=None, optimization_goal='maximize', scaler_method='standard', model_list=None, model_path=f'{os.getcwd()}/model_weights', stacking=True, acq_method='ucb', candidate_file=None, close_pool_initial_samples=10, close_pool_threshold=None, select_region=None):
        self.data_file = data_file
        self.target_props = target_props
        self.feature_props = feature_props
        self.optimization_goal = optimization_goal
        self.scaler_method = scaler_method
        self.io_manager = IOManager(method=scaler_method)
        self.model_list = model_list if model_list is not None else ['Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForest', 'SVR', 'MLPRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'XGBoost', 'LightGBM', 'GaussianProcess'] # 'LinearRegression' is deprecated
        self.model_path = model_path
        self.stacking = stacking
        self.acq_method = acq_method
        self.select_region = select_region

        # Data reading and scaling
        self.X, self.y = self.io_manager.read_data(data_file, target_props=target_props, feature_props=feature_props, handle_null=True, drop_non_numeric=True)
        self.y = -self.y if self.optimization_goal == 'minimize' else self.y
        if candidate_file is not None:
            self.X_cand = self.io_manager.read_candidate_data(candidate_file, target_props=target_props, feature_props=feature_props, drop_non_numeric=False)
        else:
            self.X_cand = None

        self.close_pool_initial_samples = min(close_pool_initial_samples, (len(self.y)//10+1))
        if close_pool_threshold is None:
            if self.select_region is None:
                self.cplb = np.min(self.y, axis=0)
                product = np.prod(self.y-self.cplb, axis=1)
                indexes = np.argsort(product)
                select_index = max(int(len(self.y)*0.99), len(self.y)-2)
                self.close_pool_threshold = product[indexes][select_index].item()
                self.close_pool_init = product[indexes][int(len(self.y)*0.6)].item()
            else:
                product = -np.abs(self.y-np.mean(self.select_region))
                indexes = np.argsort(product, axis=0)
                select_index = max(int(len(self.y)*0.99), len(self.y)-2)
                self.close_pool_threshold = product[indexes][select_index].item()
                self.close_pool_init = product[indexes][int(len(self.y)*0.6)].item()


    def close_pooling_test(self, n_bootstrap_sample_nums=20, n_iter=100, batch_size=10, hpar=0.1, save_all_info=True):
        
        print(f'Threshold is {self.close_pool_threshold}, init_sampling threshold is {self.close_pool_init}')
        X_train, X_candidate, y_train, y_candidate = train_test_split(self.X, self.y, test_size=1 - self.close_pool_initial_samples / len(self.X))
        # print(X_train.shape, y_train.shape)
        if self.select_region is None:
            current_best = np.max(np.prod(y_train-self.cplb, axis=1))
        else:
            current_best = np.max(-np.abs(y_train-np.mean(self.select_region)))

        while current_best >= self.close_pool_init:
            X_train, X_candidate, y_train, y_candidate = train_test_split(self.X, self.y, test_size=1 - self.close_pool_initial_samples / len(self.X))
            if self.select_region is None:
                current_best = np.max(np.prod(y_train-self.cplb, axis=1))
            else:
                current_best = np.max(-np.abs(y_train-np.mean(self.select_region)))

        with open('performance_record.txt', 'a') as f:
            f.write('iter_num\tnumber_of_samples\tmean_value_of_this_iteration\tstd_of_this_iteration\tbest_value_of_this_iteration\tcurrent_best_value\n')
    
        acquisition_function = AcquisitionFunction(hpar)

        for i in range(n_iter):

            X_scaled, y_scaled, candidate_X_scaled, candidate_y_scaled = self.io_manager.standardize_data(X_train, y_train, X_candidate, y_candidate)
            if self.select_region is not None:
                select_region = self.io_manager.scaler_y.transform(self.select_region)
            else:
                select_region = self.select_region
            model_evaluator = ModelEvaluator(X_scaled, y_scaled, file_path=self.model_path)
            
            target_model_res = {}
            for target_idx in range(y_train.shape[1]):
                # *** TBD: add the automatice column cls detection
                if self.stacking:
                    modelres = model_evaluator.evaluate_with_stacking(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)
                else:
                    modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)  
                target_model_res[target_idx] = modelres

            ### by adding the 'model_result' parameter, acquisition_function.select_next function can directly using the model saving in computer memory
        
            next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candidate_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, y_best=current_best, model_result=target_model_res, stack = self.stacking, select_region=select_region)
            
            ### if value of 'model_result' parameter is not provided, function will try to load the saved model from model_weight saving folder
            
            # next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candidate_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, y_best=current_best, stack = self.stacking)
            
            y_next = y_candidate[next_indexes]
            if self.select_region is None:
                current_best_next = np.max(np.prod(y_next-self.cplb, axis=1))
            else:
                current_best_next = np.max(-np.abs(y_next-np.mean(self.select_region)))
            print(f'train_best and sampling best: {current_best}, {current_best_next}')

            with open('performance_record.txt', 'a') as f:
                if self.select_region is None:
                    f.write(f'{i}\t{len(y_train)}\t{np.mean(np.prod(y_next-self.cplb, axis=1))}\t{np.std(np.prod(y_next-self.cplb, axis=1))}\t{current_best_next}\t{current_best}\n')
                else:
                    f.write(f'{i}\t{len(y_train)}\t{np.mean(-np.abs(y_next-np.mean(self.select_region)))}\t{np.std(-np.abs(y_next-np.mean(self.select_region)))}\t{current_best_next}\t{current_best}\n')
            
            if save_all_info:
                os.mkdir(f'{self.model_path}/{i}')
                os.popen(f'mv {self.model_path}/*.pkl {self.model_path}/{i}/')
                with open(f'{self.model_path}/{i}/data.pickle', 'wb') as f:
                    pickle.dump({'train': [X_train, y_train], 'test': [X_candidate, y_candidate]}, f)
            
            
            X_train = np.vstack([X_train, X_candidate[next_indexes]])
            y_train = np.vstack([y_train, y_next])
            X_candidate = np.delete(X_candidate, next_indexes, axis=0)
            y_candidate = np.delete(y_candidate, next_indexes, axis=0)

            if self.select_region is None:
                current_best = np.max(np.prod(y_train-self.cplb, axis=1))
            else:
                current_best = np.max(-np.abs(y_train-np.mean(self.select_region)))

            if current_best >= self.close_pool_threshold:
                print(f"Threshold {self.close_pool_threshold} reached at iteration {i+1}. The optimum target value is {current_best}")
                break


    ### haven't tested yet
    def optimize(self, batch_size=20, n_bootstrap_sample_nums=20, sampling_method='genetic_algorithm', num_candidate=100, n_samples=1000, iterations=30, hpar=0.1, lb=None, ub=None, all_surrogate_sampling=False, if_train=True):

        ## initializing
        print('Initialisation')
        X_train, y_train = self.X, self.y

        current_lb = np.min(self.y, axis=0)
        current_best = np.max(np.prod(y_train-current_lb, axis=1))
        
        if self.X_cand is None:
            X_scaled, y_scaled = self.io_manager.standardize_data(X=X_train, y=y_train, feature_range=(0, 1), custom_min=None, custom_max=None)
        else:
            X_cand = self.X_cand[:,1:]
            X_scaled, y_scaled, cand_X_scaled = self.io_manager.standardize_data(X=X_train, y=y_train, cand_X=X_cand, feature_range=(0, 1), custom_min=None, custom_max=None)

        if self.select_region is not None:
            select_region = self.io_manager.scaler_y.transform(self.select_region)
        else:
            select_region = self.select_region
        
        model_evaluator = ModelEvaluator(X_scaled, y_scaled, file_path=self.model_path)
        feature_dim = X_scaled.shape[1]
        sampler = Sampler(self.io_manager.scaler_X)
        acquisition_function = AcquisitionFunction(hpar)

        ## Model fitting
        if if_train:
            print('Fitting')
            target_model_res = {}
            for target_idx in range(y_train.shape[1]):
                # *** TBD: add the automatice column cls detection
                if self.stacking:
                    modelres = model_evaluator.evaluate_with_stacking(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)
                else:
                    modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)  
                target_model_res[target_idx] = modelres
        else:
            target_model_res = None

        ## Candidate generation
        if self.X_cand is None:
            print('Candidate generation')
            candidate_X_scaled = sampler.generate_candidate_parallel(method=sampling_method, feature_dim=feature_dim, model_results=target_model_res, model_list=self.model_list, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=None, all_surrogate_sampling=all_surrogate_sampling, parallel=False)
        else:
            candidate_X_scaled = cand_X_scaled

        ## Candidate selection 
        print('Candidate selection')
        next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candidate_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, model_result=target_model_res, stack = self.stacking, y_best=current_best, select_region=select_region)
        samples_next = self.io_manager.inverse_transform_X(candidate_X_scaled[next_indexes])
        pd_samples_next = pd.DataFrame(samples_next)
        pd_samples_next.to_csv(f'{os.getcwd()}/suggested_samples.csv', index=False)
        pd_samples_next_indexs = pd.DataFrame(next_indexes)
        pd_samples_next_indexs.to_csv(f'{os.getcwd()}/suggested_samples_indexes.csv', index=False)
        if self.X_cand is not None:
            pd_samples_next_origin = pd.DataFrame(self.X_cand[next_indexes])
            pd_samples_next_origin.to_csv(f'{os.getcwd()}/suggested_samples_original.csv', index=False)
        
        return samples_next, next_indexes



