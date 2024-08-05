import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.io import IOManager
from src.surrogate_model import SurrogateModel, hyperparameter_optimization
from src.evaluation import ModelEvaluator
from src.acquisition_function import AcquisitionFunction
from src.sampling import Sampler

class BayesianOptimization:
    def __init__(self, data_file, target_props, feature_props=None, optimization_goal='maximize', scaler_method='standard', model_list=None, model_path=f'{os.getcwd()}/model_weights', stacking=False, acq_method='ucb', close_pool_initial_samples=10, close_pool_threshold=None):
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

        # Data reading and scaling
        self.X, self.y = self.io_manager.read_data(data_file, target_props=target_props, feature_props=feature_props, handle_null=True, drop_non_numeric=True)
        self.y = -self.y if self.optimization_goal == 'minimize' else self.y

        self.close_pool_initial_samples = min(close_pool_initial_samples, (len(self.y)//10+1))
        if close_pool_threshold is None:
            self.cplb = np.min(self.y, axis=0)
            product = np.prod(self.y-self.cplb, axis=1)
            indexes = np.argsort(product)
            select_index = max(int(len(self.y)*0.99), len(self.y)-50)
            self.close_pool_threshold = product[indexes][select_index].item()
            self.close_pool_init = product[indexes][int(len(self.y)*0.8)].item()
        

    def close_pooling_test(self, n_bootstrap_sample_nums=20, n_iter=100, batch_size=10, hpar=0.1):
        
        print(f'Threshold is {self.close_pool_threshold}, init_sampling threshold is {self.close_pool_init}')
        X_train, X_candidate, y_train, y_candidate = train_test_split(self.X, self.y, test_size=1 - self.close_pool_initial_samples / len(self.X))
        # print(X_train.shape, y_train.shape)
        current_best = np.max(np.prod(y_train-self.cplb, axis=1))

        while current_best >= self.close_pool_init:
            X_train, X_candidate, y_train, y_candidate = train_test_split(self.X, self.y, test_size=1 - self.close_pool_initial_samples / len(self.X))
            current_best = np.max(np.prod(y_train-self.cplb, axis=1))

        with open('performance_record.txt', 'a') as f:
            f.write('iter_num\tnumber_of_samples\tmean_value_of_this_iteration\tstd_of_this_iteration\tbest_value_of_this_iteration\tcurrent_best_value\n')
    
        acquisition_function = AcquisitionFunction(hpar)

        
        for i in range(n_iter):

            X_scaled, y_scaled = self.io_manager.standardize_data(X_train, y_train)
            candidate_X_scaled = self.io_manager.standardize_data(X_candidate)
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
        
            next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candidate_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, y_best=current_best, model_result=target_model_res, stack = self.stacking)
            
            ### if value of 'model_result' parameter is not provided, function will try to load the saved model from model_weight saving folder
            
            # next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candidate_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, y_best=current_best, stack = self.stacking)
            
            y_next = y_candidate[next_indexes]
            current_best_next = np.max(np.prod(y_next-self.cplb, axis=1))
            print(f'train_best and sampling best: {current_best}, {current_best_next}')

            with open('performance_record.txt', 'a') as f:
                f.write(f'{i}\t{len(y_train)}\t{np.mean(y_next-self.cplb)}\t{np.std(y_next-self.cplb)}\t{current_best_next}\t{current_best}\n')

            X_train = np.vstack([X_train, X_candidate[next_indexes]])
            y_train = np.vstack([y_train, y_next])
            X_candidate = np.delete(X_candidate, next_indexes, axis=0)
            y_candidate = np.delete(y_candidate, next_indexes, axis=0)

            current_best = np.max(np.prod(y_train-self.cplb, axis=1))

            if current_best >= self.close_pool_threshold:
                print(f"Threshold {self.close_pool_threshold} reached at iteration {i+1}. The optimum target value is {current_best}")
                break

    ### haven't tested yet
    def optimize(self, batch_size=20, n_bootstrap_sample_nums=20, sampling_method='genetic_algorithm', num_candidate=100, n_samples=1000, iterations=30, hpar=0.1):

        ## initializing
        print('Initialisation')
        X_train, y_train = self.X, self.y
        current_lb = np.min(self.y, axis=0)
        current_best = np.max(np.prod(y_train-current_lb, axis=1))
        
        X_scaled, y_scaled = self.io_manager.standardize_data(X_train, y_train)
        model_evaluator = ModelEvaluator(X_scaled, y_scaled, file_path=self.model_path)
        feature_dim = X_scaled.shape[1]
        sampler = Sampler(self.io_manager.scaler_X)
        acquisition_function = AcquisitionFunction(hpar)

        ## Model fitting 
        print('Fitting')
        target_model_res = {}
        for target_idx in range(y_train.shape[1]):
            modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)  # *** TBD: add the automatice column cls detection
            target_model_res[target_idx] = modelres

        ## Candidate generation 
        print('Candidate generation')
        candidate_X_scaled = []
        for target_i in range(y_train.shape[1]):
            for sg_model in self.model_list:
                if target_model_res is None:
                    print(f'load models {sg_model}_{target_i}')
                    file_path = f"{model_path}/{sg_model}_{target_i}_bootstrap.pkl"
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    models = data['models']
                    model_errors = data['model_errors']
                else:
                    models = target_model_res[target_i][sg_model]['model']
                    model_errors = target_model_res[target_i][sg_model]['error']
                    
                for model in models:
                    candidates = sampler.generate_candidates(sampling_method, model, feature_dim, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations)
                    candidate_X_scaled.append(candidates)
                    
        candidate_X_scaled = np.vstack(candidate_X_scaled)

        ## Candidate selection 
        print('Candidate selection')
        next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candidate_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, y_best=current_best)
        samples_next = self.io_manager.inverse_transform_X(candidate_X_scaled[next_indexes])
        pd_samples_next = pd.DataFrame(samples_next)
        pd_samples_next.to_csv(f'{os.getcwd()}/suggested_samples.csv', index=False)
        
        return samples_next
