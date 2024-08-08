import os, ray
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
    ray.init()


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
    def optimize(self, batch_size=20, n_bootstrap_sample_nums=20, sampling_method='genetic_algorithm', num_candidate=100, n_samples=1000, iterations=30, hpar=0.1, lb=None, ub=None, all_surrogate_sampling=False):

        ## initializing
        print('Initialisation')
        X_train, y_train = self.X, self.y
        current_lb = np.min(self.y, axis=0)
        current_best = np.max(np.prod(y_train-current_lb, axis=1))
        
        X_scaled, y_scaled = self.io_manager.standardize_data(X_train, y_train, feature_range=(0, 1), custom_min=None, custom_max=None)
        model_evaluator = ModelEvaluator(X_scaled, y_scaled, file_path=self.model_path)
        feature_dim = X_scaled.shape[1]
        sampler = Sampler(self.io_manager.scaler_X)
        acquisition_function = AcquisitionFunction(hpar)

        ## Model fitting 
        print('Fitting')
        target_model_res = {}
        for target_idx in range(y_train.shape[1]):
            # *** TBD: add the automatice column cls detection
            if self.stacking:
                modelres = model_evaluator.evaluate_with_stacking(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)
            else:
                modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)  
            target_model_res[target_idx] = modelres

        ## Candidate generation
        print('Candidate generation')
        candidate_X_scaled = sampler.generate_candidate_parallel(method=sampling_method, feature_dim=feature_dim, model_results=target_model_res, model_list=self.model_list, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=None, all_surrogate_sampling=all_surrogate_sampling, parallel=False)

        ## Candidate selection 
        print('Candidate selection')
        next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candidate_X_scaled, model_name_list=self.model_list, num_target=y_train.shape[1], model_path=self.model_path, batch_size=batch_size, model_result=target_model_res, stack = self.stacking, y_best=current_best)
        samples_next = self.io_manager.inverse_transform_X(candidate_X_scaled[next_indexes])
        pd_samples_next = pd.DataFrame(samples_next)
        pd_samples_next.to_csv(f'{os.getcwd()}/suggested_samples.csv', index=False)
        
        return samples_next


### ----------------------------------------- part for parameters fitting------------------------------------------------- ###

class ParamsFitting:
    def __init__(self, func):
        self.func = func

    ### suiatble for easy quick func, normally need 10,000 estimations
    def BO_BWO(self, SearchAgents_no=100, Max_iteration=1000, record_file=f'{os.getcwd()}/Data/fit_res.csv', target_column='loss', serial_function=False):
        res = BWO.BWO(self.func, list(self.func.lb), list(self.func.ub), self.func.dim, SearchAgents_no, Max_iteration, record_file=record_file, target_column=target_column, serial_function=serial_function)
        f_best, x_best = res.best, res.bestIndividual

        return {'X_best': x_best, 'Best_value': f_best[0]}

    ### Best performance, universal application
    def BO_Turbo(self, n_trust_regions=5, batch_size=8, max_evals=1000, suggest_X=None):
        n_init = 2 * self.func.dim  # 2*dim, which corresponds to 5 batches of 4
        turbo_m = TurboM(
            f=self.func,  # Handle to objective function
            lb=self.func.lb,  # Numpy array specifying lower bounds
            ub=self.func.ub,  # Numpy array specifying upper bounds
            n_init=n_init,  # Number of initial bounds from an Symmetric Latin hypercube design
            max_evals=max_evals,  # Maximum number of evaluations
            n_trust_regions=n_trust_regions,  # Number of trust regions
            batch_size=batch_size,  # How large batch size TuRBO uses
            suggest_X=suggest_X,  # if you have suggested params set,
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP kernel
            serial_function=False,  # set to true if your function cannot calculate multiple X simultaneously
            max_cholesky_size=1000,  # When we switch from Cholesky to Lanczos
            n_training_steps=100,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cuda",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
            record_file=f'{os.getcwd()}/Data/fit_res.csv',
            target_column=f'loss',
        )
        turbo_m.optimize()
        X = turbo_m.X  # Evaluated points
        fX = turbo_m.fX  # Observed values
        ind_best = np.argmin(fX)
        f_best, x_best = fX[ind_best], X[ind_best, :]

        return {'X_best': x_best, 'Best_value': f_best[0]}

    ### Best performance, universal application, less robust than BO_Turbo
    def BO_Turbo_bwo(self, n_trust_regions=5, batch_size=8, max_evals=1000, suggest_X=None):
        n_init = 2 * self.func.dim  # 2*dim, which corresponds to 5 batches of 4
        turbo_m = TurboM_bwo(
            f=self.func,  # Handle to objective function
            lb=self.func.lb,  # Numpy array specifying lower bounds
            ub=self.func.ub,  # Numpy array specifying upper bounds
            n_init=n_init,  # Number of initial bounds from an Symmetric Latin hypercube design
            max_evals=max_evals,  # Maximum number of evaluations
            n_trust_regions=n_trust_regions,  # Number of trust regions
            batch_size=batch_size,  # How large batch size TuRBO uses
            suggest_X=suggest_X,  # if you have suggested params set,
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP kernel
            serial_function=False,  # set to true if your function cannot calculate multiple X simultaneously
            max_cholesky_size=1000,  # When we switch from Cholesky to Lanczos
            n_training_steps=100,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cuda",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
            record_file=f'{os.getcwd()}/Data/fit_res.csv',
            target_column=f'loss',
        )
        turbo_m.optimize()
        X = turbo_m.X  # Evaluated points
        fX = turbo_m.fX  # Observed values
        ind_best = np.argmin(fX)
        f_best, x_best = fX[ind_best], X[ind_best, :]

        return {'X_best': x_best, 'Best_value': f_best[0]}

    ### Slow, robust, but least efficient
    def BO_Boosting(self, n_pts=10, max_iteratio=100):
        sobol = SobolEngine(dimension=self.func.dim, scramble=True)
        init_X = sobol.draw(n=n_pts)
        init_X = unnormalize(init_X, self.func.bounds).cpu().numpy()
        init_y = self.func(init_X)
        target_props = ['loss']
        data_file = os.path.join(os.getcwd(), 'Data/fit_res.csv')
        
        all_X = []
        all_y = []
        
        for iter in range(max_iteratio):
            BO = BayesianOptimization(data_file, target_props, model_list=None, optimization_goal='minimize', acq_method='ucb', scaler_method='minmax', stacking=True)
            samples_next_X = BO.optimize(batch_size=20, n_bootstrap_sample_nums=20, sampling_method='monte_carlo', num_candidate=1000, n_samples=1000, iterations=20, hpar=0.1, lb=self.func.lb, ub=self.func.ub, all_surrogate_sampling=False)
            samples_next_y = self.func(samples_next_X)
            
            # 记录每一步的samples_next_X和samples_next_y
            all_X.append(samples_next_X)
            all_y.append(samples_next_y)
            
            print(f'Iteration {iter+1}: next minimum: {np.min(samples_next_y)}\n\n\n')
        
        all_X = np.vstack(all_X)
        all_y = np.hstack(all_y)
        
        ind_best = np.argmin(all_y)
        f_best, x_best = all_y[ind_best], all_X[ind_best, :]
        
        return {'X_best': x_best, 'Best_value': f_best[0]}

### Abstract IO and ExecuteModule for ParamsFitting
class ParamsFittingIO:
    def __init__(self, root_path, input_file_name='in.lua', output_file_name='out.Quanty', input_templates_file_name='10_RIXS_L23_M45.lua', target_file_name='10_RIXS_L23_M45.Quanty'):
        self.root_path = root_path
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.input_templates_file_name = input_templates_file_name
        self.target_file_name = target_file_name

    ### ParamsFittingExecuteModule output reading, for read the simulation output or automatic experiment output from ParamsFittingExecuteModule
    def read_output(self, folder):
        raise NotImplementedError

    ### ParamsFittingExecuteModule target reading, for read the to be fitted target value
    def read_target(self, folder):
        raise NotImplementedError

    ### ParamsFitting suggestted parameter reading, if suggestted parameters can not be automatically passed to fitting module ()
    def read_parameters(self, folder):
        raise NotImplementedError

    ### used for value interpolation, for example, the ParamsFittingExecuteModule output image has different resolution compared with target image
    def value_interpolation(self, output):
        raise NotImplementedError

    ### modify the input file of ParamsFittingExecuteModule
    def modify_input(self, dict_para):
        raise NotImplementedError

### part for Executing simulation or experiment. This part will be act as 'func' parameter in the ParamsFitting
class ParamsFittingExecuteModule:
    def __init__(self, dim=15, lb=None, ub=None):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.bounds = np.stack([self.lb, self.ub]) if lb is not None and ub is not None else None

    def __call__(self, X):
        IO = ParamsFittingIO(root_path)
        ### 1. read in target_file, get target value;
        ### 2. modify the input file/files according to X/Xs
        ### 3. Executing simulation/experiment to get output file
        ### 3.5. value interpolation (optional)
        ### 4. compare loss between output and target: eg. loss = abs(target-output)
        ### 5. write to csv file by following command:  self.append_to_csv(root_path, X, loss)
        ### 6. return loss
        raise NotImplementedError

    ### can be applied to both single input or batch input
    def append_to_csv(self, root_path, X_batch, y_batch, target_name='loss', feature_names=None, file_name='Data/fit_res.csv'):
        # Define the CSV file path
        csv_file = os.path.join(root_path, file_name)

        if os.path.isfile(csv_file):
            existing_df = pd.read_csv(csv_file)
            existing_columns = existing_df.columns.tolist()
            if target_name not in existing_columns:
                raise ValueError(f"Missing target columns: {target_name}")
            else:
                feature_names = [col for col in existing_columns if col != target_name]

        # Ensure X_batch and y_batch are 2D arrays
        if X_batch.ndim == 1:
            X_batch = np.expand_dims(X_batch, axis=0)

        # Create a dictionary with column names and values
        if feature_names is None:
            feature_names = ['index' + str(i) for i in range(X_batch.shape[1])]

        data = {feature_names[i]: X_batch[:, i] for i in range(X_batch.shape[1])}
        data[target_name] = y_batch.reshape(-1)
        
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(data)

        # Check if the file exists
        if not os.path.isfile(csv_file):
            # If the file does not exist, create it with headers
            df.to_csv(csv_file, index=False)
        else:
            # If the file exists, append the data without headers
            df.to_csv(csv_file, mode='a', header=False, index=False)

### exapmle usage: 
### func = ParamsFittingExecuteModule(dim, lb, ub)
### fitting_results = ParamsFitting(fun_q).BO_Turbo()


