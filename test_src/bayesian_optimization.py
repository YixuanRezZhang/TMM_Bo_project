import os, ray
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.io import IOManager
from src.surrogate_model import SurrogateModel, hyperparameter_optimization
from test_src.evaluation import ModelEvaluator
from test_src.acquisition_function import AcquisitionFunction
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

class MultiFidelityBayesianOptimization:
    def __init__(self, data_file, HF_props, LF_props, feature_props=None, optimization_goal='maximize', scaler_method='standard', 
                 model_list=None, model_path=f'{os.getcwd()}/model_weights', acq_method='ucb',
                 HF_initial_samples=10, LF_initial_samples=[20],HFcost=10,LFcost=[1],
                 close_pool_threshold=None):
        if HF_props in LF_props or HF_props == LF_props:
            raise ValueError('check your HF_props and LF_props!')
        if len(LF_props) != len(LF_initial_samples):
            raise ValueError('check your LF_initial_samples and LF_props!')
        if len(LF_props) != len(LFcost):
            raise ValueError('check your LF cost setting')        
        self.data_file = data_file
        # self.target_props = HF_props
        self.LF_props=LF_props
        self.feature_props = feature_props
        self.optimization_goal = optimization_goal
        self.scaler_method = scaler_method
        self.io_manager = IOManager(method=scaler_method)
        self.model_list = model_list if model_list is not None else ['Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForest', 'SVR', 'MLPRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'XGBoost', 'LightGBM'] # 'LinearRegression' is deprecated
        self.model_path = model_path
        self.LFprops_num = len(LF_props)
        self.HF_initial_samples=HF_initial_samples
        self.HFcost=HFcost
        self.acq_method = acq_method


        # Data reading and scaling
        _, self.y = self.io_manager.read_data(data_file, target_props=HF_props, feature_props=[])
        self.y = -self.y if self.optimization_goal == 'minimize' else self.y

        # Iterate over each property in LF_props
        for i, prop in enumerate(LF_props):
            # Read data for each property
            _, y_LF_temp = self.io_manager.read_data(data_file, target_props=[prop], feature_props=[])

            # Check optimization goal and adjust y_LF_temp accordingly
            if self.optimization_goal == 'minimize':
                y_LF_temp = -y_LF_temp
                

            # Dynamically create and assign attribute
            setattr(self, f'y_LF_{i}', y_LF_temp)    
            print(f"Attribute y_LF_{i} set with data (sample)")

        # Read features once if they are the same for all properties
        self.X, _ = self.io_manager.read_data(data_file, target_props=HF_props + LF_props, feature_props=None)

        self.HF_initial_samples = min(HF_initial_samples, (len(self.y)//10+1))
        self.cost=[]
        self.cost.append(HFcost)
        for i, LFIS in enumerate(LF_initial_samples):
            y_LF_length = len(getattr(self, f'y_LF_{i}'))
            LF_temp_initial_samples = min(LFIS, (y_LF_length // 10 + 1))
            # Dynamically create and assign attribute
            setattr(self, f'LF_{i}_initial_samples', LF_temp_initial_samples)
            print(f"Attribute LF_{i}_initial_samples set with data (sample): {LF_temp_initial_samples}")  
        for i, LFc in enumerate(LFcost):
            LF_temp_cost = LFc
            self.cost.append(LF_temp_cost)
            # Dynamically create and assign attribute
            setattr(self, f'LF_{i}_cost', LF_temp_cost)
            print(f"Attribute LF_{i}_cost: {LF_temp_cost}") 
            print(self.cost) 
        
        
        if close_pool_threshold is None:
            self.cplb = np.min(self.y, axis=0)
            product = np.prod(self.y-self.cplb, axis=1)
            indexes = np.argsort(product)
            select_index = int(len(self.y)*0.99)
            self.close_pool_threshold = product[indexes][select_index].item()

    def close_pooling_test(self, n_bootstrap_sample_nums=20, n_iter=100, batch_size=10, hpar=0.1):

        HFindices = np.arange(len(self.y))
        XHF_train, XHF_candidate, yHF_train, yHF_candidate, HFidx_train, HFidx_candidate = train_test_split(
            self.X, self.y, HFindices, test_size=1 - self.HF_initial_samples / len(self.X)
            )
        for i in range(self.LFprops_num):
            initial_samples = getattr(self, f'LF_{i}_initial_samples')
            y_current = getattr(self, f'y_LF_{i}')
            # Create the split and dynamically set global variables
            globals()[f'XLF{i}_train'], globals()[f'XLF{i}_candidate'], \
            globals()[f'yLF{i}_train'], globals()[f'yLF{i}_candidate'], \
            globals()[f'LF{i}idx_train'],globals()[f'LF{i}idx_candidate']= train_test_split(
                self.X, y_current, HFindices, test_size=1 - initial_samples / len(self.X)
            )
        current_best = np.max(np.prod(yHF_train-self.cplb, axis=1))
        while current_best >= self.close_pool_threshold:
            XHF_train, XHF_candidate, yHF_train, yHF_candidate, HFidx_train, HFidx_candidate = train_test_split(
                self.X, self.y, HFindices, test_size=1 - self.HF_initial_samples / len(self.X)
                )
            current_best = np.max(np.prod(yHF_train-self.cplb, axis=1))
        print(current_best)
        print(f'Threshold is {self.close_pool_threshold}')
        acquisition_function = AcquisitionFunction(hpar)
        for iter in range(n_iter):
            current_best = np.max(np.prod(yHF_train-self.cplb, axis=1))
            current_best_next = 0
            XHF_scaled, yHF_scaled = self.io_manager.standardize_data(XHF_train, yHF_train)
            for i in range(self.LFprops_num):
                    # Fetch the data from globals using the correct current names
                    XLF_train = globals()[f'XLF{i}_train']
                    yLF_train = globals()[f'yLF{i}_train']
                    # Standardize data
                    XLF_scaled, yLF_scaled = self.io_manager.standardize_data(XLF_train, yLF_train)
                    # Update the globals with the new scaled data
                    globals()[f'XLF{i}_scaled'] = XLF_scaled
                    globals()[f'yLF{i}_scaled'] = yLF_scaled

            X_candidate = XHF_candidate.copy()
            candidate_X_scaled = self.io_manager.standardize_data(X_candidate)

            HFevaluator = ModelEvaluator(XHF_scaled, yHF_scaled, file_path=self.model_path+ '_HF')
            for i in range(self.LFprops_num):
                    XLF_scaled = globals()[f'XLF{i}_scaled']
                    yLF_scaled = globals()[f'yLF{i}_scaled']
                    print(XLF_scaled.shape)
                    print(yLF_scaled.shape)
                    modified_path = f'{self.model_path}_LF{i}'
                    globals()[f'LF{i}evaluator']=ModelEvaluator(XLF_scaled, yLF_scaled, file_path=modified_path)
            HFmodelres = HFevaluator.evaluate(model_names=self.model_list, num_target=0,n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)

            for i in range(self.LFprops_num):
                LFevaluator=globals()[f'LF{i}evaluator']
                globals()[f'LF{i}modelres']=LFevaluator.evaluate(model_names=self.model_list, num_target=0,n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)

            acquisition_function = AcquisitionFunction(hpar)
            HFpathtest=self.model_path+ '_HF'
            HFres=acquisition_function.MF_predres(
                                                X_candidates=candidate_X_scaled, 
                                                model_name_list=self.model_list, 
                                                model_path=HFpathtest,
                                                    model_result=None, stack=False
                                                    )
            allLFres = []
            for i in range(self.LFprops_num):
                modified_path = f'{self.model_path}_LF{i}'
                globals()[f'LF{i}res']=acquisition_function.MF_predres(
                                                X_candidates=candidate_X_scaled, 
                                                model_name_list=self.model_list, 
                                                model_path=modified_path,
                                                    model_result=None, stack=False
                                                    )
                allLFres.append(globals()[f'LF{i}res'])

            mean_tuple = (HFres[0],) + tuple(LFres[0] for LFres in allLFres)
            std_tuple = (HFres[1],) + tuple(LFres[1] for LFres in allLFres)

            next_indexes, preferredlevel = acquisition_function.BOfusion_select_next(method=self.acq_method, X_candidate=candidate_X_scaled,HFidx_candidate=HFidx_candidate,
                                                                                mean_tuple=mean_tuple, std_tuple=std_tuple,cost=self.cost,
                                                                                batch_size=10, y_best=current_best)

            # 检查每个低保真度训练集索引是否已包含在 next_indexes 中
            for i in range(self.LFprops_num):
                LFidx_train = globals()[f'LF{i}idx_train']  # 获取当前保真度的训练集索引
                for j, idx in enumerate(next_indexes):
                    if idx in LFidx_train:
                        # 如果当前索引已存在于某个低保真度训练集中，将其首选保真度设置为0（高保真度）
                        preferredlevel[j] = 0
            print(next_indexes)
            print(preferredlevel)

            # 获取preferredlevel为0的索引，即计划在高保真度上采样的原始索引
            mask0 = preferredlevel == 0
            next_indexHF = next_indexes[mask0]

            if next_indexHF.size > 0:
                # 找到原始索引在当前HFidx_candidate中的位置
                positions = [np.where(HFidx_candidate == idx)[0][0] for idx in next_indexHF if idx in HFidx_candidate]

                # 使用找到的位置从高保真度候选数据集中采样
                XHF_samples = XHF_candidate[positions]
                yHF_samples = yHF_candidate[positions]

                # 更新高保真度训练集
                XHF_train = np.vstack([XHF_train, XHF_samples])
                yHF_train = np.vstack([yHF_train, yHF_samples])

                # 从高保真度候选集中删除已采样的样本
                XHF_candidate = np.delete(XHF_candidate, positions, axis=0)
                yHF_candidate = np.delete(yHF_candidate, positions, axis=0)

                # 同步更新高保真度的索引
                HFidx_train = np.concatenate([HFidx_train, HFidx_candidate[positions]])
                HFidx_candidate = np.delete(HFidx_candidate, positions)
                current_best_next = np.max(np.prod(yHF_samples-self.cplb, axis=1))
                print(f'train_best and sampling best: {current_best}, {current_best_next}')
            else:
                print(f'train_best and sampling best: {current_best}, not sampling in HF this iteration')

            for i in range(self.LFprops_num):
                mask = preferredlevel == (i + 1)
                # 获取低保真度的候选索引
                next_indexLF = next_indexes[mask]

                if next_indexLF.size > 0:
                    # 找到原始索引在当前LF{i}idx_candidate中的位置
                    positionsLF = [np.where(globals()[f'LF{i}idx_candidate'] == idx)[0][0] for idx in next_indexLF if idx in globals()[f'LF{i}idx_candidate']]

                    # 使用找到的位置从低保真度候选数据集中采样
                    XLF_samples = globals()[f'XLF{i}_candidate'][positionsLF]
                    yLF_samples = globals()[f'yLF{i}_candidate'][positionsLF]

                    # 更新低保真度训练集
                    globals()[f'XLF{i}_train'] = np.vstack([globals()[f'XLF{i}_train'], XLF_samples])
                    globals()[f'yLF{i}_train'] = np.vstack([globals()[f'yLF{i}_train'], yLF_samples])

                    # 从低保真度候选集中删除已采样的样本
                    globals()[f'XLF{i}_candidate'] = np.delete(globals()[f'XLF{i}_candidate'], positionsLF, axis=0)
                    globals()[f'yLF{i}_candidate'] = np.delete(globals()[f'yLF{i}_candidate'], positionsLF, axis=0)

                    # 同步更新低保真度的索引
                    globals()[f'LF{i}idx_train'] = np.concatenate([globals()[f'LF{i}idx_train'], globals()[f'LF{i}idx_candidate'][positionsLF]])
                    globals()[f'LF{i}idx_candidate'] = np.delete(globals()[f'LF{i}idx_candidate'], positionsLF)


            if current_best_next >= self.close_pool_threshold:
                print(f"Threshold {self.close_pool_threshold} reached at iteration {iter+1}. The optimum target value is {current_best_next}")
                break 


class MultiTaskBayesianOptimization:
    def __init__(self, data_file, Main_props, correlated_props, feature_props=None, optimization_goal='maximize', scaler_method='standard', 
                 model_list=None, model_path=f'{os.getcwd()}/model_weights', acq_method='ucb',
                 Main_initial_samples=10, correlated_initial_samples=[400],
                 close_pool_threshold=None):
        if Main_props in correlated_props or Main_props == correlated_props:
            raise ValueError('check your Main_props and correlated_props!')
        if len(correlated_props) != len(correlated_initial_samples):
            raise ValueError('check your correlated_initial_samples and correlated_props!')       
        self.data_file = data_file
        self.target_props = Main_props
        self.correlated_props=correlated_props
        self.feature_props = feature_props
        self.optimization_goal = optimization_goal
        self.scaler_method = scaler_method
        self.io_manager = IOManager(method=scaler_method)
        self.model_list = model_list if model_list is not None else ['Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForest', 'SVR', 'MLPRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'XGBoost', 'LightGBM'] # 'LinearRegression' is deprecated
        self.model_path = model_path
        self.correlatedprops_num = len(correlated_props)
        self.Main_initial_samples=Main_initial_samples
        self.acq_method = acq_method


        # Data reading and scaling
        _, self.y = self.io_manager.read_data(data_file, target_props=Main_props, feature_props=feature_props)
        self.y = -self.y if self.optimization_goal == 'minimize' else self.y

        # Iterate over each property in correlated_props
        for i, prop in enumerate(correlated_props):
            # Read data for each property
            _, y_correlated_temp = self.io_manager.read_data(data_file, target_props=[prop], feature_props=[])

            # Check optimization goal and adjust y_correlated_temp accordingly
            if self.optimization_goal == 'minimize':
                y_correlated_temp = -y_correlated_temp

            # Dynamically create and assign attribute
            setattr(self, f'y_correlated_{i}', y_correlated_temp)    
            print(f"Attribute y_correlated_{i} set with data (sample)")

        # Read features once if they are the same for all properties
        self.X, _ = self.io_manager.read_data(data_file, target_props= Main_props + correlated_props, feature_props=feature_props)

        self.Main_initial_samples = min(Main_initial_samples, (len(self.y)//10+1))

        for i, correlatedIS in enumerate(correlated_initial_samples):
            y_correlated_length = len(getattr(self, f'y_correlated_{i}'))
            correlated_temp_initial_samples = min(correlatedIS, (y_correlated_length // 10 + 1))
            # Dynamically create and assign attribute
            setattr(self, f'correlated_{i}_initial_samples', correlated_temp_initial_samples)
            print(f"Attribute correlated_{i}_initial_samples set with data (sample): {correlated_temp_initial_samples}")  

        
        
        if close_pool_threshold is None:
            self.cplb = np.min(self.y, axis=0)
            product = np.prod(self.y-self.cplb, axis=1)
            indexes = np.argsort(product)
            select_index = int(len(self.y)*0.99)
            self.close_pool_threshold = product[indexes][select_index].item()
    

    def close_pooling_test(self, n_bootstrap_sample_nums=20, n_iter=100, batch_size=3, hpar=0.1):

        XMain_train, XMain_candidate, yMain_train, yMain_candidate = train_test_split(
            self.X, self.y, test_size=1 - self.Main_initial_samples / len(self.X)
            )
        for i in range(self.correlatedprops_num):
            initial_samples = getattr(self, f'correlated_{i}_initial_samples')
            y_current = getattr(self, f'y_correlated_{i}')
            # Create the split and dynamically set global variables
            globals()[f'Xcorrelated{i}_train'], globals()[f'Xcorrelated{i}_candidate'], \
            globals()[f'ycorrelated{i}_train'], globals()[f'ycorrelated{i}_candidate'] = train_test_split(
                self.X, y_current, test_size=1 - initial_samples / len(self.X)
            )
        current_best = np.max(np.prod(yMain_train-self.cplb, axis=1))
        while current_best >= self.close_pool_threshold:
            XMain_train, XMain_candidate, yMain_train, yMain_candidate = train_test_split(
                self.X, self.y, test_size=1 - self.Main_initial_samples / len(self.X)
                )
            current_best = np.max(np.prod(yMain_train-self.cplb, axis=1))
        print(current_best)
        print(f'Threshold is {self.close_pool_threshold}')
        
        corr_model_save_paths=[]
        #提前训练相关task的stacking模型
        for i in range(self.correlatedprops_num):
            # Retrieve and standardize data
            Xcorrelated_train = globals()[f'Xcorrelated{i}_train']
            ycorrelated_train = globals()[f'ycorrelated{i}_train']
            Xcorrelated_scaled, ycorrelated_scaled = self.io_manager.standardize_data(Xcorrelated_train, ycorrelated_train)
            
            # Update the globals with the new scaled data
            globals()[f'Xcorrelated{i}_scaled'] = Xcorrelated_scaled
            globals()[f'ycorrelated{i}_scaled'] = ycorrelated_scaled
            
            # Print the shapes of the scaled data
            print(Xcorrelated_scaled.shape)
            print(ycorrelated_scaled.shape)
            
            # Modify the path and create a ModelEvaluator instance
            modified_path = f'{self.model_path}_correlated{i}'
            corr_model_save_paths.append(modified_path)
            globals()[f'correlated{i}evaluator'] = ModelEvaluator(Xcorrelated_scaled, ycorrelated_scaled, file_path=modified_path)
            
            correlatedevaluator=globals()[f'correlated{i}evaluator']
            globals()[f'correlated{i}modelres']=correlatedevaluator.evaluate_with_stacking(model_names=self.model_list, num_target=0, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False)  #??
            # print(globals()[f'correlated{i}modelres'])
        
    
        for iter in range(n_iter):
            current_best = np.max(np.prod(yMain_train-self.cplb, axis=1))
            current_best_next = 0
            XMain_scaled, yMain_scaled = self.io_manager.standardize_data(XMain_train, yMain_train)

            X_candidate = XMain_candidate.copy()
            candidate_X_scaled = self.io_manager.standardize_data(X_candidate)

            Mainevaluator = ModelEvaluator(XMain_scaled, yMain_scaled, file_path=self.model_path+ '_Main')
            ### stacking model list
            Mainmodel_train = Mainevaluator.MT_train_stacking_model(model_names=self.model_list, corr_model_save_paths=corr_model_save_paths, num_target=0,n_bootstrap_sample_nums=3, cls=False)
            
            acquisition_function = AcquisitionFunction(hpar)
            next_indexes = acquisition_function.MT_select_next(method=self.acq_method, X_candidate=candidate_X_scaled, Mainmodel_train=Mainmodel_train, batch_size=3, y_best=current_best)
    
            print(next_indexes)
            y_next = yMain_candidate[next_indexes]
            current_best_next = np.max(np.prod(y_next-self.cplb, axis=1))
            print(f'train_best and sampling best: {current_best}, {current_best_next}')

            with open('performance_record.txt', 'a') as f:
                f.write(f'{i}\t{len(yMain_train)}\t{np.mean(y_next-self.cplb)}\t{np.std(y_next-self.cplb)}\t{current_best_next}\t{current_best}\n')

            XMain_train = np.vstack([XMain_train, XMain_candidate[next_indexes]])
            yMain_train = np.vstack([yMain_train, y_next])
            XMain_candidate = np.delete(XMain_candidate, next_indexes, axis=0)
            yMain_candidate = np.delete(yMain_candidate, next_indexes, axis=0)

            current_best = np.max(np.prod(yMain_train-self.cplb, axis=1))

            if current_best >= self.close_pool_threshold:
                print(f"Threshold {self.close_pool_threshold} reached at iteration {i+1}. The optimum target value is {current_best}")
                break
            




