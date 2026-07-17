import os, ray, pickle, psutil, logging, shutil, tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.io import IOManager
from src.surrogate_model import SurrogateModel, hyperparameter_optimization
from src.evaluation import ModelEvaluator
from src.acquisition_function import AcquisitionFunction
from src.sampling import Sampler
from datetime import datetime

import torch
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import normalize, unnormalize
import socket, subprocess, time

logging.basicConfig(
    filename='bo.log',
    filemode='a',
    format='%(asctime)s\t%(message)s',
    level=logging.INFO
    # level=logging.DEBUG
)

def _candidate_ray_temp_dirs():
    """Create candidate per-process Ray temp directories without host-specific paths."""
    base_candidates = [
        os.path.join(os.getcwd(), "tmp"),
        tempfile.gettempdir(),
    ]
    candidates = []
    last_error = None
    for base_dir in base_candidates:
        try:
            os.makedirs(base_dir, exist_ok=True)
            ray_temp_dir = os.path.join(base_dir, f"ray_{os.getpid()}")
            os.makedirs(ray_temp_dir, exist_ok=True)
            if ray_temp_dir not in candidates:
                candidates.append(ray_temp_dir)
        except OSError as exc:
            last_error = exc
            logging.warning("Cannot create Ray temp directory under %s: %s", base_dir, exc)
    if not candidates:
        raise RuntimeError("Unable to create a Ray temp directory") from last_error
    return candidates


def cleanup_ray_runtime(ray_temp_dir=None):
    """Stop Ray and remove the per-run temporary directory created by this project."""
    try:
        ray.shutdown()
    finally:
        if ray_temp_dir:
            abs_tmp = os.path.abspath(ray_temp_dir)
            basename = os.path.basename(abs_tmp)
            if basename.startswith("ray_") and os.path.isdir(abs_tmp):
                shutil.rmtree(abs_tmp, ignore_errors=True)
                logging.info("Removed Ray temp directory: %s", abs_tmp)


def initialize_ray():
    # Check total system memory.
    total_memory = psutil.virtual_memory().total
    logging.info(f"Total system memory: {total_memory / (1024**3):.2f} GB")

    # Check total and available space in /dev/shm.
    shm_stats = psutil.disk_usage('/dev/shm')
    shm_total = shm_stats.total
    shm_available = shm_stats.free
    virtual_num_cpus = psutil.cpu_count(logical=False)
    logging.info(f"Available CPUs: {virtual_num_cpus}")
    logging.info(f"/dev/shm total size: {shm_total / (1024**3):.2f} GB")
    logging.info(f"/dev/shm available size: {shm_available / (1024**3):.2f} GB")

    ray_temp_dirs = _candidate_ray_temp_dirs()

    # Detect whether initialization is running under SLURM.
    if os.environ.get('SLURM_JOB_ID') is not None:
        logging.info('SLURM environment detected.')

        # Read resources allocated by SLURM.
        num_cpus = min(
            int(os.environ.get('SLURM_JOB_CPUS_PER_NODE')),
            int(os.environ.get('SLURM_NTASKS', 1)) * int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
        )
        memory_per_cpu = int(os.environ.get('SLURM_MEM_PER_CPU')) * 1024 * 1024
        total_slurm_memory = min(num_cpus * memory_per_cpu, int(total_memory))
        object_store_memory = int(min(shm_available * 0.5, total_slurm_memory))
        num_cpus = min(num_cpus, virtual_num_cpus)

        logging.info(
            f"SLURM INFO: num_cpus={num_cpus}, memory_per_cpu={memory_per_cpu}, total_memory={total_slurm_memory}"
        )
        logging.info(f"Setting object_store_memory to {object_store_memory / (1024**3):.2f} GB")
        init_kwargs = {
            "_memory": int(object_store_memory),
            "include_dashboard": False,
            "logging_level": logging.INFO,
        }
        mode_name = "SLURM"
    else:
        logging.info('No SLURM environment detected. Initializing Ray locally.')
        init_kwargs = {
            "include_dashboard": False,
            "logging_level": logging.INFO,
        }
        mode_name = "local"

    last_error = None
    for ray_temp_dir in ray_temp_dirs:
        try:
            cleanup_ray_runtime()
            ray.init(_temp_dir=ray_temp_dir, **init_kwargs)
            logging.info(
                "Ray initialized successfully in %s mode with temp_dir=%s",
                mode_name,
                ray_temp_dir,
            )
            return ray_temp_dir
        except Exception as exc:
            last_error = exc
            cleanup_ray_runtime(ray_temp_dir)
            logging.warning(
                "Failed to initialize Ray in %s mode with temp_dir=%s: %s",
                mode_name,
                ray_temp_dir,
                exc,
            )

    try:
        cleanup_ray_runtime()
    except Exception as exc:
        logging.warning("Ray shutdown after failed initialization also failed: %s", exc)
    logging.error("Failed to initialize Ray in %s mode with all candidate temp dirs.", mode_name)
    raise RuntimeError(
        f"Failed to initialize Ray in {mode_name} mode with all candidate temp dirs."
    ) from last_error

class BayesianOptimization:
    def __init__(
        self,
        target_props,
        data_file=None,
        feature_props=None,
        drop_columns=None,
        optimization_goal='maximize',
        scaler_method='standard',
        model_list=None,
        model_path=f'{os.getcwd()}/model_weights',
        stacking=False,
        cross_val=False,
        acq_method='ucb',
        feature_lb=None,
        feature_ub=None,
        candidate_file=None,
        close_pool=False,
        close_pool_initial_samples=10,
        close_pool_threshold=None,
        select_region=None,
        uni_hyperparameter=False,
        max_cap=None,
    ):
        self.data_file = data_file
        self.target_props = sorted(target_props)
        self.feature_props = feature_props
        self.drop_columns = drop_columns
        self.optimization_goal = optimization_goal
        self.scaler_method = scaler_method
        self.io_manager = IOManager(method=scaler_method)
        self.model_list = model_list if model_list is not None else ['Ridge', 'Lasso', 'ElasticNet', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForest', 'SVR', 'MLPRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'CatBoost', 'XGBoost', 'LightGBM', 'FastKAN'] # 'LinearRegression' and kernel based method ('GP_gpu', 'GP_cpu', 'KRR') are deprecated
        self.model_path = model_path
        self.stacking = stacking
        self.cross_val = cross_val
        self.acq_method = acq_method
        s_region = np.zeros((2,len(self.target_props)))
        if select_region is not None:
            for num, key in enumerate(self.target_props):
                s_region[:,num] = select_region[key]
            self.select_region = s_region
        else:
            self.select_region = None
        logging.info("target_props=%s, select_region=%s", self.target_props, self.select_region)
        self.feature_bounds = [feature_lb, feature_ub] if feature_lb is not None and feature_ub is not None else None
        self.uni_hyperparameter = uni_hyperparameter
        if (
            max_cap is not None
            and (
                isinstance(max_cap, (bool, np.bool_))
                or not isinstance(max_cap, (int, np.integer))
                or max_cap <= 0
            )
        ):
            raise ValueError("max_cap must be None or a positive integer.")
        self.max_cap = None if max_cap is None else int(max_cap)
        logging.info("Configured max_cap=%s", self.max_cap)

        try:
            self.ray_temp_dir = initialize_ray()
        except Exception as e:
            self.ray_temp_dir = None
            logging.error(f"Error during Ray initialization: {e}")

        # Data reading and scaling
        if data_file is not None:
            self.X, self.y = self.io_manager.read_data(data_file, target_props=self.target_props, feature_props=feature_props, drop_columns=drop_columns, handle_null=True, drop_non_numeric=True)
            self.y = -self.y if self.optimization_goal == 'minimize' else self.y
        if candidate_file is not None:
            self.X_cand = self.io_manager.read_candidate_data(candidate_file, target_props=self.target_props, feature_props=feature_props, drop_columns=drop_columns, drop_non_numeric=True)
        else:
            self.X_cand = None

        if close_pool:
            self.close_pool_initial_samples = min(close_pool_initial_samples, (len(self.y)//10+1))
            if close_pool_threshold is None:
                if self.select_region is None:
                    # 1. Store per-target min/max values for y normalization.
                    self.min_vals = np.min(self.y, axis=0)-0.001
                    self.max_vals = np.max(self.y, axis=0)+0.001
                    self.ranges = self.max_vals - self.min_vals
                    # 2. Normalize each target column in y.
                    normalized_y = (self.y - self.min_vals) / self.ranges
    
                else:
                    # Calculate distances for each target property separately
                    select_region_mean = np.mean(self.select_region, axis=0)
                    distances = -np.abs(self.y - select_region_mean)
    
                    self.min_vals = np.min(distances, axis=0)
                    self.max_vals = np.max(distances, axis=0)
                    self.ranges = self.max_vals - self.min_vals
    
                    normalized_y = (distances - self.min_vals) / self.ranges

                # Compute the product of each normalized target row.
                product = np.prod(normalized_y, axis=1)
                indexes = np.argsort(product)
                select_index = max(int(len(product) * 0.99), len(product)-5)
                select_index_init = int(len(product) * 0.5)

                # Compute the threshold.
                self.close_pool_threshold = product[indexes][select_index]
                self.close_pool_init_index = select_index_init
                self.data_index = indexes
                self.close_pool_init_threshold = product[indexes][select_index_init]
            else:
                self.min_vals = np.min(self.y, axis=0)-0.001
                self.max_vals = np.max(self.y, axis=0)+0.001
                self.ranges = self.max_vals - self.min_vals
                
                normalized_y = (self.y - self.min_vals) / self.ranges
                product = np.prod(normalized_y, axis=1)
                indexes = np.argsort(product)
                select_index = max(int(len(product) * 0.99), len(product)-10)
                select_index_init = int(len(product) * 0.5)

                # Compute the threshold.
                self.close_pool_threshold = (close_pool_threshold - self.min_vals) / self.ranges
                self.close_pool_init_index = select_index_init
                self.data_index = indexes
                self.close_pool_init_threshold = product[indexes][select_index_init]

    def _resolve_budget_max_cap(self, batch_size=None, n_iter=None):
        if self.max_cap is not None:
            logging.info("Using explicit max_cap=%s.", self.max_cap)
            return self.max_cap
        if batch_size is not None and n_iter is not None:
            automatic_cap = int(batch_size) * int(n_iter)
            if automatic_cap <= 0:
                raise ValueError(
                    "batch_size * n_iter must be positive when deriving max_cap."
                )
            logging.info(
                "No explicit max_cap; using batch_size * n_iter = %s.",
                automatic_cap,
            )
            return automatic_cap
        logging.info(
            "No explicit max_cap; using estimate_budget's default cap."
        )
        return None

    def cleanup_runtime(self):
        cleanup_ray_runtime(getattr(self, 'ray_temp_dir', None))
        self.ray_temp_dir = None

    def compute_normalized_product(self, y_values):
        normalized = (y_values - self.min_vals) / self.ranges
        product = np.prod(normalized, axis=1)
        return product

    def compute_normalized_product_region(self, y_values):
        select_region_mean = np.mean(self.select_region, axis=0)
        distances = -np.abs(y_values - select_region_mean)
        normalized = (distances - self.min_vals) / self.ranges
        product = np.prod(normalized, axis=1)
        return product

    def custom_train_test_split(self, random_state=None):

        if not hasattr(self, 'data_index') or not hasattr(self, 'close_pool_init_index'):
            raise ValueError("Ensure 'self.data_index' and 'self.close_pool_init_index' are set.")

        rng = np.random.default_rng(seed=random_state)

        self.close_pool_initial_samples = min(self.close_pool_initial_samples, self.close_pool_init_index)
        train_indices = rng.choice(self.data_index[:self.close_pool_init_index], size=self.close_pool_initial_samples, replace=False)
        candidate_indices = np.setdiff1d(np.arange(len(self.X)), train_indices)
    
        return self.X[train_indices], self.X[candidate_indices], self.y[train_indices], self.y[candidate_indices]
    
    def close_pooling_test(self, n_bootstrap_sample_nums=20, n_iter=100, batch_size=10, hpar=0.1, save_all_info=True, sampling_method='genetic_algorithm', num_candidate=100, n_samples=200, iterations=30, candidate_sampling=False, diversity_method=False, use_data_correlation=False, use_model_correlation=False):
        
        logging.info(f'Threshold is {self.close_pool_threshold}, init_sampling threshold is {self.close_pool_init_threshold}')
        budget_max_cap = self._resolve_budget_max_cap(
            batch_size=batch_size, n_iter=n_iter
        )

        iter_info = ''
        if os.path.exists('./performance_record.txt'):
            with open('./performance_record.txt', 'r') as performance_file:
                for line in performance_file:
                    iter_info = line
        if 'iter_num' not in iter_info and os.path.exists(f'./model_weights/0'):
            iteration = max([int(i) for i in os.listdir(f'./model_weights/') if i.isdigit()])
            logging.info(f'resume from iteration {iteration}')
            csv_file = f'./model_weights/{iteration}/data.pickle'
            with open(f'{csv_file}', 'rb') as f:
                data = pickle.load(f)
                X_train, y_train = data['train'][0], data['train'][1]
                X_candidate, y_candidate = data['test'][0], data['test'][1]
                products = self.compute_normalized_product(y_train) if self.select_region is None else self.compute_normalized_product_region(y_train)
                current_best = np.max(products)
                logging.info(f'current_best is {current_best}')
        else:
            logging.info('start from iteration 0')
            iteration = 0
            X_train, X_candidate, y_train, y_candidate = self.custom_train_test_split()
            products = self.compute_normalized_product(y_train) if self.select_region is None else self.compute_normalized_product_region(y_train)
            current_best = np.max(products)
            logging.info(f'current_best is {current_best}')
            with open('performance_record.txt', 'a') as f:
                f.write('iter_num\tnumber_of_samples\tmean_value_of_this_iteration\tstd_of_this_iteration\tbest_value_of_this_iteration\tcurrent_best_value\n')
        
        acquisition_function = AcquisitionFunction(hpar)

        for i in range(iteration, n_iter):

            X_scaled, y_scaled, candidate_X_scaled, candidate_y_scaled = self.io_manager.standardize_data(X_train, y_train, X_candidate, y_candidate, if_train=True, data_id=None)

            if self.feature_bounds is not None:
                dim = X_train.shape[1]
                fb = np.asarray(self.feature_bounds, dtype=float)
                if fb.ndim == 1:
                    assert fb.size == 2, "'feature_bounds' 1d input must only contain float [lb, ub]"
                    lb_vec = np.full(dim, fb[0], dtype=float)
                    ub_vec = np.full(dim, fb[1], dtype=float) 
                    fb = np.vstack([lb_vec, ub_vec])
                else:
                    assert fb.shape == (2, dim), (f"'feature_bounds' should be (2, {dim}), but receive {fb.shape}")
                scaled_feature_bounds = self.io_manager.scaler_X.transform(fb)
                logging.info("scaled_feature_bounds: %s", scaled_feature_bounds)
            else:
                scaled_feature_bounds = None

            select_region = self.io_manager.scaler_y.transform(self.select_region) if self.select_region is not None else None
            model_evaluator = ModelEvaluator(
                X_scaled,
                y_scaled,
                file_path=self.model_path,
                max_cap=budget_max_cap,
            )
            sampler = Sampler(self.io_manager.scaler_X, scaled_feature_bounds)
            feature_dim = X_scaled.shape[1]
            
            target_model_res = {}
            for target_idx in range(y_train.shape[1]):
                # *** TBD: add the automatice column cls detection
                if self.stacking:
                    modelres = model_evaluator.evaluate_with_stacking(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False, cross_val=self.cross_val, uni_hyper=self.uni_hyperparameter)
                else:
                    modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False, cross_val=self.cross_val, uni_hyper=self.uni_hyperparameter)  
                target_model_res[target_idx] = modelres

            if use_data_correlation:
                if y_train.shape[1] > 1:
                    residual_dict = model_evaluator.calculate_and_save_residual_correlation(model_names=self.model_list, stacking=self.stacking)
                else:
                    use_data_correlation = False

            if y_train.shape[1] > 1 and use_model_correlation:
                if select_region is not None:
                    scale_select_region_mean = np.mean(select_region, axis=0)
                    distances = -np.abs(y_scaled - scale_select_region_mean)
                else:
                    distances = y_scaled

                distances_min_vals = np.min(distances, axis=0)-0.001
                distances_max_vals = np.max(distances, axis=0)+0.001
                distances_ranges = distances_max_vals - distances_min_vals

                normalized_y_scale = (distances - distances_min_vals) / distances_ranges
                scale_product = np.prod(normalized_y_scale, axis=1).reshape(-1,1)
                # scale_product = np.mean(normalized_y_scale, axis=1).reshape(-1,1)
                # scale_product_mean, scale_product_std = np.mean(scale_product), np.std(scale_product)
                # norm_product = (scale_product-scale_product_mean)/scale_product_std
                # product_region = (1-scale_product_mean)/scale_product_std
                # norm_product = scale_product-0.5
                # product_region = 0.5
                norm_product = scale_product
                product_region = 1.0
                model_evaluator.y_train = np.hstack([normalized_y_scale, norm_product])
                # print(model_evaluator.y_train.shape)
                # print(scale_product)
                
                modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=len(self.target_props), n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False, cross_val=self.cross_val, uni_hyper=self.uni_hyperparameter)
                target_model_res[len(self.target_props)] = modelres
                if select_region is not None:
                    select_region = np.hstack([select_region, np.array([[product_region],[product_region]])])
                num_of_target_props = len(self.target_props)+1
                
            else:
                num_of_target_props = len(self.target_props)
                
            if candidate_sampling:
                logging.info('Candidate screening')
                candi_X_scaled = sampler.generate_candidates_parallel(method=sampling_method, feature_dim=feature_dim, model_results=target_model_res, model_list=self.model_list, num_of_targets=num_of_target_props, model_path=self.model_path, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=candidate_X_scaled, select_region=select_region)
            else:
                candi_X_scaled = candidate_X_scaled

            logging.info("select_next num_of_target_props: %s", num_of_target_props)
            labels = model_evaluator.global_labels
            next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candi_X_scaled, model_name_list=self.model_list, num_of_targets=num_of_target_props, model_path=self.model_path, batch_size=batch_size, X_train=model_evaluator.X_train, y_value=model_evaluator.y_train, model_result=target_model_res, stack = self.stacking, select_region=select_region, diversity_method=diversity_method, optimization_goal=self.optimization_goal, use_correlation=use_data_correlation, use_model_correlation=use_model_correlation, train_clsuter_labels=labels)
            
            y_next = y_candidate[next_indexes]
            products_next = self.compute_normalized_product(y_next) if self.select_region is None else self.compute_normalized_product_region(y_next)
            current_best_next = np.max(products_next)
            
            logging.info(f'train_best and sampling best: {current_best}, {current_best_next}')

            with open('performance_record.txt', 'a') as f:
                f.write(f'{i}\t{len(y_train)}\t{np.mean(products_next)}\t{np.std(products_next)}\t{current_best_next}\t{current_best}\n')
            
            if save_all_info:
                os.makedirs(f'{self.model_path}/{i}', exist_ok=True)
                os.popen(f'mv {self.model_path}/*.pkl {self.model_path}/{i}/')
                with open(f'{self.model_path}/{i}/data.pickle', 'wb') as f:
                    pickle.dump({'train': [X_train, y_train], 'test': [X_candidate, y_candidate]}, f)
            
            X_train = np.vstack([X_train, X_candidate[next_indexes]])
            y_train = np.vstack([y_train, y_next])
            X_candidate = np.delete(X_candidate, next_indexes, axis=0)
            y_candidate = np.delete(y_candidate, next_indexes, axis=0)

            products = self.compute_normalized_product(y_train) if self.select_region is None else self.compute_normalized_product_region(y_train)
            current_best = np.max(products)

            if current_best >= self.close_pool_threshold:
                logging.info(f"Threshold {self.close_pool_threshold} reached at iteration {i+1}. The optimum target value is {current_best}, the last selected set which contain maximum value is {y_next}")
                with open('performance_record.txt', 'a') as f:
                    f.write(f"Threshold {self.close_pool_threshold} reached at iteration {i+1}. The optimum target value is {current_best}")
                # Save results. The final training set is written as a CSV file.
                if hasattr(self, 'target_props'):
                    columns = [f'feature_{i}' for i in range(X_train.shape[1])] + self.target_props
                else:
                    columns = [f'feature_{i}' for i in range(X_train.shape[1])] + [f'target_{i}' for i in range(y_train.shape[1])]
            
                df_all = pd.DataFrame(np.hstack([X_train, y_train]), columns=columns)
                output_path = os.path.join(self.model_path, 'final_train_data.csv')
                df_all.to_csv(output_path, index=False)
                logging.info(f"Final training data saved to {output_path}")
                self.cleanup_runtime()
                break

        self.cleanup_runtime()

    def optimize(self, batch_size=20, n_bootstrap_sample_nums=20, sampling_method='genetic_algorithm', num_candidate=100, n_samples=100, iterations=200, hpar=0.1, if_train=True, candidate_sampling=False, n_random_models=2, seperate=True, diversity_method=True, use_data_correlation=False, use_model_correlation=False):

        ## initializing
        logging.info('Initialisation')
        budget_max_cap = self._resolve_budget_max_cap()
        X_train, y_train = self.X, self.y
        
        if self.X_cand is None:
            X_scaled, y_scaled = self.io_manager.standardize_data(X=X_train, y=y_train, minmax_feature_range=(0, 1), if_train=True, data_id=None)
        else:
            # X_cand = self.X_cand[:,1:]
            X_cand = self.X_cand[:,:]
            X_scaled, y_scaled, cand_X_scaled = self.io_manager.standardize_data(X=X_train, y=y_train, cand_X=X_cand, minmax_feature_range=(0, 1), if_train=True, data_id=None)

        if self.select_region is not None:
            select_region = self.io_manager.scaler_y.transform(self.select_region)
        else:
            select_region = None

        if self.feature_bounds is not None:
            dim = X_train.shape[1]
            fb = np.asarray(self.feature_bounds, dtype=float)
            if fb.ndim == 1:
                assert fb.size == 2, "'feature_bounds' 1d input must only contain float [lb, ub]"
                lb_vec = np.full(dim, fb[0], dtype=float)
                ub_vec = np.full(dim, fb[1], dtype=float) 
                fb = np.vstack([lb_vec, ub_vec])
            else:
                assert fb.shape == (2, dim), (f"'feature_bounds' should be (2, {dim}), but receive {fb.shape}")
                
            scaled_feature_bounds = self.io_manager.scaler_X.transform(fb)
            
            logging.info("scaled_feature_bounds: %s", scaled_feature_bounds)
        else:
            scaled_feature_bounds = None
        
        model_evaluator = ModelEvaluator(
            X_scaled,
            y_scaled,
            file_path=self.model_path,
            optimization_goal=self.optimization_goal,
            max_cap=budget_max_cap,
        )
        feature_dim = X_scaled.shape[1]
        sampler = Sampler(self.io_manager.scaler_X, scaled_feature_bounds)
        acquisition_function = AcquisitionFunction(hpar)

        ## Model fitting
        if if_train:
            logging.info('Fitting')
            target_model_res = {}
            for target_idx in range(y_train.shape[1]):
                # *** TBD: add the automatice column cls detection
                if self.stacking:
                    modelres = model_evaluator.evaluate_with_stacking(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False, cross_val=self.cross_val, uni_hyper=self.uni_hyperparameter)
                else:
                    modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=target_idx, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False, cross_val=self.cross_val, uni_hyper=self.uni_hyperparameter)
                target_model_res[target_idx] = modelres

            if use_data_correlation:
                if y_train.shape[1] > 1:
                    residual_dict = model_evaluator.calculate_and_save_residual_correlation(model_names=self.model_list, stacking=self.stacking)
                else:
                    use_data_correlation = False

            if y_train.shape[1] > 1 and use_model_correlation:
                if select_region is not None:
                    scale_select_region_mean = np.mean(select_region, axis=0)
                    distances = -np.abs(y_scaled - scale_select_region_mean)
                else:
                    distances = y_scaled

                distances_min_vals = np.min(distances, axis=0)-0.001
                distances_max_vals = np.max(distances, axis=0)+0.001
                distances_ranges = distances_max_vals - distances_min_vals

                normalized_y_scale = (distances - distances_min_vals) / distances_ranges

                scale_product = np.prod(normalized_y_scale, axis=1).reshape(-1,1)
                norm_product = scale_product
                product_region = 1.0
                model_evaluator.y_train = np.hstack([normalized_y_scale, norm_product])
                
                modelres = model_evaluator.evaluate(model_names=self.model_list, num_target=len(self.target_props), n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=False, cross_val=self.cross_val, uni_hyper=self.uni_hyperparameter)
                target_model_res[len(self.target_props)] = modelres
                if select_region is not None:
                    select_region = np.hstack([select_region, np.array([[product_region],[product_region]])])
                num_of_target_props = len(self.target_props)+1
            else:
                num_of_target_props = len(self.target_props)
                
        else:
            if y_train.shape[1] > 1 and use_model_correlation:
                if select_region is not None:
                    scale_select_region_mean = np.mean(select_region, axis=0)
                    distances = -np.abs(y_scaled - scale_select_region_mean)
                else:
                    distances = y_scaled

                distances_min_vals = np.min(distances, axis=0)-0.001
                distances_max_vals = np.max(distances, axis=0)+0.001
                distances_ranges = distances_max_vals - distances_min_vals

                normalized_y_scale = (distances - distances_min_vals) / distances_ranges
                scale_product = np.prod(normalized_y_scale, axis=1).reshape(-1,1)

                norm_product = scale_product
                product_region = 1.0
                model_evaluator.y_train = np.hstack([normalized_y_scale, norm_product])
                
                if select_region is not None:
                    select_region = np.hstack([select_region, np.array([[product_region],[product_region]])])
                num_of_target_props = len(self.target_props)+1
            else:
                num_of_target_props = len(self.target_props)
                
            target_model_res = None
            residual_dict = None
            # num_of_target_props = len(self.target_props)+1 if y_train.shape[1] > 1 and use_model_correlation else len(self.target_props)

        ## Candidate generation
        if self.X_cand is None:
            logging.info('Candidate generation')
            #generate_candidates_parallel(method, feature_dim, model_results, model_list, num_of_targets, model_path, num_candidate=100, n_samples=1000, iterations=50, candidate_list=None, n_random_models=2, Seperate=True)
            candidate_X_scaled = sampler.generate_candidates_parallel(method=sampling_method, feature_dim=feature_dim, model_results=target_model_res, model_list=self.model_list, num_of_targets=num_of_target_props, model_path=self.model_path, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=None, n_random_models=n_random_models, Seperate=seperate, select_region=select_region)
        elif candidate_sampling:
            logging.info('Candidate screening')
            candidate_X_scaled = sampler.generate_candidates_parallel(method=sampling_method, feature_dim=feature_dim, model_results=target_model_res, model_list=self.model_list, num_of_targets=num_of_target_props, model_path=self.model_path, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=cand_X_scaled, select_region=select_region)
        else:
            candidate_X_scaled = cand_X_scaled

        ## Candidate selection 
        assert np.all(candidate_X_scaled >= scaled_feature_bounds[0]) and np.all(candidate_X_scaled <= scaled_feature_bounds[1]), f'Out-of-bounds detected'
        logging.info('Candidate selection')
        # best_Y = np.max(y_scaled, axis=0)
        labels = model_evaluator.global_labels
        next_indexes = acquisition_function.select_next(method=self.acq_method, X_candidate=candidate_X_scaled, model_name_list=self.model_list, num_of_targets=num_of_target_props, model_path=self.model_path, batch_size=batch_size, X_train=model_evaluator.X_train, y_value=model_evaluator.y_train, model_result=target_model_res, stack = self.stacking, select_region=select_region, diversity_method=diversity_method, optimization_goal=self.optimization_goal, use_correlation=use_data_correlation, use_model_correlation=use_model_correlation, train_clsuter_labels=labels)
        samples_next = self.io_manager.inverse_transform_X(candidate_X_scaled[next_indexes])
        pd_samples_next = pd.DataFrame(samples_next)
        pd_samples_next.to_csv(f'{os.getcwd()}/suggested_samples.csv', index=False)
        pd_samples_next_indexs = pd.DataFrame(next_indexes)
        pd_samples_next_indexs.to_csv(f'{os.getcwd()}/suggested_samples_indexes.csv', index=False)
        if self.X_cand is not None:
            pd_samples_next_origin = pd.DataFrame(self.X_cand[next_indexes])
            pd_samples_next_origin.to_csv(f'{os.getcwd()}/suggested_samples_original.csv', index=False)

        self.cleanup_runtime()
        return samples_next, next_indexes


