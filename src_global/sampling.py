import multiprocessing
import ray, os, logging, torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import ray
import importlib
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
from sko.ACA import ACA_TSP
from sko.DE import DE
from sko.IA import IA_TSP
from sko.AFSA import AFSA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KDTree
from scipy.spatial import distance
import faiss

@ray.remote
def small_model_predict(X, model):
    return model.predict(X)

@ray.remote
def model_predict(X, modelname, models, model_score, n_random_models, select_region=None):
    # 重构后的独立函数，包含所有必要参数
    if len(models) > n_random_models:
        chosen_models = np.random.choice(models, size=n_random_models, replace=False)
    else:
        chosen_models = models
    
    # 同步调用普通远程函数
    if modelname not in ['GP_gpu', 'KAN', 'FastKAN']:
        if len(chosen_models) > 1:
            futures = [small_model_predict.remote(X, model) for model in chosen_models]
            result = ray.get(futures)
        else:
            result = [model.predict(X) for model in chosen_models] # chosen_models[0].predict(X)
    else:
        result = []
        if modelname == 'GP_gpu':
            for model in chosen_models:
                model_pred = model.model.posterior(torch.tensor(X, dtype=torch.float32))
                result.append(model_pred.mean.detach().cpu().numpy().reshape(-1))
        else:
            for model in chosen_models:
                result.append(model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy().flatten())

    pred_res = np.mean(np.array(result), axis=0)
    if select_region is not None:
        pred_res = -np.abs(pred_res-select_region)
    return pred_res*model_score

class RandomizedAbstractSurrogateModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_list, model_results, num_target, n_random_models=2, model_path=f'{os.getcwd()}/model_weights', rand_all = False, select_region=None):
        self.model_list = model_list
        self.n_random_models = n_random_models
        self.num_target = num_target
        self.rand_all = rand_all
        self.select_region = select_region
        
        self.models = {}
        if not self.rand_all:
            for target_i in range(self.num_target):
                self.models[target_i] = {}
                model_score_lists = []
                for modelname in model_list:
                    if model_results is None:
                        file_path = f"{model_path}/{modelname}_{target_i}_bootstrap.pkl"
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        target_models = data['models']
                        target_model_errors = data['errors']
                    else:
                        target_models = model_results[target_i][modelname]['models']
                        target_model_errors = model_results[target_i][modelname]['errors']
                        
                    score_mu, score_std = np.mean(target_model_errors), np.std(target_model_errors)
                    model_score = np.clip(score_mu + 0.1*score_std, 0.0000001, np.inf)
                    
                    self.models[target_i][modelname] = target_models
                    model_score_lists.append(model_score)
    
                for count, modelname in enumerate(model_list):
                    self.models[target_i][f"{modelname}_score"] = model_score_lists[count]/sum(model_score_lists)
        else:
            for target_i in range(self.num_target):
                self.models[target_i] = {}
                self.models[target_i]['all_models'] = []
                for modelname in model_list:
                    if model_results is None:
                        file_path = f"{model_path}/{modelname}_{target_i}_bootstrap.pkl"
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        target_models = data['models']
                    else:
                        target_models = model_results[target_i][modelname]['models']
                    
                    self.models[target_i]['all_models'].extend(target_models)
                    
                self.models[target_i][f"all_models_score"] = 1
                self.model_list = ['all_models']

    def fit(self, X, y):
        pass  # 已经拟合好，不需要再 fit

    def predict(self, X):
        predictions = []
        
        for k in self.models.keys():

            if self.select_region is not None:
                logging.info(f'region selection: {select_region}')
                select_reg = np.mean(self.select_region, axis=0)[k]
            else:
                select_reg = None
     
            pred = ray.get([model_predict.remote(X, modelname, self.models[k][modelname], self.models[k][f"{modelname}_score"], self.n_random_models, select_region=select_reg) for modelname in self.model_list])
            pred = np.sum(np.array(pred), axis=0)
            predictions.append(pred)

        result = np.sqrt(np.sum(np.square(np.array(predictions)+3),axis=0))
        return result

# slow version
def map_to_candidate_list_normal(samples, candidate_list, used_indices=None, metric='euclidean'):

    if used_indices is None:
        used_indices = set()

    samples = [samples] if len(samples.shape) < 2 else samples

    distances = distance.cdist(samples, candidate_list, metric=metric)

    # Set distances of already used points to infinity
    for idx in used_indices:
        distances[idx] = np.inf

    nearest_indices = np.argmin(distances, axis=1)
    return candidate_list[nearest_indices], nearest_indices


def map_to_candidate_list_slow(samples, candidate_list, used_indices=None, metric='euclidean'):

    if used_indices is None:
        used_indices = set()

    tree = KDTree(candidate_list, metric=metric)

    mask = np.ones(len(candidate_list), dtype=bool)
    mask[list(used_indices)] = False

    filtered_candidates = candidate_list[mask]
    samples = [samples] if len(samples.shape) < 2 else samples

    # Query nearest neighbors for all samples
    dists, nearest_indices_in_filtered = tree.query(samples, k=1)
    nearest_indices_in_filtered = nearest_indices_in_filtered.flatten()

    # Map filtered indices back to original candidate indices
    full_indices = np.arange(len(candidate_list))[mask]
    nearest_indices = full_indices[nearest_indices_in_filtered]

    return candidate_list[nearest_indices], nearest_indices

def map_to_candidate_list(samples, candidate_list, used_indices=None, metric='euclidean'):

    if used_indices is None:
        used_indices = set()

    if metric != 'euclidean':
        raise ValueError("Faiss currently only supports 'euclidean' distance.")

    # Create a mask for unused points
    mask = np.ones(len(candidate_list), dtype=bool)
    mask[list(used_indices)] = False
    filtered_candidates = candidate_list[mask]

    # Build the Faiss index
    index = faiss.IndexFlatL2(filtered_candidates.shape[1])
    index.add(filtered_candidates.astype(np.float32))

    # Query nearest neighbors for all samples
    samples = [samples] if len(samples.shape) < 2 else samples
    _, nearest_indices_in_filtered = index.search(samples.astype(np.float32), k=1)
    nearest_indices_in_filtered = nearest_indices_in_filtered.flatten()

    # Map filtered indices back to original candidate indices
    full_indices = np.arange(len(candidate_list))[mask]
    nearest_indices = full_indices[nearest_indices_in_filtered]

    return candidate_list[nearest_indices], nearest_indices


# 'gaussian', 'bernoulli', 'monte_carlo', 'genetic_algorithm', 'particle_swarm', 'simulated_annealing', 'ant_colony', 'differential_evolution', 'immune_algorithm', 'artificial_fish_swarm'
## suggested order based on test result: 'simulated_annealing', 'monte_carlo', 'genetic_algorithm', 'differential_evolution', 'particle_swarm', 'immune_algorithm', 'artificial_fish_swarm'

class Sampler:
    def __init__(self, scaler, feature_bounds=None):
        self.scaler = scaler
        self.feature_bounds = feature_bounds

    def gaussian_sampling(self, feature_dim, mean=None, std_dev=None, num_candidate=10):
        
        if isinstance(self.scaler, MinMaxScaler):
            mean = 0.5 if mean is None else mean
            std_dev = 0.5 if std_dev is None else std_dev
        elif isinstance(self.scaler, StandardScaler):
            mean = 0 if mean is None else mean
            std_dev = 1 if std_dev is None else std_dev
        else:
            raise ValueError("Unsupported scaler type")

        samples = np.random.normal(loc=mean, scale=std_dev, size=(num_candidate, feature_dim))
        if isinstance(self.scaler, MinMaxScaler):
            samples = np.clip(samples, 0, 1)

        return samples

    def bernoulli_sampling(self, feature_dim, p = 0.5, n_test=1, num_candidate=10):
        #return bernoulli.rvs(p, size=num_candidate)
        return np.random.binomial(n_test, p, size=(num_candidate, feature_dim))

    def levy_flight(self, Lambda, size=1):
        """
        Generate step lengths for Levy flight using the Mantegna algorithm in a vectorized manner.
        :param Lambda: The Levy exponent.
        :param size: The number of steps to generate.
        :return: A numpy array of step lengths.
        """
        sigma1 = np.power((np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                          (np.math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
        sigma2 = 1
        u = np.random.normal(0, sigma1, size)
        v = np.random.normal(0, sigma2, size)
        steps = u / np.power(np.abs(v), 1 / Lambda)
        return steps


    # def monte_carlo_sampling(self, model, feature_dim, n_samples=100, iterations=20, perturbation_scale=0.3, Lambda=1.5, num_candidate=100, candidate_list=None):
    #     """
    #     Monte Carlo optimization sampling with iterative improvement and Levy flight.
    #     """

    #     if self.feature_bounds is not None:
    #         lb, ub = self.feature_bounds
    #     elif isinstance(self.scaler, MinMaxScaler):
    #         lb, ub = [0] * feature_dim, [1] * feature_dim
    #     elif isinstance(self.scaler, StandardScaler):
    #         lb, ub = [-9] * feature_dim, [9] * feature_dim  # Assuming 3 standard deviations as bounds
    #     else:
    #         lb, ub = None, None

    #     if isinstance(self.scaler, MinMaxScaler):
    #         search_space = np.clip(np.random.rand(n_samples, feature_dim),lb,ub)
    #     else:
    #         search_space = np.clip(np.random.randn(n_samples, feature_dim),lb,ub)

    #     if n_samples*iterations < num_candidate:
    #         raise ValueError("Total sampling points must be greater than num_candidate")

    #     best_sample = None
    #     top_samples = []
    #     top_samples_values = []
    #     used_indices = set()  # 用于记录已选择的候选点索引
        
    #     num_candidates_per_iteration = max(int(num_candidate/iterations)+1, int(n_samples/10)+1)
        
    #     for iteration in range(iterations):
    #         # samples = search_space[np.random.choice(search_space.shape[0], n_samples, replace=False)]
    #         samples = search_space
    #         values = -model.predict(samples).reshape(-1)   # default optimization in sko is minimize, thus '-model'

    #         best_value = values[0]
        
    #         sorted_indices = np.argsort(values.reshape(-1))
    #         top_indices = sorted_indices[:num_candidates_per_iteration]

    #         # if candidate_list is not None:
    #         #     current_top_samples, selected_idx = map_to_candidate_list(samples[top_indices], candidate_list, used_indices)
    #         #     [used_indices.add(i) for i in selected_idx]
    #         #     current_top_samples = np.array(current_top_samples)
    #         #     current_top_values = -model.predict(current_top_samples).reshape(-1)
    #         # else:
    #         current_top_samples = samples[top_indices]
    #         current_top_values = values[top_indices]

    #         if best_sample is None:
    #             best_sample = current_top_samples[np.argmin(current_top_values)]
    #         if min(current_top_values).item() < best_value:
    #             best_sample = current_top_samples[np.argmin(current_top_values)]

    #         top_samples.extend(current_top_samples)
    #         top_samples_values.extend(current_top_values)
        
    #         print(f"iter {iteration} top 10 values: {np.array(current_top_values)[np.argsort(current_top_values.reshape(-1))[:10]]}")
            
    #         # candi_samples = [[best_sample]]
    #         # current_top_indices = [0]+list(np.random.choice(np.arange(1,num_candidates_per_iteration), size=4, replace=False))
    #         # for i in current_top_indices:
    #         steps = self.levy_flight(Lambda, size=(n_samples, feature_dim)) * perturbation_scale
    #         indices = np.random.randint(len(current_top_samples), size=n_samples)
    #         random_top_samples = current_top_samples[indices]
    #         new_samples = (best_sample + random_top_samples) / 2 + steps * np.random.randn(n_samples, feature_dim)
    #         new_samples = np.clip(new_samples, lb, ub)
        
    #         search_space = np.vstack(([best_sample], new_samples))

    #     # values = -model.predict(search_space).reshape(-1)
    #     # if candidate_list is not None:
    #     #     # final_top_samples = np.array(candidate_list[list(used_indices)])
    #     # else:
    #     final_top_samples = top_samples
            
    #     final_top_values = -model.predict(final_top_samples).reshape(-1)
    #     final_samples_arg = np.argsort(np.array(final_top_values).reshape(-1))
    #     logging.info(f'best_value: {final_top_values[final_samples_arg[:10]]}')

    #     if candidate_list is not None:
    #         final_samples, selected_idx = map_to_candidate_list(np.array(final_top_samples)[final_samples_arg][:num_candidate], candidate_list, used_indices)
    #         return final_samples, selected_idx
    #     else:
    #         return np.array(final_top_samples)[final_samples_arg][:num_candidate]

    def monte_carlo_sampling(self, model, feature_dim, n_samples=100, iterations=20, perturbation_scale=0.3, Lambda=1.5, num_candidate=100, candidate_list=None):
        """
        Monte Carlo optimization sampling with iterative improvement and Levy flight.
        """

        if self.feature_bounds is not None:
            lb, ub = self.feature_bounds
        elif isinstance(self.scaler, MinMaxScaler):
            lb, ub = [0] * feature_dim, [1] * feature_dim
        elif isinstance(self.scaler, StandardScaler):
            lb, ub = [-9] * feature_dim, [9] * feature_dim  # Assuming 3 standard deviations as bounds
        else:
            lb, ub = None, None

        if isinstance(self.scaler, MinMaxScaler):
            search_space = np.clip(np.random.rand(n_samples, feature_dim),lb,ub)
        else:
            search_space = np.clip(np.random.randn(n_samples, feature_dim),lb,ub)

        if n_samples < num_candidate:
            raise ValueError("n_samples must be greater than num_candidate")

        best_sample = None
        best_value = float('inf')
        top_sample = []
        used_indices = set()  # 用于记录已选择的候选点索引
        
        num_candidates_per_iteration = int(num_candidate/4)
        
        for iteration in range(iterations):
            # samples = search_space[np.random.choice(search_space.shape[0], n_samples, replace=False)]
            samples = search_space
            values = -model.predict(np.atleast_2d(samples)).reshape(-1)   # default optimization in sko is minimize, thus '-model'
            best_value = values[0]
        
            sorted_indices = np.argsort(values.reshape(-1))
            top_indices = sorted_indices[:num_candidates_per_iteration]
            
            if candidate_list is not None:
                current_top_samples, selected_idx = map_to_candidate_list(samples[top_indices], candidate_list, used_indices)
                [used_indices.add(i) for i in selected_idx]
                current_top_samples = np.array(current_top_samples)
                current_top_values = -model.predict(np.atleast_2d(current_top_samples)).reshape(-1)
            else:
                current_top_samples = samples[top_indices]
                current_top_values = values[top_indices]

            if best_sample is None:
                best_sample = current_top_samples[np.argmin(current_top_values)]
            if min(current_top_values).item() < best_value:
                best_sample = current_top_samples[np.argmin(current_top_values)]

            print(f"iter {iteration} top 10 values: {np.array(current_top_values)[np.argsort(current_top_values.reshape(-1))[:10]]}")

            steps = self.levy_flight(1.5, size=(n_samples, feature_dim)) * perturbation_scale
            indices = np.random.randint(len(current_top_samples), size=n_samples)
            random_top_samples = current_top_samples[indices]
            new_samples = (best_sample + random_top_samples) / 2 + steps * np.random.randn(n_samples, feature_dim)
            new_samples = np.clip(new_samples, lb, ub)

            search_space = np.vstack(([best_sample], search_space, new_samples))

        # values = -model.predict(search_space).reshape(-1)
        if candidate_list is not None:
            final_top_samples = np.array(candidate_list[list(used_indices)])
        else:
            final_top_samples = search_space
            
        final_top_values = -model.predict(np.atleast_2d(final_top_samples)).reshape(-1)
        final_samples_arg = np.argsort(np.array(final_top_values).reshape(-1))

        if candidate_list is not None:
            final_samples, selected_idx = map_to_candidate_list(np.array(final_top_samples)[final_samples_arg][:num_candidate], candidate_list, used_indices)
            return final_samples, selected_idx
        else:
            return np.array(final_top_samples)[final_samples_arg][:num_candidate]

    def genetic_algorithm_sampling(self, model, feature_dim, population_size=1000, generations=20, num_candidate=10, candidate_list=None):
        # if population_size < num_candidate:
        #     raise ValueError("Total sampling points must be greater than num_candidate")
        num_candidate = min(population_size, num_candidate)

        if self.feature_bounds is not None:
            lb, ub = self.feature_bounds
        elif isinstance(self.scaler, MinMaxScaler):
            lb, ub = [0] * feature_dim, [1] * feature_dim
        elif isinstance(self.scaler, StandardScaler):
            lb, ub = [-9] * feature_dim, [9] * feature_dim  # Assuming 3 standard deviations as bounds
        else:
            lb, ub = None, None

        def fitness_func(individual):
            return -model.predict(np.atleast_2d(np.array(individual))).reshape(-1) #.reshape(1, -1).item()
            
        ga = GA(func=fitness_func, n_dim=feature_dim, size_pop=population_size, max_iter=generations, lb=lb, ub=ub)
        best_x, best_y = ga.run()
        print(f'{model.model_list} sampling best_value: {best_y}')

        sorted_indices = np.argsort(ga.Y.reshape(-1))
        X_selected = ga.X[sorted_indices][:num_candidate]
        Y_selected = ga.Y[sorted_indices][:num_candidate]
        
        if candidate_list is not None:
            used_indices = set()
            current_top_samples, selected_idx = map_to_candidate_list(X_selected, candidate_list, used_indices)
            [used_indices.add(i) for i in selected_idx] 
            current_top_samples = np.array(current_top_samples)
            
            current_top_values = -model.predict(np.atleast_2d(current_top_samples)).reshape(-1)
            candi_arg = np.argsort(current_top_values.reshape(-1))
            X_selected = current_top_samples[candi_arg]

        return np.array(X_selected[:num_candidate])

    ### currently somehow unstable, very easy to end up on boundary
    def particle_swarm_sampling(self, model, feature_dim, population_size=50, iterations=20, num_candidate=10, candidate_list=None):
        if population_size * iterations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        if self.feature_bounds is not None:
            lb, ub = self.feature_bounds
        elif isinstance(self.scaler, MinMaxScaler):
            lb, ub = [0] * feature_dim, [1] * feature_dim
        elif isinstance(self.scaler, StandardScaler):
            lb, ub = [-9] * feature_dim, [9] * feature_dim  # Assuming 3 standard deviations as bounds
        else:
            lb, ub = None, None

        logging.info(f'used sampler bounds: {lb, ub}')

        def fitness_func(x):
            return -model.predict(np.atleast_2d(np.array(x))).reshape(-1)#.reshape(1, -1).item()

        pso = PSO(func=fitness_func, n_dim=feature_dim, pop=population_size, max_iter=iterations, lb=lb, ub=ub)
        best_x, best_y = pso.run()
        print(f'{model.model_list} sampling best_value: {best_y}')

        sorted_indices = np.argsort(pso.Y.reshape(-1))
        X_selected = pso.X[sorted_indices][:num_candidate]
        Y_selected = pso.Y[sorted_indices][:num_candidate]
        
        if candidate_list is not None:
            used_indices = set()
            current_top_samples, selected_idx = map_to_candidate_list(X_selected, candidate_list, used_indices)
            [used_indices.add(i) for i in selected_idx]   
            current_top_samples = np.array(current_top_samples)
            current_top_values = -model.predict(np.atleast_2d(current_top_samples)).reshape(-1)
            candi_arg = np.argsort(current_top_values.reshape(-1))
            X_selected = current_top_samples[candi_arg]

        return np.array(X_selected[:num_candidate])

    ## SA cannot control boundary, modification needed
    def simulated_annealing_sampling(self, model, feature_dim, iterations=200, num_candidate=10, candidate_list=None):
        
        # iterations = (num_candidate//(2*iterations)+1)*iterations if iterations<num_candidate//2 else iterations
        SA_iter_nums = int(num_candidate/70)+1

        if self.feature_bounds is not None:
            lb, ub = self.feature_bounds
        elif isinstance(self.scaler, MinMaxScaler):
            lb, ub = [0] * feature_dim, [1] * feature_dim
        elif isinstance(self.scaler, StandardScaler):
            lb, ub = [-9] * feature_dim, [9] * feature_dim  # Assuming 3 standard deviations as bounds
        else:
            lb, ub = None, None

        logging.info(f'used sampler bounds: {lb, ub}')

        samples = []
        for SA_iter in range(SA_iter_nums):
            
            if isinstance(self.scaler, MinMaxScaler):
                x0 = np.clip(np.random.rand(feature_dim),lb,ub)
            else:
                x0 = np.clip(np.random.randn(feature_dim),lb,ub)
    
            def fitness_func(x):
                return -model.predict(np.atleast_2d(np.array(x))).reshape(-1)#.reshape(1, -1).item()

            def neighbor_func(x, temperature):
                step = np.random.normal(0, 1, size=len(x)) * 0.1 * temperature
                new_sample = x + step
                new_sample = np.clip(new_sample,lb,ub)           
            
                return new_sample

            sa = SA(func=fitness_func, x0=x0, T_max=100, T_min=1e-9, L=iterations)
            sa.neighbor = neighbor_func  # Use custom neighbor function to ensure bounds
            best_x, best_y = sa.run()
            print(f'{model.model_list} sampling best_value: {best_y}')
            samples.append(np.array(sa.best_x_history))

        _X_sel = np.vstack(samples)
        _Y_sel = -model.predict(np.atleast_2d(_X_sel)).reshape(-1)
        # print(f'select_sample_num: {len(_X_sel)}')

        if candidate_list is not None:
            used_indices = set()
            current_top_samples, selected_idx = map_to_candidate_list(_X_sel, candidate_list, used_indices)
            [used_indices.add(i) for i in selected_idx]   
            current_top_samples = np.array(current_top_samples)
            current_top_values = -model.predict(np.atleast_2d(current_top_samples)).reshape(-1)
            candi_arg = np.argsort(current_top_values.reshape(-1))
            X_selected = current_top_samples[candi_arg]
        else:
            X_arg = np.argsort(_Y_sel.reshape(-1))
            X_selected = _X_sel[X_arg]

        return np.array(X_selected[:num_candidate])

    # ant_colony_sampling is ont suitable when using for property optimization tasks 
    #蚁群算法用于寻找问题的最优路径，它主要应用于路径优化和组合优化问题。对于当前版本采样代码只是将方程写出使其可以运行，但并未针对采样问题进行特殊配置，现阶段其并不适用于采样
    def ant_colony_sampling(self, model, feature_dim, n_ants=50, n_best=5, n_iterations=100, num_candidate=10, candidate_list=None):
        if n_ants * n_iterations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        # if self.feature_bounds is not None:
        #     lb, ub = self.feature_bounds
        # elif isinstance(self.scaler, MinMaxScaler):
        #     lb, ub = [0] * feature_dim, [1] * feature_dim
        # elif isinstance(self.scaler, StandardScaler):
        #     lb, ub = [-9] * feature_dim, [9] * feature_dim  # Assuming 3 standard deviations as bounds
        # else:
        #     lb, ub = None, None

        # logging.info(f'used sampler bounds: {lb, ub}')

        def fitness_func(individual):
            return -model.predict(np.atleast_2d(np.array(individual))).reshape(-1)#.reshape(1, -1).item()

        aca = ACA_TSP(func=fitness_func, n_dim=feature_dim, size_pop=n_ants, max_iter=n_iterations, distance_matrix=np.random.rand(feature_dim, feature_dim))
        best_x, best_y = aca.run()
        print(f'{model.model_list} sampling best_value: {best_y}')

        sorted_indices = np.argsort(aca.Y.reshape(-1))
        X_selected = aca.X[sorted_indices][:num_candidate]
        Y_selected = aca.Y[sorted_indices][:num_candidate]

        if candidate_list is not None:
            used_indices = set()
            current_top_samples, selected_idx = map_to_candidate_list(X_selected, candidate_list, used_indices)
            [used_indices.add(i) for i in selected_idx]
            current_top_samples = np.array(current_top_samples)
            current_top_values = -model.predict(np.atleast_2d(current_top_samples)).reshape(-1)
            candi_arg = np.argsort(current_top_values.reshape(-1))
            X_selected = current_top_samples[candi_arg]

        return np.array(X_selected[:num_candidate])

    def differential_evolution_sampling(self, model, feature_dim, population_size=50, generations=300, num_candidate=10, candidate_list=None):
        if population_size * generations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        if self.feature_bounds is not None:
            lb, ub = self.feature_bounds
        elif isinstance(self.scaler, MinMaxScaler):
            lb, ub = [0] * feature_dim, [1] * feature_dim
        elif isinstance(self.scaler, StandardScaler):
            lb, ub = [-9] * feature_dim, [9] * feature_dim  # Assuming 3 standard deviations as bounds
        else:
            lb, ub = None, None

        logging.info(f'used sampler bounds: {lb, ub}')
        # print(f'used sampler bounds: {lb, ub}')

        def fitness_func(x):
            return -model.predict(np.atleast_2d(np.array(x))).reshape(-1)#.reshape(1, -1).item()

        de = DE(func=fitness_func, n_dim=feature_dim, size_pop=population_size, max_iter=generations, lb=lb, ub=ub)
        best_x, best_y = de.run()
        print(f'{model.model_list} sampling best_value: {best_y}')

        sorted_indices = np.argsort(de.Y.reshape(-1))
        X_selected = de.X[sorted_indices][:num_candidate]
        Y_selected = de.Y[sorted_indices][:num_candidate]

        if candidate_list is not None:
            used_indices = set()
            current_top_samples, selected_idx = map_to_candidate_list(X_selected, candidate_list, used_indices)
            [used_indices.add(i) for i in selected_idx]
            current_top_samples = np.array(current_top_samples)
            current_top_values = -model.predict(np.atleast_2d(current_top_samples)).reshape(-1)
            candi_arg = np.argsort(current_top_values.reshape(-1))
            X_selected = current_top_samples[candi_arg]

        return np.array(X_selected[:num_candidate])

    #免疫优化算法同样用于寻找问题的最优路径，它主要应用于路径优化和组合优化问题。
    def immune_algorithm_sampling(self, model, feature_dim, population_size=50, generations=20, num_candidate=10, candidate_list=None):
        if population_size * generations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        # if self.feature_bounds is not None:
        #     lb, ub = self.feature_bounds
        # elif isinstance(self.scaler, MinMaxScaler):
        #     lb, ub = [0] * feature_dim, [1] * feature_dim
        # elif isinstance(self.scaler, StandardScaler):
        #     lb, ub = [-9] * feature_dim, [9] * feature_dim  # Assuming 3 standard deviations as bounds
        # else:
        #     lb, ub = None, None

        # logging.info(f'used sampler bounds: {lb, ub}')

        def fitness_func(individual):
            return -model.predict(np.atleast_2d(np.array(individual))).reshape(-1)#.reshape(1, -1).item()

        ia = IA_TSP(func=fitness_func, n_dim=feature_dim, size_pop=population_size, max_iter=generations, prob_mut=0.2, T=0.7, alpha=0.95)
        best_x, best_y = ia.run()
        print(f'{model.model_list} sampling best_value: {best_y}')

        sorted_indices = np.argsort(ia.Y.reshape(-1))
        X_selected = ia.X[sorted_indices][:num_candidate]
        Y_selected = ia.Y[sorted_indices][:num_candidate]

        if candidate_list is not None:
            used_indices = set()
            current_top_samples, selected_idx = map_to_candidate_list(X_selected, candidate_list, used_indices)
            [used_indices.add(i) for i in selected_idx]   
            current_top_samples = np.array(current_top_samples)
            current_top_values = -model.predict(np.atleast_2d(current_top_samples)).reshape(-1)
            candi_arg = np.argsort(current_top_values.reshape(-1))
            X_selected = current_top_samples[candi_arg]

        return np.array(X_selected[:num_candidate])

    # 人工鱼群算法,其表现需要进一步测试
    def artificial_fish_swarm_sampling(self, model, feature_dim, population_size=50, iterations=20, num_candidate=10, candidate_list=None):
        if population_size * iterations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        # if self.feature_bounds is not None:
        #     lb, ub = self.feature_bounds
        # elif isinstance(self.scaler, MinMaxScaler):
        #     lb, ub = [0] * feature_dim, [1] * feature_dim
        # elif isinstance(self.scaler, StandardScaler):
        #     lb, ub = [-9] * feature_dim, [9] * feature_dim  # Assuming 3 standard deviations as bounds
        # else:
        #     lb, ub = None, None

        # logging.info(f'used sampler bounds: {lb, ub}')

        def fitness_func(x):
            return -model.predict(np.atleast_2d(np.array(x))).reshape(-1)#.reshape(1, -1).item()

        afsa = AFSA(func=fitness_func, n_dim=feature_dim, size_pop=population_size, max_iter=iterations, max_try_num=100, step=0.5, visual=0.3, q=0.98, delta=0.5)
        best_x, best_y = afsa.run()
        print(f'{model.model_list} sampling best_value: {best_y}')

        sorted_indices = np.argsort(afsa.Y.reshape(-1))
        X_selected = afsa.X[sorted_indices][:num_candidate]
        Y_selected = afsa.Y[sorted_indices][:num_candidate]

        if candidate_list is not None:
            used_indices = set()
            current_top_samples, selected_idx = map_to_candidate_list(X_selected, candidate_list, used_indices)
            [used_indices.add(i) for i in selected_idx] 
            current_top_samples = np.array(current_top_samples)
            current_top_values = -model.predict(np.atleast_2d(current_top_samples)).reshape(-1)
            candi_arg = np.argsort(current_top_values.reshape(-1))
            X_selected = current_top_samples[candi_arg]

        return np.array(X_selected[:num_candidate])


    # Suggest using monte_carlo sampling
    ## the robustness of [gaussian, monte_carlo] has been tested; [GA, PSO, DE, AFS] are functional, but slow; SA can not control the boundary; [ACA, IA] are used for path optimisation
    ### highly suggest the developeer hack the skopt code, replacing mulltiprocess in sko with Ray Actor for parallelisation. (since the multiuprocess may conflict with ray in resources and process management)

    ### Serial candidate generation
    def generate_candidates(self, method, model, feature_dim, num_candidate=100, n_samples=100, iterations=500, candidate_list=None):
        ### method, model, feature_dim are the requested inputs
        if method == 'gaussian':
            return self.gaussian_sampling(feature_dim, num_candidate=num_candidate)
        elif method == 'bernoulli':
            return self.bernoulli_sampling(feature_dim, num_candidate=num_candidate)
        elif method == 'monte_carlo':
            return self.monte_carlo_sampling(model, feature_dim, n_samples=n_samples, iterations=iterations, perturbation_scale=0.1, Lambda=1.5, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'genetic_algorithm':
            # GA = importlib.import_module('sko.GA').GA
            return self.genetic_algorithm_sampling(model, feature_dim, population_size=n_samples, generations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'particle_swarm':
            # PSO = importlib.import_module('sko.PSO').PSO
            return self.particle_swarm_sampling(model, feature_dim, population_size=n_samples, iterations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'differential_evolution':
            # DE = importlib.import_module('sko.DE').DE
            return self.differential_evolution_sampling(model, feature_dim, population_size=n_samples, generations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'artificial_fish_swarm':
            # AFSA = importlib.import_module('sko.AFSA').AFSA
            return self.artificial_fish_swarm_sampling(model, feature_dim, population_size=n_samples, iterations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'simulated_annealing':
            # SA = importlib.import_module('sko.SA').SA
            return self.simulated_annealing_sampling(model, feature_dim, iterations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'ant_colony':
            # ACA_TSP = importlib.import_module('sko.ACA').ACA_TSP
            return self.ant_colony_sampling(model, feature_dim, n_ants=n_samples, n_best=5, n_iterations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'immune_algorithm':
            # IA_TSP = importlib.import_module('sko.IA').IA_TSP
            return self.immune_algorithm_sampling(model, feature_dim, population_size=n_samples, generations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    ### ray function for parallel candidate generation
    @ray.remote
    def generate_candidates_ray(self, method, model, feature_dim, num_candidate=100, n_samples=1000, iterations=50, candidate_list=None):
        ### method, model, feature_dim are the requested inputs
        if method == 'gaussian':
            sample_results = self.gaussian_sampling(feature_dim, num_candidate=num_candidate)
        elif method == 'bernoulli':
            sample_results = self.bernoulli_sampling(feature_dim, num_candidate=num_candidate)
        elif method == 'monte_carlo':
            sample_results = self.monte_carlo_sampling(model, feature_dim, n_samples=n_samples, iterations=iterations, perturbation_scale=0.1, Lambda=1.5, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'genetic_algorithm':
            # GA = importlib.import_module('sko.GA').GA
            sample_results = self.genetic_algorithm_sampling(model, feature_dim, population_size=n_samples, generations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'particle_swarm':
            # PSO = importlib.import_module('sko.PSO').PSO
            sample_results = self.particle_swarm_sampling(model, feature_dim, population_size=n_samples, iterations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'differential_evolution':
            # DE = importlib.import_module('sko.DE').DE
            sample_results = self.differential_evolution_sampling(model, feature_dim, population_size=n_samples, generations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'artificial_fish_swarm':
            # AFSA = importlib.import_module('sko.AFSA').AFSA
            sample_results = self.artificial_fish_swarm_sampling(model, feature_dim, population_size=n_samples, iterations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'simulated_annealing':
            # SA = importlib.import_module('sko.SA').SA
            sample_results = self.simulated_annealing_sampling(model, feature_dim, iterations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'ant_colony':
            # ACA_TSP = importlib.import_module('sko.ACA').ACA_TSP
            sample_results = self.ant_colony_sampling(model, feature_dim, n_ants=n_samples, n_best=5, n_iterations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        elif method == 'immune_algorithm':
            # IA_TSP = importlib.import_module('sko.IA').IA_TSP
            sample_results = self.immune_algorithm_sampling(model, feature_dim, population_size=n_samples, generations=iterations, num_candidate=num_candidate, candidate_list=candidate_list)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        #print(f'finish {model.model_list}')
        
        return sample_results

    ### Parallel candidate generation
    def generate_candidates_parallel(self, method, feature_dim, model_results, model_list, num_target, model_path, num_candidate=100, n_samples=1000, iterations=50, candidate_list=None, n_random_models=2, Seperate=True, rand_all=False, select_region=None):

        candidate_list_ref = ray.put(candidate_list)
        
        if Seperate:
            candidate_X_per_model=[]
            for ag_model in model_list:
                logging.info(f'start {ag_model}')
                random_model = RandomizedAbstractSurrogateModel(model_list=[ag_model], model_results=model_results, num_target=num_target, n_random_models=n_random_models, model_path=model_path, select_region=select_region) 
                # candidate_X_per_model.append(self.generate_candidates(method=method, model=random_model, feature_dim=feature_dim, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=candidate_list))
                candidate_X_per_model.append(self.generate_candidates_ray.remote(self, method=method, model=random_model, feature_dim=feature_dim, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=candidate_list))
            if rand_all:
                random_model = RandomizedAbstractSurrogateModel(model_list=model_list, model_results=model_results, num_target=num_target, n_random_models=n_random_models+2, model_path=model_path, rand_all=rand_all, select_region=select_region)
                candidate_X_per_model.append(self.generate_candidates_ray.remote(self, method=method, model=random_model, feature_dim=feature_dim, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations*2, candidate_list=candidate_list))
            # candidate_X_scaled = np.vstack(candidate_X_per_model)
            ray_candidate_X_per_model = ray.get(candidate_X_per_model)
            candidate_X_scaled = np.vstack(ray_candidate_X_per_model)
        else:
            random_model = RandomizedAbstractSurrogateModel(model_list=model_list, model_results=model_results, num_target=num_target, n_random_models=n_random_models, model_path=model_path, select_region=select_region)
            candidate_X_scaled = self.generate_candidates(method=method, model=random_model, feature_dim=feature_dim, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations, candidate_list=candidate_list)
        
        return candidate_X_scaled

