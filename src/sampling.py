import multiprocessing
import ray
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
from src.evaluation import AbstractSurrogateModel

if not ray.is_initialized():
    ray.init()
### TBD: 检查模型模型从candidate_list中随机采样的功能和添加离散化采样函数

class Sampler:
    def __init__(self, scaler):
        self.scaler = scaler

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

    def levy_flight(self, Lambda):
        """
        Generate a step length for Levy flight using the Mantegna algorithm.
        """
        sigma1 = np.power((np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                          (np.math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
        sigma2 = 1
        u = np.random.normal(0, sigma1)
        v = np.random.normal(0, sigma2)
        step = u / np.power(np.abs(v), 1 / Lambda)
        return step

    def monte_carlo_sampling(self, model, feature_dim, n_samples=1000, iterations=20, perturbation_scale=0.1, Lambda=1.5, num_candidate=100, candidate_list=None):
        """
        Monte Carlo optimization sampling with iterative improvement and Levy flight.
        """
        if candidate_list is None:
            # Generate initial samples based on scaler
            if isinstance(self.scaler, MinMaxScaler):
                search_space = np.random.rand(n_samples, feature_dim)
            elif isinstance(self.scaler, StandardScaler):
                search_space = np.random.randn(n_samples, feature_dim)
            else:
                raise ValueError("Unsupported scaler type")
        else:
            search_space = candidate_list

        if n_samples*iterations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        best_sample = None
        best_value = float('inf')
        top_samples = []
        
        num_candidates_per_iteration = int(num_candidate / iterations) + 1
        
        for iteration in range(iterations):
            samples = search_space[np.random.choice(search_space.shape[0], n_samples, replace=False)]
            values = -model.predict(samples).reshape(-1)   # default optimization in sko is minimize, thus '-model'
        
            sorted_indices = np.argsort(values)
            top_indices = sorted_indices[:num_candidates_per_iteration]
            current_top_samples = samples[top_indices]
            current_top_values = values[top_indices]
        
            if current_top_values[0].item() < best_value:
                best_sample = current_top_samples[0]
                best_value = current_top_values[0].item()
        
            top_samples.extend(current_top_samples)
        
            new_samples = []
            for _ in range(n_samples):
                step = self.levy_flight(Lambda) * perturbation_scale
                new_sample = best_sample + step * np.random.randn(feature_dim)
                if isinstance(self.scaler, MinMaxScaler):
                    new_sample = np.clip(new_sample, 0, 1)
                elif isinstance(self.scaler, StandardScaler):
                    new_sample = np.clip(new_sample, -3, 3)  # Assuming 3 standard deviations as bounds
                new_samples.append(new_sample)
        
            new_samples = np.array(new_samples)
            search_space = np.vstack((search_space, new_samples))

        print(f'best_value: {best_value}')

        return np.array(top_samples[-num_candidate:])

    def genetic_algorithm_sampling(self, model, feature_dim, population_size=1000, generations=20, num_candidate=10, candidate_list=None):
        # if population_size < num_candidate:
        #     raise ValueError("Total sampling points must be greater than num_candidate")
        num_candidate = min(population_size, num_candidate)

        if candidate_list is None:
            if isinstance(self.scaler, MinMaxScaler):
                lb, ub = [0] * feature_dim, [1] * feature_dim
            elif isinstance(self.scaler, StandardScaler):
                lb, ub = [-3] * feature_dim, [3] * feature_dim  # Assuming 3 standard deviations as bounds
            else:
                raise ValueError("Unsupported scaler type")
        else:
            lb, ub = np.min(candidate_list, axis=0), np.max(candidate_list, axis=0)

        def fitness_func(individual):
            return -model.predict(np.array(individual).reshape(1, -1)).item()
            
        ga = GA(func=fitness_func, n_dim=feature_dim, size_pop=population_size, max_iter=generations, lb=lb, ub=ub)
        best_x, best_y = ga.run()
        print(f'best_value: {best_y}')

        return np.array(ga.X[-num_candidate:])

    ### currently somehow unstable, very easy to end up on boundary
    def particle_swarm_sampling(self, model, feature_dim, population_size=50, iterations=20, num_candidate=10, candidate_list=None):
        if population_size * iterations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        if candidate_list is None:
            if isinstance(self.scaler, MinMaxScaler):
                lb, ub = [0] * feature_dim, [1] * feature_dim
            elif isinstance(self.scaler, StandardScaler):
                lb, ub = [-3] * feature_dim, [3] * feature_dim  # Assuming 3 standard deviations as bounds
            else:
                raise ValueError("Unsupported scaler type")
        else:
            lb, ub = np.min(candidate_list, axis=0), np.max(candidate_list, axis=0)

        def fitness_func(x):
            return -model.predict(np.array(x).reshape(1, -1))

        pso = PSO(func=fitness_func, n_dim=feature_dim, pop=population_size, max_iter=iterations, lb=lb, ub=ub)
        best_x, best_y = pso.run()
        print(f'best_value: {best_y}')

        return np.array(pso.X[-num_candidate:])

    ## SA cannot control boundary, modification needed
    def simulated_annealing_sampling(self, model, feature_dim, iterations=1000, num_candidate=10, candidate_list=None):
        
        # iterations = (num_candidate//(2*iterations)+1)*iterations if iterations<num_candidate//2 else iterations
        # SA_iter_nums = num_candidate//70+1
        iterations = (num_candidate//(2*iterations)+1)*iterations if iterations<num_candidate//2 else iterations
        SA_iter_nums = 1

        samples = []
        for SA_iter in range(SA_iter_nums):
            if candidate_list is None:
                if isinstance(self.scaler, MinMaxScaler):
                    x0 = np.clip(np.random.rand(feature_dim),0,1)
                elif isinstance(self.scaler, StandardScaler):
                    x0 = np.random.randn(feature_dim)
                else:
                    raise ValueError("Unsupported scaler type")
            else:
                x0 = np.mean(candidate_list, axis=0)
    
            def fitness_func(x):
                return -model.predict(np.array(x).reshape(1, -1)).item()

            def neighbor_func(x):
                step = np.random.normal(0, 1, size=feature_dim) * 0.1
                new_sample = x + step

                if isinstance(self.scaler, MinMaxScaler):
                    # Apply reflection to keep new_sample within bounds
                    new_sample = np.where(new_sample < 0, -new_sample, new_sample)
                    new_sample = np.where(new_sample > 1, 2-new_sample, new_sample)
                elif isinstance(self.scaler, StandardScaler):
                    new_sample = np.where(new_sample < -3, -6-new_sample, new_sample)
                    new_sample = np.where(new_sample > 3, 6-new_sample, new_sample)                   
            
                return new_sample

            # def neighbor_func(x):
            #     step = np.random.normal(0, 1, size=feature_dim) * 0.1
            #     new_sample = x + step
            #     new_sample = np.clip(new_sample, 0, 1)  # Ensure the new sample is within the bounds [0, 1]
            #     return new_sample

            sa = SA(func=fitness_func, x0=x0, T_max=100, T_min=1e-9, L=iterations)
            sa.neighbor = neighbor_func  # Use custom neighbor function to ensure bounds
            best_x, best_y = sa.run()
            print(f'best_value: {best_y}')

            samples.append(np.array(sa.best_x_history))

        Samples = np.vstack(samples)

        return Samples[-num_candidate:]

    #蚁群算法用于寻找问题的最优路径，它主要应用于路径优化和组合优化问题。对于当前版本采样代码只是将方程写出使其可以运行，但并未针对采样问题进行特殊配置，现阶段其并不适用于采样
    def ant_colony_sampling(self, model, feature_dim, n_ants=50, n_best=5, n_iterations=100, num_candidate=10, candidate_list=None):
        if n_ants * n_iterations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        if candidate_list is None:
            if isinstance(self.scaler, MinMaxScaler):
                lb, ub = [0] * feature_dim, [1] * feature_dim
            elif isinstance(self.scaler, StandardScaler):
                lb, ub = [-3] * feature_dim, [3] * feature_dim  # Assuming 3 standard deviations as bounds
            else:
                raise ValueError("Unsupported scaler type")
        else:
            lb, ub = np.min(candidate_list, axis=0), np.max(candidate_list, axis=0)

        def fitness_func(individual):
            return -model.predict(np.array(individual).reshape(1, -1)).item()

        aca = ACA_TSP(func=fitness_func, n_dim=feature_dim, size_pop=n_ants, max_iter=n_iterations, distance_matrix=np.random.rand(feature_dim, feature_dim))
        best_x, best_y = aca.run()
        print(f'best_value: {best_y}')

        return np.array(aca.X[-num_candidate:])

    def differential_evolution_sampling(self, model, feature_dim, population_size=50, generations=20, num_candidate=10, candidate_list=None):
        if population_size * generations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        if candidate_list is None:
            if isinstance(self.scaler, MinMaxScaler):
                lb, ub = [0] * feature_dim, [1] * feature_dim
            elif isinstance(self.scaler, StandardScaler):
                lb, ub = [-3] * feature_dim, [3] * feature_dim  # Assuming 3 standard deviations as bounds
            else:
                raise ValueError("Unsupported scaler type")
        else:
            lb, ub = np.min(candidate_list, axis=0), np.max(candidate_list, axis=0)

        def fitness_func(x):
            return -ray.get(model.predict(np.array(x).reshape(1, -1))).item()

        de = DE(func=fitness_func, n_dim=feature_dim, size_pop=population_size, max_iter=generations, lb=lb, ub=ub)
        best_x, best_y = de.run()
        print(f'best_value: {best_y}')

        return np.array(de.X[-num_candidate:])

    #免疫优化算法同样用于寻找问题的最优路径，它主要应用于路径优化和组合优化问题。对于当前版本采样代码只是将方程写出使其可以运行，但并未针对采样问题进行特殊配置，现阶段其并不适用于采样
    def immune_algorithm_sampling(self, model, feature_dim, population_size=50, generations=20, num_candidate=10, candidate_list=None):
        if population_size * generations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        if candidate_list is None:
            if isinstance(self.scaler, MinMaxScaler):
                lb, ub = [0] * feature_dim, [1] * feature_dim
            elif isinstance(self.scaler, StandardScaler):
                lb, ub = [-3] * feature_dim, [3] * feature_dim  # Assuming 3 standard deviations as bounds
            else:
                raise ValueError("Unsupported scaler type")
        else:
            lb, ub = np.min(candidate_list, axis=0), np.max(candidate_list, axis=0)

        def fitness_func(individual):
            return -model.predict(np.array(individual).reshape(1, -1)).item()

        ia = IA_TSP(func=fitness_func, n_dim=feature_dim, size_pop=population_size, max_iter=generations, prob_mut=0.2, T=0.7, alpha=0.95)
        best_x, best_y = ia.run()
        print(f'best_value: {best_y}')

        return np.array(ia.X[-num_candidate:])

    # 人工鱼群算法,其表现需要进一步测试
    def artificial_fish_swarm_sampling(self, model, feature_dim, population_size=50, iterations=20, num_candidate=10, candidate_list=None):
        if population_size * iterations < num_candidate:
            raise ValueError("Total sampling points must be greater than num_candidate")

        if candidate_list is None:
            if isinstance(self.scaler, MinMaxScaler):
                lb, ub = [0] * feature_dim, [1] * feature_dim
            elif isinstance(self.scaler, StandardScaler):
                lb, ub = [-3] * feature_dim, [3] * feature_dim  # Assuming 3 standard deviations as bounds
            else:
                raise ValueError("Unsupported scaler type")
        else:
            lb, ub = np.min(candidate_list, axis=0), np.max(candidate_list, axis=0)

        def fitness_func(x):
            return -model.predict(np.array(x).reshape(1, -1)).item()

        afsa = AFSA(func=fitness_func, n_dim=feature_dim, size_pop=population_size, max_iter=iterations, max_try_num=100, step=0.5, visual=0.3, q=0.98, delta=0.5)
        best_x, best_y = afsa.run()
        print(f'best_value: {best_y}')

        return np.array(afsa.X[-num_candidate:])

    # Suggest using monte_carlo sampling
    ## the robustness of [gaussian, monte_carlo] has been tested; [GA, PSO, DE, AFS] are functional, but slow; SA can not control the boundary; [ACA, IA] are used for path optimisation
    ### highly suggest the developeer hack the skopt code, replacing mulltiprocess in sko with Ray Actor for parallelisation. (since the multiuprocess may conflict with ray in resources and process management)

    ### Serial candidate generation
    def generate_candidates(self, method, model, feature_dim, num_candidate=100, n_samples=1000, iterations=50, candidate_list=None):
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


    ### the robustness of [gaussian, monte_carlo] has been tested; [GA, PSO, DE, AFS] are functional, but slow; SA can not control the boundary; [ACA, IA] are used for path optimisation

    ### ray function for parallel candidate generation
    @ray.remote
    def generate_candidates_ray(self, method, model, feature_dim, num_candidate=100, n_samples=1000, iterations=50, candidate_list=None):
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

    ### Parallel candidate generation
    def generate_candidate_parallel(self, method, feature_dim, model_results, model_list, num_candidate=100, n_samples=1000, iterations=50, candidate_list=None, all_surrogate_sampling=False, parallel=True):

        candidate_tasks = []
        candidates = []
        for target_i in model_results.keys():
            for sg_model in model_list:
                models = model_results[target_i][sg_model]['models']
                model_errors = model_results[target_i][sg_model]['errors']

                if parallel:
                    if all_surrogate_sampling:
                        for model in models:
                            candidate_tasks.append(self.generate_candidates_ray.remote(self, method=method, model=model, feature_dim=feature_dim, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations))
                    else:
                        ensemble_model = AbstractSurrogateModel(sg_model, models)
                        candidate_tasks.append(self.generate_candidates_ray.remote(self, method=method, model=ensemble_model, feature_dim=feature_dim, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations))
                else:
                    if all_surrogate_sampling:
                        for model in models:
                            candidate = self.generate_candidates(method=method, model=model, feature_dim=feature_dim, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations)
                            candidates.append(candidate)
                    else:
                        ensemble_model = AbstractSurrogateModel(sg_model, models)
                        candidate = self.generate_candidates(method=method, model=ensemble_model, feature_dim=feature_dim, num_candidate=num_candidate, n_samples=n_samples, iterations=iterations)
                        candidates.append(candidate)

        if parallel:
            candidate_X_scaled = ray.get(candidate_tasks)  
        else:
            candidate_X_scaled = candidates
            
        candidate_X_scaled = np.vstack(candidate_X_scaled)
        
        return candidate_X_scaled

    

# 示例用法
# if __name__ == "__main__":
#     class DummyModel(BaseEstimator, RegressorMixin):
#         def fit(self, X, y):
#             pass
    
#         def predict(self, X):
#             return np.sum(X, axis=1)
    
#     # 创建一个样本数据集
#     X = np.random.rand(100, 10)
#     y = np.sum(X, axis=1)
    
#     scaler = StandardScaler().fit(X)  # 假设这里使用的是StandardScaler
#     model = DummyModel()
#     model.fit(X, y)
    
#     sampler = Sampler(scaler)
#     feature_dim = X.shape[1]
#     candidates = sampler.generate_candidates('genetic_algorithm', model, feature_dim, n_samples=100, num_candidate=10)
#     print(candidates)
