import numpy as np
import os, pickle, torch, ray
from scipy.stats import norm
import pygmo as pg

### 所有内部优化问题的初始设置都是最大化

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

@ray.remote
def model_predict(model, X_candidate, sg_model):
    if sg_model == 'GaussianProcess':
        Gmodel_res = model.model.posterior(torch.tensor(X_candidate, dtype=torch.float32))
        Gmean = Gmodel_res.mean.detach().cpu().numpy().reshape(-1)
        Gstd = torch.sqrt(Gmodel_res.variance).detach().cpu().numpy().reshape(-1)
        return Gmean, Gstd
    else:
        preds = model.predict(X_candidate)
        return preds, None

class AcquisitionFunction:
    def __init__(self, hpar=0.1):
        self.hpar = hpar
        
    def ucb(self, mean, std):
        return mean + self.hpar * std

    def ei(self, mean, std, y_best):
        with np.errstate(divide='warn'):
            imp = mean - y_best - self.hpar
            Z = imp / std
            EI = imp * norm.cdf(Z) + std * norm.pdf(Z)
            EI[std == 0.0] = 0.0
        return EI

    def pi(self, mean, std, y_best):
        with np.errstate(divide='warn'):
            imp = mean - y_best - self.hpar
            Z = imp / std
            PI = norm.cdf(Z)
            PI[std == 0.0] = 0.0
        return PI

    # a contribution of hypervolume and Euclidean distance
    def hypervolume(self, points, reference_point=None):
        points = -points
        hv = pg.hypervolume(points)
        if reference_point is None:
            reference_point = hv.refpoint()+1
        else:
            reference_point = -reference_point

        contri_rank = hv.contributions(reference_point)
        distances = np.linalg.norm(points - reference_point, axis=1)
        
        return contri_rank + distances

    def select_next(self, method, X_candidate, model_name_list, num_target, model_path, batch_size=10, y_best=None, model_result=None, stack=False):

        all_acq_vaules = []
               
        # Load models
        for target_i in range(num_target):
            
            if stack:
                if model_result is None:
                    stack_file_path = f"{model_path}/stacking_results_{target_i}.pkl"
                    with open(stack_file_path, 'rb') as f:
                        data = pickle.load(f)
                    stacking_model = data['stacking_model']
                    stacking_score = data['stacking_error']
                else:
                    stacking_model = model_result[target_i]['stacking_model']
                    stacking_score = model_result[target_i]['stacking_error']
                    
            acq_values = np.zeros(X_candidate.shape[0])
            for sg_model in model_name_list:

                if model_result is None:
                    print(f'load models {sg_model}_{target_i}')
                    file_path = f"{model_path}/{sg_model}_{target_i}_bootstrap.pkl"
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    # model_name = data['model_name']
                    # optimized_params = data['optimized_params']
                    models = data['models']
                    model_errors = data['errors']
                else:
                    models = model_result[target_i][sg_model]['models']
                    model_errors = model_result[target_i][sg_model]['errors']

                # design of score can be discussed. 
                ### TBD:***maybe using a model ensemble score to direct represent score, thus a ensemble fitting should be added in evaluation.py***
                if stack:
                    model_score = stacking_score[sg_model]
                else:
                    score_mu, score_std = np.mean(model_errors), np.std(model_errors)
                    model_score = np.clip(score_mu-0.01*score_std, 0, np.inf)
                    # model_score = np.clip(score_mu, 0, np.inf)

                tasks = []
                for model in models:
                    tasks.append(model_predict.remote(model, X_candidate, sg_model))

                # make prediction using all bootstrapping generated models to get mean and std
                results = ray.get(tasks)
                preds = []
                uncertains = []
                for res in results:
                    if sg_model == 'GaussianProcess':
                        preds.append(res[0])
                        uncertains.append(res[1])
                    else:
                        preds.append(res[0])

                preds = np.array(preds)
                if sg_model == 'GaussianProcess':
                    # print(uncertains)
                    uncertains = np.array(uncertains)
                    mean = preds.mean(axis=0).reshape(-1)
                    std = uncertains.mean(axis=0).reshape(-1)
                else:
                    mean = preds.mean(axis=0).reshape(-1)
                    std = preds.std(axis=0).reshape(-1)

                # *** TBD: the location of adding in the acq is flexible, 
                # *** one can either calculate the hypervolume or all targets of candidates first, then calculate the acq of hypervolume value
                # *** or calculate the acq of each terget first, then calculate the hypervolume of all acqs
                # *** the differences between two approach in my mind are the first approach are more robust in find the candidate behaves better in all target, while the second one are more sensitive to the ones which maybe more outstanding in single target
                # *** here we choose the second approach since it is logically easy to implement
                
                if method == 'ucb':
                    acq_value = self.ucb(mean, std)
                elif method == 'ei' or method == 'pi':
                    if y_best is None:
                        raise ValueError(f"Unknown current best target value y_best, add current best target value if you want to use EI or PI")
                    if method == 'ei':
                        acq_value = self.ei(mean, std, y_best)
                    elif method == 'pi':
                        acq_value = self.pi(mean, std, y_best)
                else:
                    raise ValueError(f"Unknown acquisition method: {method}")

                acq_values += acq_value*model_score

            if stack:
                acq_values += stacking_model.predict(X_candidate)
                
            all_acq_vaules.append(acq_values)
                  
        all_acq_vaules = np.array(all_acq_vaules)
        
        if all_acq_vaules.shape[0]>1:
            sort_result = self.hypervolume(all_acq_vaules.T)
        else:
            sort_result = all_acq_vaules.reshape(-1)
        
        next_indexes = np.argsort(sort_result)[::-1][:batch_size]
        
        return next_indexes


