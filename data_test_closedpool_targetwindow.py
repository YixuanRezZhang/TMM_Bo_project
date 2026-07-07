import os
import time
import numpy as np
import pandas as pd
from src.bayesian_optimization import BayesianOptimization

# -------------------------------
# 1. Run optimization with the project's BayesianOptimization class
# -------------------------------
# Use target columns from the CSV and a regression-model ensemble as base models.
csv_file = [i for i in os.listdir(".") if i.endswith('.csv')]
data = pd.read_csv(csv_file[0])
target = [i for i in data.columns if i.startswith('target_')]
print(target)

target_props = target
data_file = csv_file[0]
# model_list = ['Lasso', 'Ridge', 'ElasticNet', 'MLPRegressor', 'KNeighborsRegressor', 'LightGBM', 'ExtraTreesRegressor']#, 'DecisionTreeRegressor', 'ExtraTreesRegressor', 'XGBoost', 'RandomForest', 'AdaBoostRegressor']#, 'GP_cpu', 'KRR']#, 'ExtraTreesRegressor', , 'CatBoost'] 
# model_list = ['Lasso', 'Ridge', 'ElasticNet', 'KNeighborsRegressor', 'SVR', 'MLPRegressor', 'LightGBM', 'XGBoost', 'RandomForest', 'ExtraTreesRegressor', 'DecisionTreeRegressor']#, 'AdaBoostRegressor'],
# model_list = ['Lasso', 'Ridge', 'ElasticNet', 'RandomForest', 'LightGBM','XGBoost', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'ExtraTreesRegressor', 'MLPRegressor']
# model_list = ['Lasso', 'Ridge', 'ElasticNet', 'RandomForest', 'LightGBM','XGBoost', 'KNeighborsRegressor', 'DecisionTreeRegressor',  'MLPRegressor', 'ExtraTreesRegressor']
model_list = ['Lasso', 'Ridge', 'ElasticNet', 'MLPRegressor', 'LightGBM','XGBoost', 'KNeighborsRegressor', 'DecisionTreeRegressor']#, 'SVR MLPRegressor']

# Create the BayesianOptimization instance; tune these parameters for each task.
BO = BayesianOptimization(
    target_props, 
    data_file, 
    model_list=model_list, 
    scaler_method = 'standard',  #'standard', 'minmax'
    optimization_goal = 'maximize', #'minimize',   #'maximize'
    stacking=False, 
    acq_method='ucb', 
    candidate_file=None,
    # feature_lb = lb,
    # feature_ub = ub,
    close_pool=True,
    close_pool_initial_samples=20, 
    close_pool_threshold=None, 
    select_region={'target__alexandria_scan_magnetization':[19.50, 20.50], 'target__alexandria_scan_hull_distance':[-0.00, -0.00], 'target__alexandria_scan_band_gap':[1.45, 1.55]},
    uni_hyperparameter=True
)

BO.close_pooling_test(
    n_bootstrap_sample_nums=10, 
    n_iter=50, 
    batch_size=10, 
    hpar=0.1, 
    save_all_info=True, 
    sampling_method='genetic_algorithm', 
    candidate_sampling=False, 
    diversity_method=True,
    use_data_correlation=False,
    use_model_correlation=False
)

# samples_next, next_indexes = BO.optimize(
#     batch_size=batch_size,
#     n_bootstrap_sample_nums=20,
#     sampling_method='differential_evolution',
#     num_candidate=10000,
#     n_samples=100,
#     iterations=1000,
#     hpar=0.1,
#     if_train=True,
#     candidate_sampling=False,
#     n_random_models=2, 
#     seperate=True,
# ##     bs_rescale_method = 'hdbscan',  #'hdbscan', 'gmm' or 'kde'
#     diversity_method = 'gmm', #'auto', 'gmm', 'hdbscan', 'leiden', None
#     # alpha = 0.5 
# )

