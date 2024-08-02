import argparse
from src.bayesian_optimization import BayesianOptimization

def main():
    parser = argparse.ArgumentParser(description='Bayesian Optimization for Material Design')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file')
    parser.add_argument('--features', nargs='+', help='List of feature columns (optional)')
    parser.add_argument('--targets', nargs='+', required=True, help='List of target columns')
    parser.add_argument('--optimization_goal', type=str, choices=['maximize', 'minimize'], default='maximize', help='Optimization goal')
    parser.add_argument('--scaler_method', type=str, choices=['standard', 'minmax'], default='standard', help='Scaling method')
    parser.add_argument('--initial_samples', type=int, default=10, help='Number of initial samples for close pooling test')
    parser.add_argument('--close_pooling_test', type=bool, help='Whether close pooling test')
    parser.add_argument('--model_names', nargs='+', required=True, help='List of model names for evaluation')
    parser.add_argument('--sampling_method', type=str, default='genetic_algorithm', help='Sampling method for candidate generation')
    parser.add_argument('--num_candidate', type=int, default=10, help='Number of candidate samples to select')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples for sampling method')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations for sampling method')
    parser.add_argument('--n_bootstrap_sample_nums', type=int, default=20, help='Number of bootstrap samples for evaluation')
    parser.add_argument('--n_iter', type=int, default=100, help='Number of iterations for close pooling test')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for acquisition function')
    parser.add_argument('--reference_point', nargs='+', type=float, help='Reference point for hypervolume calculation')
    parser.add_argument('--multi_objective', action='store_true', help='Perform multi-objective optimization')

    args = parser.parse_args()

    bo = BayesianOptimization(data_file=args.data_file,
                              target_props=args.targets,
                              feature_props=args.features,
                              optimization_goal=args.optimization_goal,
                              scaler_method=args.scaler_method,
                              initial_samples=args.initial_samples,
                              threshold=args.threshold)

    if args.threshold:
        bo.close_pooling_test(model_names=args.model_names,
                              n_bootstrap_sample_nums=args.n_bootstrap_sample_nums,
                              n_iter=args.n_iter,
                              batch_size=args.batch_size)
    elif args.multi_objective:
        candidates = bo.optimize(model_names=args.model_names,
                                 sampling_method=args.sampling_method,
                                 num_candidate=args.num_candidate,
                                 n_samples=args.n_samples,
                                 iterations=args.iterations,
                                 n_bootstrap_sample_nums=args.n_bootstrap_sample_nums,
                                 reference_point=args.reference_point)
        print("Recommended candidate samples:")
        print(candidates)
    else:
        candidates = bo.optimize(model_names=args.model_names,
                                 sampling_method=args.sampling_method,
                                 num_candidate=args.num_candidate,
                                 n_samples=args.n_samples,
                                 iterations=args.iterations,
                                 n_bootstrap_sample_nums=args.n_bootstrap_sample_nums)

        print("Recommended candidate samples:")
        print(candidates)

if __name__ == "__main__":
    main()
