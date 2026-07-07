import argparse
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _add_script_runner(subparsers, name, script_name, help_text):
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the script that would be executed without running it.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments forwarded to the target script.",
    )
    parser.set_defaults(func=lambda args: _run_script(script_name, args.script_args, args.dry_run))


def _run_script(script_name, script_args, dry_run=False):
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find script: {script_path}")

    command = [sys.executable, str(script_path), *script_args]
    if dry_run:
        print("Would run:", " ".join(command))
        return 0

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *script_args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
    return 0


def _parse_optional_float_list(values):
    return None if values is None else [float(value) for value in values]


def _run_parameterized_bo(args):
    if args.dry_run:
        print("Would instantiate BayesianOptimization with:")
        print(f"  data_file={args.data_file}")
        print(f"  targets={args.targets}")
        print(f"  features={args.features}")
        print(f"  models={args.model_names}")
        print(f"  close_pool={args.close_pool}")
        print(f"  close_pooling_test={args.close_pooling_test}")
        return 0

    from src.bayesian_optimization import BayesianOptimization

    bo = BayesianOptimization(
        args.targets,
        args.data_file,
        feature_props=args.features,
        optimization_goal=args.optimization_goal,
        scaler_method=args.scaler_method,
        model_list=args.model_names,
        stacking=args.stacking,
        acq_method=args.acq_method,
        feature_lb=_parse_optional_float_list(args.feature_lb),
        feature_ub=_parse_optional_float_list(args.feature_ub),
        candidate_file=args.candidate_file,
        close_pool=args.close_pool,
        close_pool_initial_samples=args.close_pool_initial_samples,
        close_pool_threshold=args.close_pool_threshold,
        uni_hyperparameter=args.uni_hyperparameter,
    )

    if args.close_pooling_test:
        bo.close_pooling_test(
            n_bootstrap_sample_nums=args.n_bootstrap_sample_nums,
            n_iter=args.n_iter,
            batch_size=args.batch_size,
            hpar=args.hpar,
            save_all_info=args.save_all_info,
            sampling_method=args.sampling_method,
            num_candidate=args.num_candidate,
            n_samples=args.n_samples,
            iterations=args.iterations,
            candidate_sampling=args.candidate_sampling,
            diversity_method=args.diversity_method,
            use_data_correlation=args.use_data_correlation,
            use_model_correlation=args.use_model_correlation,
        )
        return 0

    samples_next, next_indexes = bo.optimize(
        batch_size=args.batch_size,
        n_bootstrap_sample_nums=args.n_bootstrap_sample_nums,
        sampling_method=args.sampling_method,
        num_candidate=args.num_candidate,
        n_samples=args.n_samples,
        iterations=args.iterations,
        hpar=args.hpar,
        if_train=not args.skip_train,
        candidate_sampling=args.candidate_sampling,
        n_random_models=args.n_random_models,
        seperate=args.seperate,
        diversity_method=args.diversity_method,
        use_data_correlation=args.use_data_correlation,
        use_model_correlation=args.use_model_correlation,
    )
    print("Recommended candidate samples:")
    print(samples_next)
    print("Recommended candidate indexes:")
    print(next_indexes)
    return 0


def _add_bo_parser(subparsers):
    parser = subparsers.add_parser(
        "bo",
        help="Run BayesianOptimization directly from command-line arguments.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate CLI arguments without importing BO dependencies.")
    parser.add_argument("--data-file", required=True, help="Path to the training data CSV file.")
    parser.add_argument("--features", nargs="+", help="Feature column names. Defaults to all non-target numeric columns.")
    parser.add_argument("--targets", nargs="+", required=True, help="Target column names.")
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["Lasso", "Ridge", "ElasticNet", "MLPRegressor", "LightGBM", "XGBoost"],
        help="Surrogate model names passed to BayesianOptimization.",
    )
    parser.add_argument("--optimization-goal", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--scaler-method", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--stacking", action="store_true", help="Enable model stacking.")
    parser.add_argument("--acq-method", default="ucb", help="Acquisition method name.")
    parser.add_argument("--candidate-file", help="Optional closed-pool candidate CSV file.")
    parser.add_argument("--feature-lb", nargs="+", help="Lower bounds for open-pool optimization.")
    parser.add_argument("--feature-ub", nargs="+", help="Upper bounds for open-pool optimization.")
    parser.add_argument("--close-pool", action="store_true", help="Use closed-pool candidate selection.")
    parser.add_argument("--close-pool-initial-samples", type=int, default=10)
    parser.add_argument("--close-pool-threshold", type=float)
    parser.add_argument("--uni-hyperparameter", action="store_true")
    parser.add_argument("--close-pooling-test", action="store_true", help="Run close_pooling_test instead of optimize.")
    parser.add_argument("--n-bootstrap-sample-nums", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--hpar", type=float, default=0.1)
    parser.add_argument("--save-all-info", action="store_true")
    parser.add_argument("--sampling-method", default="genetic_algorithm")
    parser.add_argument("--num-candidate", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--candidate-sampling", action="store_true")
    parser.add_argument("--n-random-models", type=int, default=2)
    parser.add_argument("--seperate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diversity-method", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-data-correlation", action="store_true")
    parser.add_argument("--use-model-correlation", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing model artifacts where supported.")
    parser.set_defaults(func=_run_parameterized_bo)


def build_parser():
    parser = argparse.ArgumentParser(
        description="One-command entry points for TMM Bayesian optimization workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_script_runner(subparsers, "math", "math_test.py", "Run the synthetic benchmark workflow.")
    _add_script_runner(subparsers, "data-closedpool", "data_test_closedpool.py", "Run the generic closed-pool data workflow.")
    _add_script_runner(
        subparsers,
        "data-target-window",
        "data_test_closedpool_targetwindow.py",
        "Run the closed-pool workflow with target-window selection.",
    )
    _add_bo_parser(subparsers)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
