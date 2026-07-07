from setuptools import setup


setup(
    name="TMM_Bo_project",
    version="0.1.0",
    description="A modular framework for Bayesian optimization with adaptive sampling and acquisition functions.",
    url="https://github.com/YixuanRezZhang/TMM_Bo_project",
    packages=["src"],
    py_modules=["main"],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "psutil",
        "PyWavelets",
        "optuna",
        "xgboost",
        "lightgbm",
        "catboost",
        "torch",
        "botorch",
        "gpytorch",
        "pygmo",
        "scikit-opt",
        "ray[data,train,tune,serve]",
        "pykan",
        "scikit-dimension",
        "sliced",
        "igraph",
        "leidenalg",
        "umap-learn",
    ],
    entry_points={
        "console_scripts": [
            "tmm-bo=main:main",
        ],
    },
)
