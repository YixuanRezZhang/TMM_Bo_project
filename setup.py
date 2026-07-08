from setuptools import setup


setup(
    name="TMM_Bo_project",
    version="0.1.0",
    description="A modular framework for Bayesian optimization with adaptive sampling and acquisition functions.",
    url="https://github.com/YixuanRezZhang/TMM_Bo_project",
    license="Apache-2.0",
    packages=["src"],
    py_modules=["main"],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
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
