# TMM_Bo_project

A modular framework for Bayesian optimization with ensemble surrogate models, adaptive sampling, acquisition functions, closed-pool tests, and parameter-fitting workflows.

## Installation

The recommended environment is based on the current tested `Bo_project` conda environment:

- Python 3.12
- CUDA 12.x runtime, tested with `cuda-version=12.2`
- RAPIDS 25.06, including `cudf`, `cuml`, and `cupy`
- PyTorch 2.5.1 CUDA 12.1 wheels
- Ray, BoTorch/GPyTorch, scikit-learn, PyGMO, FAISS, KAN/FastKAN, and model libraries used by `src/surrogate_model.py`

Create and activate the environment:

```bash
conda create -n Bo_project python=3.12 -y
conda activate Bo_project
```

Install RAPIDS and compiled conda dependencies:

```bash
conda install -c rapidsai -c conda-forge -c nvidia \
  rapids=25.06 python=3.12 cuda-version=12.2

conda install -c conda-forge -c pytorch \
  numpy pandas scipy scikit-learn matplotlib psutil pywavelets \
  pygmo hdbscan faiss-cpu optuna lightgbm
```

Install PyTorch and Python packages:

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -U "ray[data,train,tune,serve]"

pip install botorch gpytorch xgboost catboost scikit-opt pykan \
  scikit-dimension sliced igraph leidenalg umap-learn

pip install git+https://github.com/ZiyaoLi/fast-kan.git
```

Install this project in editable mode:

```bash
pip install -e .
```

For a single-file environment bootstrap, use:

```bash
conda env create -f conda_environment.yml
conda activate Bo_project
pip install -e .
```

## Dependency Comparison Notes

Compared with the older installation notes:

- PyTorch has moved from CUDA 11.8 wheels to the tested CUDA 12.1 wheels: `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, and `torchaudio==2.5.1+cu121`.
- RAPIDS has moved from `rapids=25.04` with CUDA 11.4-11.8 constraints to the tested `rapids=25.06` with `cuda-version=12.2`.
- `faiss-cpu` is sufficient for the current nearest-neighbor and clustering code paths. GPU acceleration for clustering is handled through RAPIDS/cuML when CUDA is available.
- `pykan` provides the `kan` module. `fast-kan` provides the `fastkan` module.
- `multi_task.py` and `multi_fidelity.py` are included as active modules but are still being updated and tested.

## Command-line Entry Points

After activating the project conda environment, use `main.py` for one-command workflows:

```bash
python main.py math
python main.py data-closedpool
python main.py data-target-window
```

Use `--dry-run` to validate routing without starting a long optimization job.

Documentation entry points:

- English: [docs/code_reference_en.md](docs/code_reference_en.md)
- 中文: [docs/code_reference_zh.md](docs/code_reference_zh.md)
- CLI and equation notes: [docs/usage.md](docs/usage.md)
