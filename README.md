# TMM_Bo_project

A modular framework for Bayesian optimization with various sampling and acquisition functions.

## Installation

Clone the repository and navigate to the project directory:

install via:
- pip: 'pip install -r requirements.txt' or 'pip install -e .'
- conda: 'conda env create -f environment.yml'



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

## To Do list

- ~~multi-objective,~~ multi-task and multi-fidelity implementation
- material compositions-to-descriptors processing
- ~~automatized characterization (parameter fitting function)~~
- ~~Ray parallel processing~~
- Universal ParamsFittingIO and ParamsFittingExecuteModule for parameter fitting
- state tracking to use historical information
- Testing on different closed-pool tasks:
    - ~~formation energy~~
    - ~~band gap~~
    - ~~magnetization~~
    - ~~poisson ratio~~
    - ~~multiobjective: (fE, bg, mag)~~
    - ...
- Testing on different parameter-fitting tasks:
    - RIXS
    - HRTEM
