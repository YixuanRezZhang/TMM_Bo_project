# TMM Bayesian Optimization Usage

This project exposes three ready-to-run workflows and one parameterized command-line workflow through `main.py`.

Detailed workflow and function references are available in [code_reference_en.md](code_reference_en.md) and [code_reference_zh.md](code_reference_zh.md).

## Environment

Activate the project environment before running workflows:

```bash
# If your shell is not initialized for conda, first run your local conda shell hook.
conda activate Bo_project
```

Use your local conda shell hook first if `conda activate` is not available in the current shell.

## One-command workflows

Run the synthetic benchmark workflow:

```bash
python main.py math
```

Run the generic closed-pool workflow. It reads the first CSV file in the current directory and uses columns whose names start with `target_` as targets:

```bash
python main.py data-closedpool
```

Run the closed-pool workflow with the target-window constraints encoded in `data_test_closedpool_targetwindow.py`:

```bash
python main.py data-target-window
```

Preview any of these commands without starting the heavy optimization job:

```bash
python main.py math --dry-run
python main.py data-closedpool --dry-run
python main.py data-target-window --dry-run
```

## Parameterized BO command

Use `bo` when you want to call `BayesianOptimization` directly from external scripts or a shell command:

```bash
python main.py bo \
  --data-file Data/bm_data.csv \
  --targets bm_target \
  --model-names Lasso Ridge ElasticNet MLPRegressor LightGBM XGBoost \
  --optimization-goal maximize \
  --scaler-method minmax \
  --sampling-method differential_evolution \
  --num-candidate 10000 \
  --n-samples 100 \
  --iterations 500 \
  --batch-size 10
```

For a closed-pool test, add `--close-pool --close-pooling-test` and tune `--n-iter`, `--n-bootstrap-sample-nums`, and `--batch-size`.

## Equation Notes

These notes summarize the main equations used in the code comments and helper routines.

### Gaussian log density

For prediction target `y`, predictive mean `mu`, standard deviation `sigma`, and standardized residual `z = (y - mu) / sigma`, the Gaussian log density is:

```text
log p(y | mu, sigma) = -0.5 * (log(2*pi) + 2*log(sigma) + z^2)
```

This is used for expected log predictive density (ELPD). Larger ELPD means the probabilistic prediction assigns higher probability to the observed value.

### Gaussian CRPS

For a Gaussian predictive distribution, the continuous ranked probability score has the closed form:

```text
CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
```

`Phi` is the standard normal CDF and `phi` is the PDF. Lower CRPS means a sharper and better calibrated predictive distribution.

### Student-t predictive scale

For a Student-t distribution with degrees of freedom `nu`, location `mu`, and scale `s`:

```text
Var = s^2 * nu / (nu - 2)
s = sigma * sqrt((nu - 2) / nu)
```

The code maps a target standard deviation `sigma` to the Student-t scale `s` so the Student-t predictive variance matches `sigma^2`.

### Student-t CRPS by Monte Carlo

The Student-t CRPS is estimated as:

```text
CRPS = E|X - y| - 0.5 * E|X - X'|
```

`X` and `X'` are independent samples from the predictive Student-t distribution. This avoids needing a closed form and works with heavier-tailed uncertainty.

### Safe coefficient of variation

The safe CV is:

```text
CV = std / (abs(mean) + tau)
```

The buffer `tau` prevents division by values close to zero. This keeps the score stable when the mean response is near zero.

### Structural score from low-frequency trend

The structural score estimates whether a candidate-response surface is smooth and globally coherent. It combines:

```text
S_low   = low-frequency wavelet energy ratio
S_curv  = 1 / (1 + kappa)
SF      = geometric_mean(power) / arithmetic_mean(power)
rho     = correlation between two pathwise trend reconstructions
```

`S_curv` compresses the curvature-to-gradient ratio `kappa`; lower curvature relative to gradient gives a larger score. `SF` is spectral flatness of residuals after trend removal; larger values indicate residuals closer to white noise. The final score uses a geometric-style aggregation so one weak component can reduce the overall structural confidence.

### Disagreement and entropy scores

Model disagreement is normalized with either Gini-Simpson diversity or Shannon entropy:

```text
Gini-Simpson = (1 - sum(p^2)) / (1 - 1/M)
Entropy      = H(p) / log(M)
```

`M` is the number of models. Both scores are scaled to `[0, 1]`; larger values mean stronger disagreement and therefore more exploration pressure.

### Kalman-style acquisition blending

The code uses a normalized Kalman gain:

```text
K = P_pred_ens / (P_pred_ens + R_eff)
beta = clip(1 - K, 0, 1)
```

`P_pred_ens` represents ensemble predictive spread and `R_eff` represents effective noise or reliability. When ensemble uncertainty dominates, `K` grows; when model consensus is high, `beta` increases the trust in consensus-oriented correction and reduces purely acquisition-driven exploration.

### Improvement, PI, and SNR

For best observed value `y_star`, predictive mean `mu`, standard deviation `sigma`, and exploration weight `kappa`, the margin is conceptually:

```text
margin = mu + kappa * sigma - y_star
```

Positive margin means a candidate can improve over the current best under the UCB-style prediction. The code softens this margin with a softplus hinge to avoid large zero-valued regions, then aggregates signal-to-noise style scores over peak-top and peak-shoulder regions.
