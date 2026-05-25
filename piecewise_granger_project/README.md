# Piecewise Time-Invariant Neural Granger Causality Detection

This project detects a regime boundary in a multivariate time series using normalized cross-regime prediction, then fits local Granger-causal graphs inside the estimated regimes.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Directory structure

```text
project/
|-- README.md
|-- requirements.txt
|-- data/
|-- src/
|-- experiments/
|-- results/
|-- notebooks/
```

The notebook is included only as a readable development artifact. The main reusable code is in `src/`, and the reproduction commands are in `experiments/`.

## Experiments: Reproduce main results

Run commands from the project root.

### 1. Boundary benchmark on nonlinear VAR

```bash
python experiments/run_boundary_replicates.py
```

Expected outputs:

```text
results/boundary_replicates.csv
results/boundary_summary.csv
results/boundary_comparison.csv
results/boundary_best_method_proportions.csv
results/figures/boundary_error_boxplot.png
results/figures/best_method_proportions.png
results/figures/ours_vs_best_baseline.png
```

Expected runtime:
About 10 minutes or less on a laptop Intel i9 CPU.

### 2. Boundary localization on two-regime generalized Lorenz

```bash
python experiments/run_lorenz_boundary.py
```

Expected outputs:

```text
results/lorenz_boundary_results.json
results/figures/lorenz_two_regime_series.png
results/figures/lorenz_boundary_score.png
```

Expected runtime:
About 2 minutes on a laptop Intel i9 CPU.

### 3. Local versus global Granger-causal recovery on Lorenz

```bash
python experiments/run_lorenz_local_global.py
```

Expected outputs:

```text
results/lorenz_local_global_summary.csv
results/figures/lorenz_cmlp_graphs.png
results/figures/lorenz_local_vs_global_f1.png
```

Expected runtime:
About 5 minutes on a laptop intel i9 CPU.
