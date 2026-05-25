"""Run the two-regime generalized Lorenz boundary-localization example."""
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.core import (
    simulate_two_regime_generalized_lorenz,
    compute_local_cross_regime_boundary_score_normalized,
    plot_two_regime_result,
    plot_cross_regime_boundary_score,
)


def make_lorenz_graphs():
    """
    Hardcoded 6-variable source-target adjacency matrices.

    A[source, target] = 1 means source Granger-causes target.

    Each target has exactly one incoming edge in each regime.
    No diagonal self-edges.
    """
    A = np.array([
        [0, 1, 1, 0, 0, 0],  
        [0, 0, 1, 1, 0, 0],  
        [0, 0, 0, 1, 1, 0],  
        [0, 0, 0, 0, 1, 1],  
        [1, 0, 0, 0, 0, 1],  
        [1, 1, 0, 0, 0, 0],  
    ], dtype=int)

    B = np.array([
        [0, 0, 1, 1, 0, 0],  
        [0, 0, 0, 1, 1, 0],  
        [0, 0, 0, 0, 1, 1],  
        [1, 0, 0, 0, 0, 1],  
        [1, 1, 0, 0, 0, 0],  
        [0, 1, 1, 0, 0, 0], 
    ], dtype=int)

    return A, B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    results_dir = args.results_dir
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    A1, A2 = make_lorenz_graphs()
    result = simulate_two_regime_generalized_lorenz(
        A1=A1,
        A2=A2,
        T1=4000,
        T2=6000,
        dt=0.0005,
        burn_in=1000,
        F=12,
        damping=0.6,
        linear_scale=0.04,
        nonlinear_scale=0.18,
        pairwise_scale=0.01,
        noise_std=0.01,
        method="rk4",
        standardize=True,
        param_seed_1=42,
        param_seed_2=43,
        initial_seed=123 + args.seed,
        noise_seed=999 + args.seed,
    )
    X = result["X"]
    true_tau = int(result["tau"])

    score = compute_local_cross_regime_boundary_score_normalized(
        X=X,
        lag_order=20,
        h=200,
        train_window=1000,
        val_window=200,
        alpha=1.0,
        step=10,
        smooth_window=11,
        score_mode="log_ratio",
    )
    tau_hat = int(score["tau_hat"])

    fig, ax = plot_two_regime_result(result, title="Two-Regime Generalized Lorenz Time Series")
    fig.savefig(fig_dir / "lorenz_two_regime_series.png", dpi=300, bbox_inches="tight")

    fig, ax = plot_cross_regime_boundary_score(score, true_tau=true_tau,
                                                title="Normalized Local Cross-Regime Boundary Score")
    fig.savefig(fig_dir / "lorenz_boundary_score.png", dpi=300, bbox_inches="tight")

    out = {
        "true_tau": true_tau,
        "tau_hat": tau_hat,
        "abs_error": abs(tau_hat - true_tau),
        "norm_abs_error": abs(tau_hat - true_tau) / X.shape[0],
        "T": int(X.shape[0]),
        "d": int(X.shape[1]),
    }
    with open(results_dir / "lorenz_boundary_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(out)


if __name__ == "__main__":
    main()
