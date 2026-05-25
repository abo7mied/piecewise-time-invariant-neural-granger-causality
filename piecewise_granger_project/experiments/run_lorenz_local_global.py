"""Run local-vs-global Granger graph recovery on the two-regime Lorenz data."""
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.run_lorenz_boundary import make_lorenz_graphs
from src.core import (
    simulate_two_regime_generalized_lorenz,
    fit_local_var_granger_after_boundary,
    fit_local_cmlp_granger_after_boundary,
    fit_global_var_granger_baseline,
    fit_global_cmlp_granger_baseline,
    compare_global_vs_local_graphs,
    compute_local_cross_regime_boundary_score_normalized,
    graph_metrics_source_target,
    normalized_shd_directed,
)


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_adj(ax, A, title):
    im = ax.imshow(A, vmin=0, vmax=1, aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    d = A.shape[0]
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.set_xticklabels([f"x{i}" for i in range(d)], fontsize=7)
    ax.set_yticklabels([f"x{i}" for i in range(d)], fontsize=7)
    return im


def summarize_repeated_rows(rows):
    summary = {
        "model": "cMLP repeated training",
        "n_train_runs": len(rows),
    }

    keys = [
        "global_mean_f1",
        "local_mean_f1",
        "global_mean_nshd",
        "local_mean_nshd",
    ]

    for key in keys:
        vals = np.array([float(row[key]) for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau_hat", type=int, default=4000)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=10)

    # New: repeat only stochastic cMLP training.
    parser.add_argument("--n_train_runs", type=int, default=10)

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
        pairwise_scale=0.02,
        noise_std=0.05,
        method="rk4",
        standardize=True,
        param_seed_1=42,
        param_seed_2=43,
        initial_seed=123 + args.seed,
        noise_seed=999 + args.seed,
    )
    X = result["X"]
    A1_true = result["A1"].T
    A2_true = result["A2"].T

    true_tau = int(result["tau"])

    if args.tau_hat is None:
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
    else:
        tau_hat = int(args.tau_hat)

    print({
        "true_tau": true_tau,
        "tau_hat": tau_hat,
        "abs_error": abs(tau_hat - true_tau),
        "norm_abs_error": abs(tau_hat - true_tau) / X.shape[0],
    })

    # VAR/Ridge is deterministic, so run it once.
    local_var = fit_local_var_granger_after_boundary(
        X, tau_hat, p=6, alpha=1.0, top_k_per_target=2,
        standardize_segment=False, min_segment_length=500,
    )

    global_var = fit_global_var_granger_baseline(
        X, p=6, alpha=1.0, top_k_per_target=2, standardize_segment=False,
    )

    var_compare = compare_global_vs_local_graphs(
        global_var,
        local_var["var_left"]["A_hat"],
        local_var["var_right"]["A_hat"],
        A1_true,
        A2_true,
        name="VAR/Ridge",
    )

    # Repeat only stochastic cMLP training on the same X and tau_hat.
    cmlp_rows = []

    # Keep the last run for graph visualization.
    local_cmlp = None
    global_cmlp = None
    cmlp_compare = None

    for train_run in range(args.n_train_runs):
        print(f"\n================ cMLP training run {train_run + 1}/{args.n_train_runs} ================")

        # Only changes the random initialization/training randomness.
        np.random.seed(1000 + train_run)
        try:
            import torch
            torch.manual_seed(1000 + train_run)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1000 + train_run)
        except ImportError:
            pass

        local_cmlp = fit_local_cmlp_granger_after_boundary(
            X, tau_hat, p=6, hidden_dims=(32, 16), group_lambda=0.1,
            lr=5e-3, n_epochs=args.epochs, batch_size=64,
            standardize_segment=False, group_prune_epsilon=1e-4,
            top_k_per_target=2, epsilon=None, quantile=0.6,
            min_segment_length=500, verbose=True,
        )

        global_cmlp = fit_global_cmlp_granger_baseline(
            X, p=6, hidden_dims=(32, 16), group_lambda=0.1,
            lr=5e-3, n_epochs=args.epochs, batch_size=64,
            standardize_segment=False, group_prune_epsilon=1e-4,
            top_k_per_target=2, epsilon=None, quantile=0.6, verbose=True
        )

        cmlp_compare = compare_global_vs_local_graphs(
            global_cmlp,
            local_cmlp["cmlp_left"]["A_hat"],
            local_cmlp["cmlp_right"]["A_hat"],
            A1_true,
            A2_true,
            name=f"cMLP_train_run_{train_run}",
        )

        row = dict(cmlp_compare["summary"])
        row["train_run"] = train_run
        row["train_seed"] = 1000 + train_run
        cmlp_rows.append(row)

        print(row)

    cmlp_summary = summarize_repeated_rows(cmlp_rows)

    # Save detailed repeated cMLP results.
    write_rows(cmlp_rows, results_dir / "lorenz_cmlp_training_repeats.csv")
    write_rows([cmlp_summary], results_dir / "lorenz_cmlp_training_repeats_summary.csv")

    # Save compact summary: VAR once, cMLP as repeated-training mean.
    rows = [
        var_compare["summary"],
        {
            "model": "cMLP repeated training mean",
            "global_mean_f1": cmlp_summary["global_mean_f1_mean"],
            "local_mean_f1": cmlp_summary["local_mean_f1_mean"],
            "global_mean_nshd": cmlp_summary["global_mean_nshd_mean"],
            "local_mean_nshd": cmlp_summary["local_mean_nshd_mean"],
        },
    ]
    write_rows(rows, results_dir / "lorenz_local_global_summary.csv")

    # Keep graph visualization from the final cMLP training run.
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.4), constrained_layout=True)
    plot_adj(axes[0], A1_true, "True R1")
    plot_adj(axes[1], A2_true, "True R2")
    plot_adj(axes[2], local_cmlp["cmlp_left"]["A_hat"], "Local cMLP R1")
    plot_adj(axes[3], local_cmlp["cmlp_right"]["A_hat"], "Local cMLP R2")
    plot_adj(axes[4], global_cmlp["A_hat"], "Global cMLP")
    fig.savefig(fig_dir / "lorenz_cmlp_graphs.png", dpi=300, bbox_inches="tight")

    # Boxplot instead of bar plot.
    local_cmlp_f1 = [row["local_mean_f1"] for row in cmlp_rows]
    global_cmlp_f1 = [row["global_mean_f1"] for row in cmlp_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(
        [local_cmlp_f1, global_cmlp_f1],
        labels=["Local cMLP", "Global cMLP"],
        showmeans=True,
    )

    ax.axhline(
        var_compare["summary"]["local_mean_f1"],
        linestyle="--",
        linewidth=1,
        label="Local VAR/Ridge",
    )
    ax.axhline(
        var_compare["summary"]["global_mean_f1"],
        linestyle=":",
        linewidth=1,
        label="Global VAR/Ridge",
    )

    ax.set_ylabel("Mean F1")
    ax.set_title("Repeated cMLP Training on Fixed Lorenz Dataset")
    ax.legend()
    fig.savefig(fig_dir / "lorenz_local_vs_global_f1_boxplot.png", dpi=300, bbox_inches="tight")

    # Optional nSHD boxplot, useful for the report.
    local_cmlp_nshd = [row["local_mean_nshd"] for row in cmlp_rows]
    global_cmlp_nshd = [row["global_mean_nshd"] for row in cmlp_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(
        [local_cmlp_nshd, global_cmlp_nshd],
        labels=["Local cMLP", "Global cMLP"],
        showmeans=True,
    )

    ax.axhline(
        var_compare["summary"]["local_mean_nshd"],
        linestyle="--",
        linewidth=1,
        label="Local VAR/Ridge",
    )
    ax.axhline(
        var_compare["summary"]["global_mean_nshd"],
        linestyle=":",
        linewidth=1,
        label="Global VAR/Ridge",
    )

    ax.set_ylabel("Mean nSHD")
    ax.set_title("Repeated cMLP Training on Fixed Lorenz Dataset")
    ax.legend()
    fig.savefig(fig_dir / "lorenz_local_vs_global_nshd_boxplot.png", dpi=300, bbox_inches="tight")

    print("VAR summary:")
    print(var_compare["summary"])

    print("\ncMLP repeated-training summary:")
    print(cmlp_summary)


if __name__ == "__main__":
    main()