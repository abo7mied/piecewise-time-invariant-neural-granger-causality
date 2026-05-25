"""Run the repeated nonlinear-VAR boundary-localization benchmark."""
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import csv
from pathlib import Path

from src.core import (
    run_full_experiment,
    summarize_methods,
    compare_ours_to_baselines,
    compute_best_method_proportions,
    save_rows_to_csv,
    print_rows,
    plot_error_boxplot,
    plot_best_method_proportions,
    compute_best_baseline_per_replicate,
    plot_ours_vs_best_baseline_error_curve,
)


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--T", type=int, default=12000)
    args = parser.parse_args()

    results_dir = args.results_dir
    fig_dir = results_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = run_full_experiment(
        n_replicates=args.n_replicates,
        base_seed=args.seed,
        T=args.T,
        save_every=None,
    )
    write_rows(rows, results_dir / "boundary_replicates.csv")

    summary = summarize_methods(rows)
    comparison = compare_ours_to_baselines(rows)
    best_props = compute_best_method_proportions(rows)

    write_rows(summary, results_dir / "boundary_summary.csv")
    write_rows(comparison, results_dir / "boundary_comparison.csv")
    write_rows(best_props, results_dir / "boundary_best_method_proportions.csv")

    fig, ax = plot_error_boxplot(rows)
    fig.savefig(fig_dir / "boundary_error_boxplot.png", dpi=300, bbox_inches="tight")

    fig, ax, _ = plot_best_method_proportions(rows)
    fig.savefig(fig_dir / "best_method_proportions.png", dpi=300, bbox_inches="tight")

    fig, ax, best_baseline = plot_ours_vs_best_baseline_error_curve(rows)
    fig.savefig(fig_dir / "ours_vs_best_baseline.png", dpi=300, bbox_inches="tight")

    print("Summary:")
    print_rows(summary)
    print("\nComparison:")
    print_rows(comparison)


if __name__ == "__main__":
    main()
