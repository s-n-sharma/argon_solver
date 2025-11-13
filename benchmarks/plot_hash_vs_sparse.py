from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = {
    "base_rows",
    "num_cols",
    "scenario",
    "solver",
    "mean_time_seconds",
    "mean_residual",
    "conflict_rate",
    "correct_rate",
    "detection_rate",
}

MODE_ORDER = ["cached_hash_qr", "sparse_qr"]

FIGURE_DIR = Path(__file__).parent / "hash_vs_sparse"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["solver"] = pd.Categorical(df["solver"], categories=MODE_ORDER, ordered=True)
    df["mean_time_ms"] = df["mean_time_seconds"] * 1e3
    df["scenario"] = df["scenario"].astype(str)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["scenario", "solver"], as_index=False)
        .agg(
            mean_time_ms=("mean_time_ms", "mean"),
            mean_residual=("mean_residual", "mean"),
            conflict_rate=("conflict_rate", "mean"),
            correct_rate=("correct_rate", "mean"),
            detection_rate=("detection_rate", "mean"),
        )
    )
    return summary.sort_values(["scenario", "solver"])


def plot_time_vs_size(df: pd.DataFrame) -> None:
    for scenario, subset in df.groupby("scenario", observed=True):
        if subset.empty:
            continue
        subset = subset.sort_values(["base_rows", "solver"])
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        sns.lineplot(
            data=subset,
            x="base_rows",
            y="mean_time_ms",
            hue="solver",
            hue_order=MODE_ORDER,
            marker="o",
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Constraints (m)")
        ax.set_ylabel("Mean update time (ms)")
        ax.set_title(f"Update time vs. size — {scenario}")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.legend(title="Solver", fontsize="small", title_fontsize="small", framealpha=0.9)
        fig.tight_layout()
        out_path = FIGURE_DIR / f"time_vs_size_{scenario}.png"
        fig.savefig(out_path, dpi=250)
        plt.close(fig)


def plot_correct_rate(df: pd.DataFrame) -> None:
    for scenario, subset in df.groupby("scenario", observed=True):
        if subset.empty:
            continue
        subset = subset.sort_values(["base_rows", "solver"])
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        sns.lineplot(
            data=subset,
            x="base_rows",
            y="correct_rate",
            hue="solver",
            hue_order=MODE_ORDER,
            marker="o",
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Constraints (m)")
        ax.set_ylabel("Agreement with sparse baseline")
        ax.set_title(f"Conflict-detection agreement — {scenario}")
        ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.legend(title="Solver", fontsize="small", title_fontsize="small", framealpha=0.9)
        fig.tight_layout()
        out_path = FIGURE_DIR / f"correct_rate_{scenario}.png"
        fig.savefig(out_path, dpi=250)
        plt.close(fig)


def plot_conflict_rate(df: pd.DataFrame) -> None:
    for scenario, subset in df.groupby("scenario", observed=True):
        if subset.empty:
            continue
        subset = subset.sort_values(["base_rows", "solver"])
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        sns.lineplot(
            data=subset,
            x="base_rows",
            y="conflict_rate",
            hue="solver",
            hue_order=MODE_ORDER,
            marker="o",
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Constraints (m)")
        ax.set_ylabel("Conflict detection rate")
        ax.set_title(f"Fraction of updates flagged conflicting — {scenario}")
        ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.legend(title="Solver", fontsize="small", title_fontsize="small", framealpha=0.9)
        fig.tight_layout()
        out_path = FIGURE_DIR / f"conflict_rate_{scenario}.png"
        fig.savefig(out_path, dpi=250)
        plt.close(fig)


def main(csv_path: Path | None = None) -> None:
    if csv_path is None:
        if len(sys.argv) > 1:
            csv_path = Path(sys.argv[1])
        else:
            csv_path = Path(__file__).parent / "hash_vs_sparse_comparison.csv"

    df = load_results(csv_path)
    summary = summarize(df)

    sns.set_theme(style="whitegrid", context="talk")

    print("Scenario summary:")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        formatters = {
            "mean_time_ms": "{:.3f}".format,
            "mean_residual": "{:.2e}".format,
            "conflict_rate": "{:.3f}".format,
            "correct_rate": "{:.3f}".format,
            "detection_rate": "{:.3f}".format,
        }
        print(summary.to_string(index=False, formatters=formatters))

    plot_time_vs_size(df)
    plot_correct_rate(df)
    plot_conflict_rate(df)

    print(f"Saved figures to {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
