from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = {
    "generator",
    "mode",
    "base_rows",
    "num_cols",
    "update_index",
    "time_seconds",
}

FIGURE_DIR = Path(__file__).parent / "givens_plots"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load benchmark data and validate that required columns are present."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated updates to per-size timing statistics."""
    grouped = (
        df.groupby(["generator", "mode", "base_rows", "num_cols"], as_index=False)["time_seconds"]
        .agg(
            mean_time="mean",
            median_time="median",
            min_time="min",
            max_time="max",
            std_time="std",
        )
    )
    # Convert to milliseconds for readability
    grouped["time_ms"] = grouped["mean_time"] * 1e3
    grouped["time_us"] = grouped["mean_time"] * 1e6
    grouped["std_ms"] = grouped["std_time"].fillna(0.0) * 1e3
    return grouped


def _use_line_style(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)


def plot_timings(agg: pd.DataFrame) -> None:
    """Plot mean per-update timing vs. constraint count for each generator."""
    sns.set_theme(style="whitegrid", context="talk")
    order = _mode_sort(agg["mode"].unique())

    for generator, subset in agg.groupby("generator"):
        subset = subset.sort_values(["base_rows", "mode"])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=subset,
            x="base_rows",
            y="time_ms",
            hue="mode",
            hue_order=order,
            marker="o",
            ax=ax,
        )

        _use_line_style(ax)
        ax.set_title(f"Per-update time vs. base size — {generator}")
        ax.set_xlabel("Base constraints (m)")
        ax.set_ylabel("Mean time per update (ms)")
        ax.legend(
            title="Mode",
            loc="upper left",
            fontsize="small",
            title_fontsize="small",
            framealpha=0.9,
        )
        fig.tight_layout()

        out_path = FIGURE_DIR / f"timings_{generator}.png"
        fig.savefig(out_path, dpi=250)
        plt.close(fig)


def plot_relative_speed(agg: pd.DataFrame, baseline: str = "dense_scratch") -> None:
    """Plot timing ratios relative to a baseline mode for each generator."""
    if baseline not in agg["mode"].unique():
        raise ValueError(f"Baseline mode '{baseline}' not present in data")

    for generator, subset in agg.groupby("generator"):
        base_df = subset[subset["mode"] == baseline][["base_rows", "num_cols", "time_ms"]]
        base_df = base_df.rename(columns={"time_ms": "baseline_ms"})

        merged = subset.merge(base_df, on=["base_rows", "num_cols"], how="left")
        merged = merged[merged["mode"] != baseline].copy()
        merged["ratio"] = merged["time_ms"] / merged["baseline_ms"]

        if merged.empty:
            continue

        order = _mode_sort(merged["mode"].unique())

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=merged.sort_values(["base_rows", "mode"]),
            x="base_rows",
            y="ratio",
            hue="mode",
            hue_order=order,
            marker="o",
            ax=ax,
        )

        _use_line_style(ax)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Parity")
        ax.set_title(f"Speed relative to {baseline} — {generator}")
        ax.set_xlabel("Base constraints (m)")
        ax.set_ylabel("Mean time ratio")
        ax.legend(
            title="Mode",
            loc="upper left",
            fontsize="small",
            title_fontsize="small",
            framealpha=0.9,
        )
        fig.tight_layout()

        out_path = FIGURE_DIR / f"ratio_vs_{baseline}_{generator}.png"
        fig.savefig(out_path, dpi=250)
        plt.close(fig)


def plot_heatmap(agg: pd.DataFrame) -> None:
    """Heatmap of log10 time per generator."""
    for generator, subset in agg.groupby("generator"):
        pivot = subset.pivot_table(
            index="mode",
            columns="base_rows",
            values="time_ms",
            aggfunc="first",
        )
        mode_order = _mode_sort(pivot.index)
        pivot = pivot.reindex(mode_order)
        pivot = pivot[sorted(pivot.columns)]
        pivot = pivot.replace(0, np.nan).astype(float)
        log_pivot = np.log10(pivot)
        annot = pivot.map(lambda val: "" if pd.isna(val) else f"{val:.3g}")

        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.heatmap(
            log_pivot,
            cmap="rocket",
            annot=annot,
            fmt="",
            cbar_kws={"label": "log10(ms)"},
            ax=ax,
        )
        ax.set_title(f"Mean per-update time (ms) — {generator}")
        ax.set_xlabel("Base constraints (m)")
        ax.set_ylabel("Mode")
        fig.tight_layout()

        out_path = FIGURE_DIR / f"timing_heatmap_{generator}.png"
        fig.savefig(out_path, dpi=250)
        plt.close(fig)


def _mode_sort(modes: Iterable[str]) -> list[str]:
    order = ["sparse", "dense_scratch", "dense_givens_update"]
    mode_list = list(modes)
    mode_list.sort(key=lambda m: order.index(m) if m in order else len(order))
    return mode_list


def main(csv_path: Path | None = None) -> None:
    if csv_path is None:
        if len(sys.argv) > 1:
            csv_path = Path(sys.argv[1])
        else:
            csv_path = Path(__file__).parent / "givens_test.csv"
    df = load_results(csv_path)
    agg = aggregate_results(df)

    # Give a quick textual summary for sanity checking
    summary = (
        agg[["generator", "mode", "base_rows", "num_cols", "time_ms", "time_us", "std_ms"]]
        .sort_values(["generator", "mode", "base_rows"])
    )
    print("Aggregated timing summary (ms / us):")
    print(summary.to_string(index=False))

    plot_timings(agg)
    plot_relative_speed(agg, baseline="dense_scratch")
    plot_relative_speed(agg, baseline="sparse")
    plot_heatmap(agg)
    print(f"Saved figures to {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
