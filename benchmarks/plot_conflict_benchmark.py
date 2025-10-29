from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from io import StringIO
import os

# The new data provided by the user
NEW_DATA = """generator,size,nnz,sparse_ms,dense_ms,svd_ms,sparse_detected,dense_detected,svd_detected
random_sparse,5,1,0.238,2.944,2.391,1,1,1
random_sparse,10,1,0.050,0.009,0.045,1,1,1
random_sparse,25,3,0.051,0.008,0.042,1,1,1
random_sparse,50,12,0.092,0.062,0.315,1,1,1
random_sparse,100,50,0.499,0.292,2.977,1,1,1
random_sparse,250,311,2.180,0.955,5.233,1,1,1
random_sparse,500,1245,14.735,5.150,24.025,1,1,1
random_sparse,1000,4974,84.959,62.190,142.985,1,1,1
random_sparse,2000,19902,602.151,281.303,618.977,1,1,1
random_sparse,5000,124333,6090.471,5381.190,8423.061,1,1,1
circular_graph,5,10,0.068,0.009,0.022,1,1,1
circular_graph,10,20,0.031,0.003,0.021,1,1,1
circular_graph,25,50,0.054,0.008,0.049,1,1,1
circular_graph,50,100,0.099,0.033,0.139,1,1,1
circular_graph,100,200,0.253,0.069,0.679,1,1,1
circular_graph,250,500,1.261,0.672,3.147,1,1,1
circular_graph,500,1000,5.347,3.218,13.999,1,1,1
circular_graph,1000,2000,22.872,33.634,86.293,1,1,1
circular_graph,2000,4000,117.960,262.556,597.906,1,1,1
circular_graph,5000,10000,1182.736,5387.097,8236.752,1,1,1
block_diagonal,5,5,0.066,0.008,0.020,1,1,1
block_diagonal,10,10,0.027,0.003,0.012,1,1,1
block_diagonal,25,25,0.042,0.008,0.021,1,1,1
block_diagonal,50,50,0.055,0.028,0.047,1,1,1
block_diagonal,100,100,0.096,0.054,0.345,1,1,1
block_diagonal,250,250,0.226,0.351,2.085,1,1,1
block_diagonal,500,500,0.532,1.445,10.843,1,1,1
block_diagonal,1000,1000,2.208,8.636,70.128,1,1,1
block_diagonal,2000,2000,4.863,46.422,500.770,1,1,1
block_diagonal,5000,5000,43.349,677.836,7394.375,1,1,1
"""

# Use a relative path for the output directory
FIGURE_DIR = Path("conflict_plots")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_results(csv_path: Path | None = None) -> pd.DataFrame:
    """Loads benchmark results from CSV data."""
    if csv_path is None:
        # Use the new data string if no path is provided
        return pd.read_csv(StringIO(NEW_DATA))
    return pd.read_csv(csv_path)


def plot_timings(df: pd.DataFrame) -> None:
    """Plots timings for sparse, dense, and SVD methods on log-log scale."""
    sns.set_theme(style="whitegrid")

    for generator, subset in df.groupby("generator"):
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = subset.sort_values("size")
        
        # Plot all three timing columns
        ax.plot(subset["size"], subset["sparse_ms"], marker="o", label="Sparse QR (SuiteSparse)")
        ax.plot(subset["size"], subset["dense_ms"], marker="s", label="Dense QR (pivoted)")
        ax.plot(subset["size"], subset["svd_ms"], marker="x", label="SVD (Dense)") 
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Timing vs constraint count: {generator}")
        ax.set_xlabel("Number of constraints (m)")
        ax.set_ylabel("Solve time (ms)")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        
        # Save figure to the designated directory
        fig.savefig(FIGURE_DIR / f"timing_{generator}.png", dpi=200)
        plt.close(fig)
        print(f"Saved timing plot for {generator}")


def plot_detection(df: pd.DataFrame) -> None:
    """Plots detection success heatmaps for all three methods."""
    
    # Sparse QR Detection Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        df.pivot_table(
            index="generator",
            columns="size",
            values="sparse_detected",
            aggfunc="first",
        ),
        cmap="YlGn",
        annot=True,
        fmt=".0f",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Sparse QR detection success (1=yes)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "detection_sparse.png", dpi=200)
    plt.close(fig)
    print("Saved sparse detection plot")

    # Dense QR Detection Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        df.pivot_table(
            index="generator",
            columns="size",
            values="dense_detected",
            aggfunc="first",
        ),
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Dense QR detection success (1=yes)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "detection_dense.png", dpi=200)
    plt.close(fig)
    print("Saved dense detection plot")

    # SVD Detection Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        df.pivot_table(
            index="generator",
            columns="size",
            values="svd_detected",
            aggfunc="first",
        ),
        cmap="YlGnBu", # Use a different colormap for SVD
        annot=True,
        fmt=".0f",
        cbar=False,
        ax=ax,
    )
    ax.set_title("SVD detection success (1=yes)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "detection_svd.png", dpi=200)
    plt.close(fig)
    print("Saved SVD detection plot")


def plot_speedup(df: pd.DataFrame) -> None:
    """Plots timing ratios relative to the sparse method."""
    df = df.copy()
    
    # Calculate ratios relative to sparse QR
    df["Dense / Sparse"] = df["dense_ms"] / df["sparse_ms"].replace(0, pd.NA)
    df["SVD / Sparse"] = df["svd_ms"] / df["sparse_ms"].replace(0, pd.NA)

    # Melt the dataframe to long format for easier plotting with seaborn
    df_melted = df.melt(
        id_vars=["generator", "size"],
        value_vars=["Dense / Sparse", "SVD / Sparse"],
        var_name="Ratio Type",
        value_name="Timing Ratio",
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(
        data=df_melted,
        x="size",
        y="Timing Ratio",
        hue="generator",
        style="Ratio Type", # Differentiate ratios by line style
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log") # Use log scale for ratios as well
    ax.set_title("Timing ratio relative to Sparse QR")
    ax.set_xlabel("Number of constraints (m)")
    ax.set_ylabel("Timing Ratio (Method / Sparse)")
    
    # Add a horizontal line at y=1.0 to mark where methods are equal
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Ratio = 1.0")
    
    # Place legend outside the plot to avoid overlap
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    
    fig.savefig(FIGURE_DIR / "speedup_ratios.png", dpi=200)
    plt.close(fig)
    print("Saved speedup ratio plot")


def main() -> None:
    """Main function to load data and generate all plots."""
    df = load_results()
    plot_timings(df)
    plot_detection(df)
    plot_speedup(df)
    print(f"All plots saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()