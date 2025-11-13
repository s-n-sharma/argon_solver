from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = {
	"generator",
	"mode",
	"base_rows",
	"num_cols",
	"update_index",
	"update_type",
	"active_rows",
	"time_seconds",
	"detection_type",
	"detection_row_index",
	"detection_correct",
	"residual_norm",
}

MODE_ORDER = [
	"cached_dense_qr",
	"dense_scratch",
	"sparse_scratch",
	"kaczmarz_warm_start",
]

UPDATE_ORDER = ["add", "delete", "modify"]

FIGURE_DIR = Path(__file__).parent / "dynamic_constraint_solving"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_results(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	missing = REQUIRED_COLUMNS - set(df.columns)
	if missing:
		raise ValueError(f"Missing required columns: {sorted(missing)}")

	df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
	df["update_type"] = pd.Categorical(df["update_type"], categories=UPDATE_ORDER, ordered=True)
	df["time_ms"] = df["time_seconds"] * 1e3
	df["time_us"] = df["time_seconds"] * 1e6
	df["detection_type"] = df["detection_type"].astype(str)
	df["detection_row_index"] = df["detection_row_index"].astype(int)
	df["detection_correct"] = df["detection_correct"].astype(int)
	df["residual_norm"] = df["residual_norm"].astype(float)
	return df


def aggregate_by_base(df: pd.DataFrame) -> pd.DataFrame:
	keys = ["generator", "mode", "base_rows", "num_cols"]
	grouped = (
		df.groupby(keys, as_index=False, observed=True)
		.agg(
			mean_time=pd.NamedAgg(column="time_seconds", aggfunc="mean"),
			median_time=pd.NamedAgg(column="time_seconds", aggfunc="median"),
			min_time=pd.NamedAgg(column="time_seconds", aggfunc="min"),
			max_time=pd.NamedAgg(column="time_seconds", aggfunc="max"),
			std_time=pd.NamedAgg(column="time_seconds", aggfunc="std"),
			p95_time=pd.NamedAgg(column="time_seconds", aggfunc=lambda s: s.quantile(0.95)),
			mean_residual=pd.NamedAgg(column="residual_norm", aggfunc="mean"),
			median_residual=pd.NamedAgg(column="residual_norm", aggfunc="median"),
			min_residual=pd.NamedAgg(column="residual_norm", aggfunc="min"),
			max_residual=pd.NamedAgg(column="residual_norm", aggfunc="max"),
			std_residual=pd.NamedAgg(column="residual_norm", aggfunc="std"),
			p95_residual=pd.NamedAgg(column="residual_norm", aggfunc=lambda s: s.quantile(0.95)),
		)
	)
	grouped["time_ms"] = grouped["mean_time"] * 1e3
	grouped["time_us"] = grouped["mean_time"] * 1e6
	grouped["p95_ms"] = grouped["p95_time"] * 1e3
	grouped["std_ms"] = grouped["std_time"].fillna(0.0) * 1e3
	grouped["std_residual"] = grouped["std_residual"].fillna(0.0)
	return grouped


def aggregate_by_type(df: pd.DataFrame) -> pd.DataFrame:
	keys = ["generator", "update_type", "mode", "base_rows", "num_cols"]
	grouped = (
		df.groupby(keys, as_index=False, observed=True)
		.agg(
			mean_time=pd.NamedAgg(column="time_seconds", aggfunc="mean"),
			median_time=pd.NamedAgg(column="time_seconds", aggfunc="median"),
			min_time=pd.NamedAgg(column="time_seconds", aggfunc="min"),
			max_time=pd.NamedAgg(column="time_seconds", aggfunc="max"),
			std_time=pd.NamedAgg(column="time_seconds", aggfunc="std"),
			p95_time=pd.NamedAgg(column="time_seconds", aggfunc=lambda s: s.quantile(0.95)),
			mean_residual=pd.NamedAgg(column="residual_norm", aggfunc="mean"),
			median_residual=pd.NamedAgg(column="residual_norm", aggfunc="median"),
			min_residual=pd.NamedAgg(column="residual_norm", aggfunc="min"),
			max_residual=pd.NamedAgg(column="residual_norm", aggfunc="max"),
			std_residual=pd.NamedAgg(column="residual_norm", aggfunc="std"),
			p95_residual=pd.NamedAgg(column="residual_norm", aggfunc=lambda s: s.quantile(0.95)),
			count=pd.NamedAgg(column="time_seconds", aggfunc="size"),
		)
	)
	grouped["time_ms"] = grouped["mean_time"] * 1e3
	grouped["time_us"] = grouped["mean_time"] * 1e6
	grouped["p95_ms"] = grouped["p95_time"] * 1e3
	grouped["std_ms"] = grouped["std_time"].fillna(0.0) * 1e3
	grouped["std_residual"] = grouped["std_residual"].fillna(0.0)
	grouped["count"] = grouped["count"].astype(int)
	return grouped


def aggregate_by_active_rows(df: pd.DataFrame) -> pd.DataFrame:
	grouped = (
		df.groupby(["generator", "mode", "active_rows"], as_index=False, observed=True)
		.agg(
			mean_time=pd.NamedAgg(column="time_seconds", aggfunc="mean"),
			median_time=pd.NamedAgg(column="time_seconds", aggfunc="median"),
			std_time=pd.NamedAgg(column="time_seconds", aggfunc="std"),
			mean_residual=pd.NamedAgg(column="residual_norm", aggfunc="mean"),
			std_residual=pd.NamedAgg(column="residual_norm", aggfunc="std"),
			count=pd.NamedAgg(column="time_seconds", aggfunc="size"),
		)
	)
	grouped["time_ms"] = grouped["mean_time"] * 1e3
	grouped["std_ms"] = grouped["std_time"].fillna(0.0) * 1e3
	grouped["std_residual"] = grouped["std_residual"].fillna(0.0)
	grouped["count"] = grouped["count"].astype(int)
	return grouped


def _mode_sort(modes: Iterable[str]) -> list[str]:
	unique = list(pd.Series(modes).dropna().unique())
	unique.sort(key=lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else len(MODE_ORDER))
	return unique


def _update_sort(updates: Iterable[str]) -> list[str]:
	unique = list(pd.Series(updates).dropna().unique())
	order_map = {name: idx for idx, name in enumerate(UPDATE_ORDER)}
	unique.sort(key=lambda u: order_map.get(u, len(order_map)))
	return unique


def _configure_axes(ax: plt.Axes) -> None:
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)


def plot_time_vs_base(agg: pd.DataFrame) -> None:
	for (generator, update_type), subset in agg.groupby(["generator", "update_type"], observed=True):
		if subset.empty:
			continue
		subset = subset.sort_values(["base_rows", "mode"])
		fig, ax = plt.subplots(figsize=(8.5, 5.2))
		sns.lineplot(
			data=subset,
			x="base_rows",
			y="time_ms",
			hue="mode",
			hue_order=_mode_sort(subset["mode"].unique()),
			marker="o",
			ax=ax,
		)
		_configure_axes(ax)
		ax.set_title(f"Mean time vs. base rows — {generator} ({update_type})")
		ax.set_xlabel("Base constraints (m)")
		ax.set_ylabel("Mean time per update (ms)")
		ax.legend(title="Mode", fontsize="small", title_fontsize="small", framealpha=0.9)
		fig.tight_layout()
		out_path = FIGURE_DIR / f"time_vs_base_{generator}_{update_type}.png"
		fig.savefig(out_path, dpi=250)
		plt.close(fig)


def plot_ratio_vs_baseline(agg: pd.DataFrame, baseline: str = "dense_scratch") -> None:
	for (generator, update_type), subset in agg.groupby(["generator", "update_type"], observed=True):
		if baseline not in subset["mode"].unique():
			continue
		base_df = subset[subset["mode"] == baseline][["base_rows", "num_cols", "time_ms"]]
		base_df = base_df.rename(columns={"time_ms": "baseline_ms"})
		merged = subset.merge(base_df, on=["base_rows", "num_cols"], how="left")
		merged = merged[merged["mode"] != baseline].copy()
		merged["ratio"] = merged["time_ms"] / merged["baseline_ms"]
		if merged.empty:
			continue
		merged = merged.sort_values(["base_rows", "mode"])
		fig, ax = plt.subplots(figsize=(8.5, 5.2))
		sns.lineplot(
			data=merged,
			x="base_rows",
			y="ratio",
			hue="mode",
			hue_order=_mode_sort(merged["mode"].unique()),
			marker="o",
			ax=ax,
		)
		ax.set_xscale("log")
		ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
		ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Parity")
		ax.set_title(f"Speed vs. {baseline} — {generator} ({update_type})")
		ax.set_xlabel("Base constraints (m)")
		ax.set_ylabel("Mean time ratio")
		ax.legend(title="Mode", fontsize="small", title_fontsize="small", framealpha=0.9)
		fig.tight_layout()
		out_path = FIGURE_DIR / f"ratio_vs_{baseline}_{generator}_{update_type}.png"
		fig.savefig(out_path, dpi=250)
		plt.close(fig)


def plot_update_traces(df: pd.DataFrame) -> None:
	for generator, subset in df.groupby("generator", observed=True):
		if subset.empty:
			continue
		subset = subset.sort_values(["base_rows", "update_index", "mode"])
		g = sns.FacetGrid(
			subset,
			col="base_rows",
			hue="mode",
			hue_order=_mode_sort(subset["mode"].unique()),
			col_wrap=3,
			height=3.4,
			sharey=False,
		)
		g.map_dataframe(sns.lineplot, x="update_index", y="time_ms")
		g.add_legend(title="Mode")
		for ax in g.axes.flat:
			if ax is None:
				continue
			ax.set_ylabel("Time per update (ms)")
			ax.set_xlabel("Update index")
			ax.set_yscale("log")
			ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
		g.fig.subplots_adjust(top=0.9)
		g.fig.suptitle(f"Update trajectories — {generator}")
		out_path = FIGURE_DIR / f"update_traces_{generator}.png"
		g.savefig(out_path, dpi=250)
		plt.close(g.fig)


def plot_box_by_update_type(df: pd.DataFrame) -> None:
	for generator, subset in df.groupby("generator", observed=True):
		if subset.empty:
			continue
		fig, ax = plt.subplots(figsize=(9, 5.2))
		sns.boxplot(
			data=subset,
			x="update_type",
			y="time_ms",
			hue="mode",
			hue_order=_mode_sort(subset["mode"].unique()),
			order=_update_sort(subset["update_type"].unique()),
			showfliers=False,
			ax=ax,
		)
		sns.stripplot(
			data=subset,
			x="update_type",
			y="time_ms",
			hue="mode",
			hue_order=_mode_sort(subset["mode"].unique()),
			order=_update_sort(subset["update_type"].unique()),
			dodge=True,
			alpha=0.25,
			size=2,
			linewidth=0,
			ax=ax,
		)
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles[: len(labels) // 2], labels[: len(labels) // 2], title="Mode", fontsize="small", title_fontsize="small")
		ax.set_yscale("log")
		ax.set_xlabel("Update type")
		ax.set_ylabel("Time per update (ms)")
		ax.set_title(f"Distribution by update type — {generator}")
		ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
		fig.tight_layout()
		out_path = FIGURE_DIR / f"box_by_update_type_{generator}.png"
		fig.savefig(out_path, dpi=250)
		plt.close(fig)


def plot_active_rows_profile(agg_active: pd.DataFrame) -> None:
	for generator, subset in agg_active.groupby("generator", observed=True):
		if subset.empty:
			continue
		subset = subset.sort_values(["active_rows", "mode"])
		fig, ax = plt.subplots(figsize=(8.5, 5.0))
		sns.lineplot(
			data=subset,
			x="active_rows",
			y="time_ms",
			hue="mode",
			hue_order=_mode_sort(subset["mode"].unique()),
			marker="o",
			ax=ax,
		)
		ax.set_yscale("log")
		ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
		ax.set_title(f"Mean time vs. active rows — {generator}")
		ax.set_xlabel("Active constraints")
		ax.set_ylabel("Mean time (ms)")
		ax.legend(title="Mode", fontsize="small", title_fontsize="small", framealpha=0.9)
		fig.tight_layout()
		out_path = FIGURE_DIR / f"time_vs_active_rows_{generator}.png"
		fig.savefig(out_path, dpi=250)
		plt.close(fig)


def plot_detection_accuracy(df: pd.DataFrame) -> None:
	cached = df[df["mode"] == "cached_dense_qr"].copy()
	if cached.empty:
		return

	cached["accuracy"] = cached["detection_correct"].astype(float)
	summary = (
		cached.groupby(["generator", "update_type"], observed=True)["accuracy"]
		.mean()
		.reset_index()
	)

	for generator, subset in summary.groupby("generator", observed=True):
		if subset.empty:
			continue
		subset = subset.sort_values("update_type")
		fig, ax = plt.subplots(figsize=(6.5, 4.2))
		sns.barplot(
			data=subset,
			x="update_type",
			y="accuracy",
			order=_update_sort(subset["update_type"].unique()),
			ax=ax,
			color="#4c72b0",
		)
		ax.set_ylim(0.0, 1.05)
		ax.set_ylabel("Detection accuracy")
		ax.set_xlabel("Update type")
		ax.set_title(f"Change-detection accuracy — {generator}")
		ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
		for container in ax.containers:
			ax.bar_label(container, fmt="{:.2f}")
		fig.tight_layout()
		out_path = FIGURE_DIR / f"detection_accuracy_{generator}.png"
		fig.savefig(out_path, dpi=250)
		plt.close(fig)


def summarize_detection_types(df: pd.DataFrame) -> pd.DataFrame:
	cached = df[df["mode"] == "cached_dense_qr"].copy()
	if cached.empty:
		return pd.DataFrame(columns=["generator", "update_type", "detection_type", "count", "share"])

	summary = (
		cached.groupby(["generator", "update_type", "detection_type"], observed=True)["update_index"]
		.agg(count="size")
		.reset_index()
	)
	summary["count"] = summary["count"].astype(int)
	summary["share"] = summary.groupby(["generator", "update_type"], observed=True)["count"].transform(lambda s: s / s.sum())
	return summary.sort_values(["generator", "update_type", "detection_type"])


def plot_detection_breakdown(df: pd.DataFrame) -> None:
	breakdown = summarize_detection_types(df)
	if breakdown.empty:
		return

	for generator, subset in breakdown.groupby("generator", observed=True):
		if subset.empty:
			continue
		subset = subset.sort_values(["update_type", "detection_type"])
		fig, ax = plt.subplots(figsize=(7.0, 4.4))
		sns.barplot(
			data=subset,
			x="update_type",
			y="share",
			hue="detection_type",
			order=_update_sort(subset["update_type"].unique()),
			ax=ax,
		)
		ax.set_ylim(0.0, 1.05)
		ax.set_ylabel("Detection share")
		ax.set_xlabel("Update type")
		ax.set_title(f"Detection outcome breakdown — {generator}")
		ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
		handles, labels = ax.get_legend_handles_labels()
		if handles:
			ax.legend(title="Detection type", fontsize="small", title_fontsize="small", framealpha=0.9)
		elif ax.legend_:
			ax.legend_.remove()
		fig.tight_layout()
		out_path = FIGURE_DIR / f"detection_breakdown_{generator}.png"
		fig.savefig(out_path, dpi=250)
		plt.close(fig)


def plot_residual_by_mode(df: pd.DataFrame) -> None:
	eps = 1e-18
	for generator, subset in df.groupby("generator", observed=True):
		if subset.empty:
			continue
		data = subset.copy()
		data["residual_plot"] = data["residual_norm"].clip(lower=eps)
		fig, ax = plt.subplots(figsize=(9.0, 5.0))
		sns.boxplot(
			data=data,
			x="update_type",
			y="residual_plot",
			hue="mode",
			hue_order=_mode_sort(data["mode"].unique()),
			order=_update_sort(data["update_type"].unique()),
			showfliers=False,
			ax=ax,
		)
		sns.stripplot(
			data=data,
			x="update_type",
			y="residual_plot",
			hue="mode",
			hue_order=_mode_sort(data["mode"].unique()),
			order=_update_sort(data["update_type"].unique()),
			dodge=True,
			alpha=0.25,
			size=2,
			linewidth=0,
			ax=ax,
		)
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles[: len(labels) // 2], labels[: len(labels) // 2], title="Mode", fontsize="small", title_fontsize="small")
		ax.set_yscale("log")
		ax.set_ylabel(r"Residual norm $\|Ax - b\|_2$")
		ax.set_xlabel("Update type")
		ax.set_title(f"Residual distribution by update — {generator}")
		ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
		fig.tight_layout()
		out_path = FIGURE_DIR / f"residual_by_update_{generator}.png"
		fig.savefig(out_path, dpi=250)
		plt.close(fig)


def plot_residual_trend(agg: pd.DataFrame, metric: str = "p95_residual") -> None:
	if metric not in agg.columns:
		return

	label_map = {
		"mean_residual": "Mean residual norm",
		"median_residual": "Median residual norm",
		"p95_residual": "95th percentile residual norm",
		"max_residual": "Max residual norm",
	}
	y_label = label_map.get(metric, f"{metric.replace('_', ' ').title()}")

	for (generator, update_type), subset in agg.groupby(["generator", "update_type"], observed=True):
		if subset.empty:
			continue
		subset = subset.sort_values(["base_rows", "mode"])
		fig, ax = plt.subplots(figsize=(8.5, 5.0))
		sns.lineplot(
			data=subset,
			x="base_rows",
			y=metric,
			hue="mode",
			hue_order=_mode_sort(subset["mode"].unique()),
			marker="o",
			ax=ax,
		)
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
		ax.set_title(f"{y_label} vs. base rows — {generator} ({update_type})")
		ax.set_xlabel("Base constraints (m)")
		ax.set_ylabel(y_label)
		ax.legend(title="Mode", fontsize="small", title_fontsize="small", framealpha=0.9)
		fig.tight_layout()
		metric_suffix = metric.replace("_", "-")
		out_path = FIGURE_DIR / f"residual_trend_{metric_suffix}_{generator}_{update_type}.png"
		fig.savefig(out_path, dpi=250)
		plt.close(fig)


def summarize_modes(df: pd.DataFrame) -> pd.DataFrame:
	summary = (
		df.groupby(["generator", "update_type", "mode"], as_index=False, observed=True)
		.agg(
			mean_ms=pd.NamedAgg(column="time_ms", aggfunc="mean"),
			median_ms=pd.NamedAgg(column="time_ms", aggfunc="median"),
			p95_ms=pd.NamedAgg(column="time_ms", aggfunc=lambda s: s.quantile(0.95)),
			min_ms=pd.NamedAgg(column="time_ms", aggfunc="min"),
			max_ms=pd.NamedAgg(column="time_ms", aggfunc="max"),
			mean_residual=pd.NamedAgg(column="residual_norm", aggfunc="mean"),
			median_residual=pd.NamedAgg(column="residual_norm", aggfunc="median"),
			p95_residual=pd.NamedAgg(column="residual_norm", aggfunc=lambda s: s.quantile(0.95)),
			max_residual=pd.NamedAgg(column="residual_norm", aggfunc="max"),
		)
	)
	return summary.sort_values(["generator", "update_type", "mode"])


def summarize_detection(df: pd.DataFrame) -> pd.DataFrame:
	cached = df[df["mode"] == "cached_dense_qr"].copy()
	if cached.empty:
		return pd.DataFrame(columns=["generator", "update_type", "accuracy", "failures", "total"])

	summary = (
		cached.groupby(["generator", "update_type"], observed=True)["detection_correct"]
		.agg(
			accuracy="mean",
			failures=lambda s: int((s == 0).sum()),
			total="size",
		)
		.reset_index()
	)
	summary["accuracy"] = summary["accuracy"].astype(float)
	summary["total"] = summary["total"].astype(int)
	return summary.sort_values(["generator", "update_type"])


def main(csv_path: Path | None = None) -> None:
	if csv_path is None:
		if len(sys.argv) > 1:
			csv_path = Path(sys.argv[1])
		else:
			csv_path = Path(__file__).parent / "dynamic_qr_benchmark_results.csv"

	df = load_results(csv_path)
	agg_base = aggregate_by_base(df)
	agg_type = aggregate_by_type(df)
	agg_active = aggregate_by_active_rows(df)
	mode_summary = summarize_modes(df)
	detection_summary = summarize_detection(df)
	detection_breakdown = summarize_detection_types(df)

	sns.set_theme(style="whitegrid", context="talk")

	print("Per-mode timing and residual summary:")
	with pd.option_context("display.max_rows", None, "display.max_columns", None):
		formatters = {
			"mean_residual": "{:.2e}".format,
			"median_residual": "{:.2e}".format,
			"p95_residual": "{:.2e}".format,
			"max_residual": "{:.2e}".format,
		}
		print(mode_summary.to_string(index=False, formatters=formatters))

	if not detection_summary.empty:
		print("\nChange-detection accuracy (cached solver only):")
		det_display = detection_summary.copy()
		det_display["accuracy_pct"] = det_display["accuracy"] * 100.0
		with pd.option_context("display.max_rows", None, "display.max_columns", None):
			print(det_display[["generator", "update_type", "accuracy_pct", "failures", "total"]].to_string(index=False, formatters={"accuracy_pct": "{:.2f}".format}))

	if not detection_breakdown.empty:
		print("\nDetection outcomes by type (cached solver only):")
		break_display = detection_breakdown.copy()
		break_display["share_pct"] = break_display["share"] * 100.0
		with pd.option_context("display.max_rows", None, "display.max_columns", None):
			print(break_display[["generator", "update_type", "detection_type", "count", "share_pct"]].to_string(index=False, formatters={"share_pct": "{:.2f}".format}))

	plot_time_vs_base(agg_type)
	plot_ratio_vs_baseline(agg_type, baseline="dense_scratch")
	plot_ratio_vs_baseline(agg_type, baseline="sparse_scratch")
	plot_update_traces(df)
	plot_box_by_update_type(df)
	plot_active_rows_profile(agg_active)
	plot_detection_accuracy(df)
	plot_detection_breakdown(df)
	plot_residual_by_mode(df)
	plot_residual_trend(agg_type, metric="p95_residual")

	print(f"Saved figures to {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
	main()

