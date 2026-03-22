from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# -----------------------------
# Config (no CLI args required)
# -----------------------------
RUN_NAME = "military_matrix_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "src" / "eval_api" / "results" / RUN_NAME
SUMMARY_CSV_PATH = RESULTS_DIR / "summary.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

LANG_ORDER = ["nob", "pt", "de"]
LANG_LABELS = {"nob": "EN->NO", "pt": "EN->PT", "de": "EN->DE"}

BASE_MODEL_ORDER = ["OpusMT", "NLLB-600M", "M2M-418M", "M2M-1.2B"]
MODEL_ORDER = ["OpusMT", "OpusMT-FT", "NLLB-600M", "M2M-418M", "M2M-1.2B"]
MODEL_LABELS = {
    "OpusMT": "Opus Base",
    "OpusMT-FT": "Opus Finetuned",
    "NLLB-600M": "NLLB 600M",
    "M2M-418M": "M2M100 418M",
    "M2M-1.2B": "M2M100 1.2B",
}
MODEL_COLORS = {
    "OpusMT": "#4C78A8",
    "OpusMT-FT": "#2E8B57",
    "NLLB-600M": "#8E6BBE",
    "M2M-418M": "#E39A47",
    "M2M-1.2B": "#D86C70",
}


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#EFEFF2",
            "axes.facecolor": "#EFEFF2",
            "axes.edgecolor": "#C3C7CF",
            "axes.titlesize": 17,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )


def _ordered_models(df: pd.DataFrame, preferred_order: list[str]) -> list[str]:
    present = set(df["model_label"].unique())
    return [model for model in preferred_order if model in present]


def _aggregate_by_model(summary_df: pd.DataFrame, models: list[str], metric_cols: list[str]) -> pd.DataFrame:
    agg = (
        summary_df[summary_df["model_label"].isin(models)]
        .groupby("model_label", as_index=False)[metric_cols]
        .mean()
    )
    agg["model_label"] = pd.Categorical(agg["model_label"], categories=models, ordered=True)
    return agg.sort_values("model_label").reset_index(drop=True)


def _latency_column(summary_df: pd.DataFrame) -> tuple[str, str]:
    pure_col = "average_pure_inference_latency_ms"
    if pure_col in summary_df.columns and summary_df[pure_col].notna().any():
        return pure_col, "Pure Inference Latency (ms)"
    return "average_latency_ms", "Profiled Translation Latency (ms)"


def _add_headroom(ax: plt.Axes, ratio: float = 0.14) -> None:
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if span <= 0:
        span = 1.0
    ax.set_ylim(y0, y1 + span * ratio)


def _annotate_bars(ax: plt.Axes, bars, fmt: str, unit: str = "") -> None:
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    if span <= 0:
        span = 1.0
    for bar in bars:
        value = bar.get_height()
        if pd.isna(value):
            continue
        x = bar.get_x() + bar.get_width() / 2
        y = value + span * 0.015
        label = f"{fmt.format(value)}{unit}"
        ax.text(x, y, label, ha="center", va="bottom", fontsize=10, color="#1F2937")


def _plot_figure1_base_quality(summary_df: pd.DataFrame) -> None:
    models = _ordered_models(summary_df, BASE_MODEL_ORDER)
    metrics = [
        ("comet_mean", "COMET", "{:.3f}"),
        ("bleu_mean", "BLEU", "{:.1f}"),
        ("chrf_mean", "chrF", "{:.1f}"),
    ]
    agg = _aggregate_by_model(summary_df, models, [m[0] for m in metrics])

    fig, axes = plt.subplots(1, 3, figsize=(20, 8.5))
    x = np.arange(len(agg), dtype=float)
    labels = [MODEL_LABELS[m] for m in agg["model_label"]]
    colors = [MODEL_COLORS[m] for m in agg["model_label"]]

    for ax, (metric_col, title, fmt) in zip(axes, metrics):
        bars = ax.bar(x, agg[metric_col], color=colors, edgecolor="white", linewidth=0.9, alpha=0.95)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=14, ha="right")
        ax.grid(axis="y", alpha=0.35)
        _add_headroom(ax)
        _annotate_bars(ax, bars, fmt=fmt)

    fig.suptitle("Figure 1. Base Models: Quality Metrics (Averaged Across EN->NO/PT/DE)", fontsize=21, y=1.01)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(str(PLOTS_DIR / "figure_01_base_quality_aggregated.png"), dpi=200)
    plt.close(fig)


def _plot_figure2_base_latency(summary_df: pd.DataFrame) -> None:
    models = _ordered_models(summary_df, BASE_MODEL_ORDER)
    plot_df = summary_df.copy()
    plot_df["baseline_rss_gb"] = plot_df["baseline_rss_mb"] / 1024.0
    latency_col, latency_title = _latency_column(plot_df)

    metrics = [
        (latency_col, latency_title, "{:.0f}"),
        ("cpu_percent_per_core_mean", "CPU Usage (% per core)", "{:.1f}"),
        ("baseline_rss_gb", "Model Memory Footprint (GB RSS)", "{:.2f}"),
    ]
    agg = _aggregate_by_model(plot_df, models, [m[0] for m in metrics])

    fig, axes = plt.subplots(1, 3, figsize=(20, 8.5))
    x = np.arange(len(agg), dtype=float)
    labels = [MODEL_LABELS[m] for m in agg["model_label"]]
    colors = [MODEL_COLORS[m] for m in agg["model_label"]]

    for ax, (metric_col, title, fmt) in zip(axes, metrics):
        bars = ax.bar(x, agg[metric_col], color=colors, edgecolor="white", linewidth=0.9, alpha=0.95)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=14, ha="right")
        ax.grid(axis="y", alpha=0.35)
        _add_headroom(ax)
        _annotate_bars(ax, bars, fmt=fmt)

    fig.suptitle("Figure 2. Base Models: Latency + Resource Usage (Averaged Across EN->NO/PT/DE)", fontsize=21, y=1.01)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(str(PLOTS_DIR / "figure_02_base_latency_aggregated.png"), dpi=200)
    plt.close(fig)


def _plot_figure3_per_language_all_models(summary_df: pd.DataFrame) -> None:
    models = _ordered_models(summary_df, MODEL_ORDER)
    metrics = [("comet_mean", "COMET"), ("bleu_mean", "BLEU"), ("chrf_mean", "chrF")]

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))
    x = np.arange(len(LANG_ORDER), dtype=float)
    width = 0.84 / max(1, len(models))

    for ax, (metric_col, title) in zip(axes, metrics):
        for idx, model in enumerate(models):
            vals = []
            for lang in LANG_ORDER:
                match = summary_df[(summary_df["model_label"] == model) & (summary_df["display_lang"] == lang)]
                vals.append(float(match[metric_col].iloc[0]) if not match.empty else np.nan)
            offset = (idx - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.96,
                label=MODEL_LABELS[model],
            )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS[l] for l in LANG_ORDER], rotation=0)
        ax.grid(axis="y", alpha=0.35)
        _add_headroom(ax, ratio=0.08)

    handles = [Line2D([0], [0], color=MODEL_COLORS[m], lw=8, label=MODEL_LABELS[m]) for m in models]
    fig.legend(handles=handles, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 0.92), frameon=True)
    fig.suptitle("Figure 3. Per-Language Comparison: Base vs Finetuned", fontsize=21, y=1.01)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(str(PLOTS_DIR / "figure_03_per_language_finetuned_vs_base.png"), dpi=200)
    plt.close(fig)


def _plot_figure4_tradeoff(summary_df: pd.DataFrame) -> None:
    models = _ordered_models(summary_df, MODEL_ORDER)
    latency_col, latency_title = _latency_column(summary_df)
    agg = _aggregate_by_model(
        summary_df,
        models,
        [latency_col, "comet_mean", "params_m"],
    )
    agg["params_m"] = agg["params_m"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(11.5, 8.3))
    x = agg[latency_col]
    y = agg["comet_mean"]
    sizes = 260 + np.clip(agg["params_m"], 0, None) * 1.15
    colors = [MODEL_COLORS[m] for m in agg["model_label"]]

    ax.scatter(x, y, s=sizes, c=colors, alpha=0.88, edgecolors="white", linewidth=1.1)
    for _, row in agg.iterrows():
        label = MODEL_LABELS[row["model_label"]]
        ax.annotate(
            label,
            (row[latency_col], row["comet_mean"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
            color="#1F2937",
        )

    ax.set_xlabel(latency_title)
    ax.set_ylabel("COMET (avg)")
    ax.set_title("Figure 4. Accuracy vs Latency Trade-off (Bubble Size ~ Params)")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "figure_04_accuracy_latency_tradeoff.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    if not SUMMARY_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV_PATH}")

    _apply_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(SUMMARY_CSV_PATH)

    _plot_figure1_base_quality(summary_df)
    _plot_figure2_base_latency(summary_df)
    _plot_figure3_per_language_all_models(summary_df)
    _plot_figure4_tradeoff(summary_df)

    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
