from __future__ import annotations

import json
import re
from pathlib import Path
from urllib import error, request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# -----------------------------
# Config (no CLI args required)
# -----------------------------
API_BASE_URL = "http://127.0.0.1:8000"
HTTP_TIMEOUT_SECONDS = 3600
MAX_ITEMS_PER_LANGUAGE: int | None = None
RUN_NAME_BASE = "military_matrix_v2"
USE_CT2_FOR_ALL_MODELS = True
# If True and existing result CSVs are present, skip /evaluate calls and only rebuild tables/plots.
REUSE_EXISTING_RESULTS = True
RUN_NAME = f"{RUN_NAME_BASE}_ct2_all" if USE_CT2_FOR_ALL_MODELS else RUN_NAME_BASE

METRICS = ["bleu", "chrf", "ter", "comet"]
TOP_EXAMPLES_PER_PAIR = 10

CT2_MODEL_ID_OVERRIDES = {
    "m2m-100-418m": "m2m-100-418m-ct2",
    "m2m-100-1.2b": "m2m-100-1.2b-ct2",
    "nllb-200-distilled-600m": "nllb-200-distilled-600m-ct2",
}

MODEL_RUNS = [
    {
        "run_id": "opus_base_en_de",
        "model_id": "opus-mt-tc-big-en-de",
        "model_label": "OpusMT",
        "model_short": "OPUS-base",
        "family": "opus",
        "stage": "baseline",
        "params_m": 200,
        "src_lang": "en",
        "tgt_lang": "de",
        "display_lang": "de",
        "reference_key": "deu",
    },
    {
        "run_id": "opus_ft_en_de",
        "model_id": "opus-mt-tc-big-en-de-military-v1",
        "model_label": "OpusMT-FT",
        "model_short": "OPUS-ft",
        "family": "opus_ft",
        "stage": "finetuned",
        "params_m": 200,
        "src_lang": "en",
        "tgt_lang": "de",
        "display_lang": "de",
        "reference_key": "deu",
    },
    {
        "run_id": "opus_base_en_nob",
        "model_id": "opus-mt-tc-big-en-gmq",
        "model_label": "OpusMT",
        "model_short": "OPUS-base",
        "family": "opus",
        "stage": "baseline",
        "params_m": 200,
        "src_lang": "en",
        "tgt_lang": "nob",
        "display_lang": "nob",
        "reference_key": "nob",
    },
    {
        "run_id": "opus_ft_en_nob",
        "model_id": "opus-mt-tc-big-en-nob-military",
        "model_label": "OpusMT-FT",
        "model_short": "OPUS-ft",
        "family": "opus_ft",
        "stage": "finetuned",
        "params_m": 200,
        "src_lang": "en",
        "tgt_lang": "nob",
        "display_lang": "nob",
        "reference_key": "nob",
    },
    {
        "run_id": "opus_base_en_pt",
        "model_id": "opus-mt-tc-big-en-pt",
        "model_label": "OpusMT",
        "model_short": "OPUS-base",
        "family": "opus",
        "stage": "baseline",
        "params_m": 200,
        "src_lang": "en",
        "tgt_lang": "pt",
        "display_lang": "pt",
        "reference_key": "por",
    },
    {
        "run_id": "opus_ft_en_pt",
        "model_id": "opus-mt-tc-big-en-pt-military",
        "model_label": "OpusMT-FT",
        "model_short": "OPUS-ft",
        "family": "opus_ft",
        "stage": "finetuned",
        "params_m": 200,
        "src_lang": "en",
        "tgt_lang": "pt",
        "display_lang": "pt",
        "reference_key": "por",
    },
    {
        "run_id": "m2m_418_en_de",
        "model_id": "m2m-100-418m",
        "model_label": "M2M-418M",
        "model_short": "M2M-418M",
        "family": "m2m",
        "stage": "baseline",
        "params_m": 418,
        "src_lang": "en",
        "tgt_lang": "de",
        "display_lang": "de",
        "reference_key": "deu",
    },
    {
        "run_id": "m2m_418_en_nob",
        "model_id": "m2m-100-418m",
        "model_label": "M2M-418M",
        "model_short": "M2M-418M",
        "family": "m2m",
        "stage": "baseline",
        "params_m": 418,
        "src_lang": "en",
        "tgt_lang": "no",
        "display_lang": "nob",
        "reference_key": "nob",
    },
    {
        "run_id": "m2m_418_en_pt",
        "model_id": "m2m-100-418m",
        "model_label": "M2M-418M",
        "model_short": "M2M-418M",
        "family": "m2m",
        "stage": "baseline",
        "params_m": 418,
        "src_lang": "en",
        "tgt_lang": "pt",
        "display_lang": "pt",
        "reference_key": "por",
    },
    {
        "run_id": "m2m_12b_en_de",
        "model_id": "m2m-100-1.2b",
        "model_label": "M2M-1.2B",
        "model_short": "M2M-1.2B",
        "family": "m2m",
        "stage": "baseline",
        "params_m": 1200,
        "src_lang": "en",
        "tgt_lang": "de",
        "display_lang": "de",
        "reference_key": "deu",
    },
    {
        "run_id": "m2m_12b_en_nob",
        "model_id": "m2m-100-1.2b",
        "model_label": "M2M-1.2B",
        "model_short": "M2M-1.2B",
        "family": "m2m",
        "stage": "baseline",
        "params_m": 1200,
        "src_lang": "en",
        "tgt_lang": "no",
        "display_lang": "nob",
        "reference_key": "nob",
    },
    {
        "run_id": "m2m_12b_en_pt",
        "model_id": "m2m-100-1.2b",
        "model_label": "M2M-1.2B",
        "model_short": "M2M-1.2B",
        "family": "m2m",
        "stage": "baseline",
        "params_m": 1200,
        "src_lang": "en",
        "tgt_lang": "pt",
        "display_lang": "pt",
        "reference_key": "por",
    },
    {
        "run_id": "nllb_600_en_de",
        "model_id": "nllb-200-distilled-600m",
        "model_label": "NLLB-600M",
        "model_short": "NLLB-600M",
        "family": "nllb",
        "stage": "baseline",
        "params_m": 600,
        "src_lang": "eng_Latn",
        "tgt_lang": "deu_Latn",
        "display_lang": "de",
        "reference_key": "deu",
    },
    {
        "run_id": "nllb_600_en_nob",
        "model_id": "nllb-200-distilled-600m",
        "model_label": "NLLB-600M",
        "model_short": "NLLB-600M",
        "family": "nllb",
        "stage": "baseline",
        "params_m": 600,
        "src_lang": "eng_Latn",
        "tgt_lang": "nob_Latn",
        "display_lang": "nob",
        "reference_key": "nob",
    },
    {
        "run_id": "nllb_600_en_pt",
        "model_id": "nllb-200-distilled-600m",
        "model_label": "NLLB-600M",
        "model_short": "NLLB-600M",
        "family": "nllb",
        "stage": "baseline",
        "params_m": 600,
        "src_lang": "eng_Latn",
        "tgt_lang": "por_Latn",
        "display_lang": "pt",
        "reference_key": "por",
    },
]

OPUS_COMPARISON_PAIRS = [
    {
        "pair_id": "en-de",
        "pair_label": "EN -> DE",
        "base_model_id": "opus-mt-tc-big-en-de",
        "ft_model_id": "opus-mt-tc-big-en-de-military-v1",
    },
    {
        "pair_id": "en-nob",
        "pair_label": "EN -> NOB",
        "base_model_id": "opus-mt-tc-big-en-gmq",
        "ft_model_id": "opus-mt-tc-big-en-nob-military",
    },
    {
        "pair_id": "en-pt",
        "pair_label": "EN -> PT",
        "base_model_id": "opus-mt-tc-big-en-pt",
        "ft_model_id": "opus-mt-tc-big-en-pt-military",
    },
]

LANG_ORDER = ["de", "nob", "pt"]
BASELINE_MODEL_ORDER = ["OpusMT", "NLLB-600M", "M2M-418M", "M2M-1.2B"]
BASELINE_QUALITY_METRICS = [
    ("comet_mean", "COMET"),
    ("chrf_mean", "chrF"),
    ("bleu_mean", "BLEU"),
]
BASELINE_PERFORMANCE_METRICS = [
    ("average_pure_inference_latency_ms", "Pure Inference Latency (ms)"),
    ("cpu_percent_per_core_mean", "CPU Usage (% per core)"),
    ("baseline_rss_mb", "Average RAM Usage (MB)"),
]

MODEL_COLORS = {
    "OpusMT": "#4C78A8",
    "NLLB-600M": "#B279A2",
    "M2M-418M": "#54A24B",
    "M2M-1.2B": "#72B7B2",
}
FAMILY_COLORS = {
    "opus": "#4C78A8",
    "opus_ft": "#F58518",
    "m2m": "#54A24B",
    "nllb": "#B279A2",
}
LANG_MARKERS = {"de": "o", "nob": "s", "pt": "^"}
OPUS_LANG_ORDER = ["nob", "pt", "de"]
LANG_PAIR_LABELS = {"nob": "EN->NO", "pt": "EN->PT", "de": "EN->DE"}


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "finetune" / "data" / "TEST_SET.jsonl"
RUN_DIR = REPO_ROOT / "src" / "eval_api" / "results" / RUN_NAME
RAW_DIR = RUN_DIR / "raw"
PLOTS_DIR = RUN_DIR / "plots"
SUMMARY_CSV_PATH = RUN_DIR / "summary.csv"
SUMMARY_JSON_PATH = RUN_DIR / "summary.json"
ITEM_LEVEL_CSV_PATH = RUN_DIR / "item_level_metrics.csv"
BASELINE_SUMMARY_CSV_PATH = RUN_DIR / "baseline_summary.csv"
COMPARATIVE_TABLE_CSV_PATH = RUN_DIR / "comparative_summary_table.csv"
OPUS_COMPARISON_CSV_PATH = RUN_DIR / "opus_ft_vs_base_per_example.csv"
OPUS_TOP_EXAMPLES_CSV_PATH = RUN_DIR / "opus_top_examples_by_comet_delta.csv"


def _apply_report_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFC",
            "axes.edgecolor": "#D0D0D0",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _truncate(text: str | None, max_chars: int = 70) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


def _resolve_model_id(model_id: str) -> str:
    if not USE_CT2_FOR_ALL_MODELS:
        return model_id
    return CT2_MODEL_ID_OVERRIDES.get(model_id, model_id)


def _http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} calling {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def _load_items_by_reference_key() -> dict[str, list[dict]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    grouped: dict[str, list[dict]] = {"deu": [], "nob": [], "por": []}
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            refs = row.get("references") or {}
            for ref_key in list(grouped):
                ref_value = refs.get(ref_key)
                if not ref_value:
                    continue
                grouped[ref_key].append(
                    {
                        "item_id": row.get("item_id"),
                        "source": row["source"],
                        "reference": ref_value,
                    }
                )

    if MAX_ITEMS_PER_LANGUAGE is not None:
        for ref_key, items in grouped.items():
            grouped[ref_key] = items[:MAX_ITEMS_PER_LANGUAGE]

    return grouped


def _format_metric_value(metric_col: str, value: float) -> str:
    if pd.isna(value):
        return ""
    if metric_col == "comet_mean":
        return f"{value:.3f}"
    if metric_col in {"bleu_mean", "chrf_mean", "cpu_percent_per_core_mean"}:
        return f"{value:.1f}"
    if metric_col in {"average_pure_inference_latency_ms", "baseline_rss_mb"}:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _annotate_bar_values(ax, bars, metric_col: str) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-9)
    for bar in bars:
        value = bar.get_height()
        if pd.isna(value):
            continue
        # Skip labels for tiny bars to avoid unreadable clutter.
        if abs(value - ymin) < span * 0.05:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + span * 0.015,
            _format_metric_value(metric_col, value),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )


def _plot_grouped_baseline_metric(
    ax,
    baseline_df: pd.DataFrame,
    metric_col: str,
    title: str,
) -> None:
    model_values = baseline_df.groupby("model_label", as_index=True)[metric_col].mean()
    models = [model for model in BASELINE_MODEL_ORDER if model in model_values.index]
    if not models:
        ax.set_visible(False)
        return

    x = np.arange(len(models), dtype=float)
    values = [model_values.loc[model] for model in models]
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=[MODEL_COLORS.get(model, "#888888") for model in models],
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=16, ha="right")
    _annotate_bar_values(ax, bars, metric_col)
    ax.grid(axis="y", alpha=0.25)


def _plot_baseline_report_figure(
    summary_df: pd.DataFrame,
    output_path: Path,
    metrics_layout: list[tuple[str, str]],
    title: str,
) -> None:
    baseline_df = summary_df[summary_df["stage"] == "baseline"].copy()
    if baseline_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (metric_col, metric_title) in zip(axes.ravel(), metrics_layout):
        _plot_grouped_baseline_metric(ax, baseline_df, metric_col, metric_title)

    fig.suptitle(title, fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def _plot_opus_base_vs_finetuned_by_language(summary_df: pd.DataFrame, output_path: Path) -> None:
    opus_df = summary_df[summary_df["model_short"].isin(["OPUS-base", "OPUS-ft"])].copy()
    if opus_df.empty:
        return

    metrics = [
        ("comet_mean", "COMET"),
        ("bleu_mean", "BLEU"),
        ("chrf_mean", "chrF"),
    ]

    languages = [lang for lang in OPUS_LANG_ORDER if lang in set(opus_df["display_lang"])]
    if not languages:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(languages), dtype=float)
    width = 0.36

    for ax, (metric_col, metric_title) in zip(axes, metrics):
        base_values = []
        ft_values = []
        for lang in languages:
            lang_df = opus_df[opus_df["display_lang"] == lang]
            base_values.append(lang_df.loc[lang_df["model_short"] == "OPUS-base", metric_col].mean())
            ft_values.append(lang_df.loc[lang_df["model_short"] == "OPUS-ft", metric_col].mean())

        bars_base = ax.bar(
            x - width / 2,
            base_values,
            width=width,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.8,
            label="Opus Base",
        )
        bars_ft = ax.bar(
            x + width / 2,
            ft_values,
            width=width,
            color="#54A24B",
            edgecolor="white",
            linewidth=0.8,
            label="Opus Finetuned",
        )

        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels([LANG_PAIR_LABELS.get(lang, lang.upper()) for lang in languages])
        ax.grid(axis="y", alpha=0.25)
        _annotate_bar_values(ax, bars_base, metric_col)
        _annotate_bar_values(ax, bars_ft, metric_col)

    fig.suptitle("Opus Base vs Opus Finetuned by Language Pair", fontsize=14, y=1.05)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#4C78A8", lw=8, label="Opus Base"),
            Line2D([0], [0], color="#54A24B", lw=8, label="Opus Finetuned"),
        ],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def _build_opus_item_comparison(item_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for pair in OPUS_COMPARISON_PAIRS:
        base_df = item_df[item_df["model_id"] == pair["base_model_id"]].copy()
        ft_df = item_df[item_df["model_id"] == pair["ft_model_id"]].copy()
        if base_df.empty or ft_df.empty:
            continue

        base_df = base_df.rename(
            columns={
                "translated_value": "base_translated_value",
                "latency_ms": "base_latency_ms",
                "cpu_percent_per_core": "base_cpu_percent_per_core",
                "ram_peak_mb": "base_ram_peak_mb",
                "bleu": "base_bleu",
                "chrf": "base_chrf",
                "ter": "base_ter",
                "comet": "base_comet",
                "cometkiwi": "base_cometkiwi",
            }
        )
        ft_df = ft_df.rename(
            columns={
                "translated_value": "ft_translated_value",
                "latency_ms": "ft_latency_ms",
                "cpu_percent_per_core": "ft_cpu_percent_per_core",
                "ram_peak_mb": "ft_ram_peak_mb",
                "bleu": "ft_bleu",
                "chrf": "ft_chrf",
                "ter": "ft_ter",
                "comet": "ft_comet",
                "cometkiwi": "ft_cometkiwi",
            }
        )

        merged = pd.merge(
            base_df[
                [
                    "item_id",
                    "source",
                    "reference",
                    "display_lang",
                    "base_translated_value",
                    "base_latency_ms",
                    "base_cpu_percent_per_core",
                    "base_ram_peak_mb",
                    "base_bleu",
                    "base_chrf",
                    "base_ter",
                    "base_comet",
                    "base_cometkiwi",
                ]
            ],
            ft_df[
                [
                    "item_id",
                    "display_lang",
                    "ft_translated_value",
                    "ft_latency_ms",
                    "ft_cpu_percent_per_core",
                    "ft_ram_peak_mb",
                    "ft_bleu",
                    "ft_chrf",
                    "ft_ter",
                    "ft_comet",
                    "ft_cometkiwi",
                ]
            ],
            on=["item_id", "display_lang"],
            how="inner",
        )
        if merged.empty:
            continue

        merged["pair_id"] = pair["pair_id"]
        merged["pair_label"] = pair["pair_label"]
        merged["base_model_id"] = pair["base_model_id"]
        merged["ft_model_id"] = pair["ft_model_id"]

        merged["delta_bleu"] = merged["ft_bleu"] - merged["base_bleu"]
        merged["delta_chrf"] = merged["ft_chrf"] - merged["base_chrf"]
        merged["delta_ter"] = merged["ft_ter"] - merged["base_ter"]
        merged["delta_comet"] = merged["ft_comet"] - merged["base_comet"]
        merged["delta_latency_ms"] = merged["ft_latency_ms"] - merged["base_latency_ms"]
        merged["delta_cpu_percent_per_core"] = (
            merged["ft_cpu_percent_per_core"] - merged["base_cpu_percent_per_core"]
        )
        merged["delta_ram_peak_mb"] = merged["ft_ram_peak_mb"] - merged["base_ram_peak_mb"]

        frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _plot_opus_delta_distributions(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return

    metrics = [
        ("delta_comet", "COMET delta (FT - base)"),
        ("delta_chrf", "chrF delta (FT - base)"),
        ("delta_bleu", "BLEU delta (FT - base)"),
    ]

    pair_order = [pair["pair_id"] for pair in OPUS_COMPARISON_PAIRS]
    pair_labels = [pair["pair_label"] for pair in OPUS_COMPARISON_PAIRS]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (metric_col, title) in zip(axes, metrics):
        data = []
        for pair_id in pair_order:
            vals = comparison_df.loc[comparison_df["pair_id"] == pair_id, metric_col].dropna().values
            data.append(vals)

        ax.boxplot(
            data,
            labels=pair_labels,
            patch_artist=True,
            boxprops={"facecolor": "#C9D8FF", "edgecolor": "#5A6EAF"},
            medianprops={"color": "#1B2A6B", "linewidth": 1.8},
            whiskerprops={"color": "#5A6EAF"},
            capprops={"color": "#5A6EAF"},
        )
        for idx, vals in enumerate(data, start=1):
            if len(vals) == 0:
                continue
            jitter_x = np.random.normal(loc=idx, scale=0.04, size=len(vals))
            ax.scatter(jitter_x, vals, s=9, alpha=0.28, color="#2C2C2C")

        ax.axhline(0.0, color="#333333", linewidth=1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Fine-tuning Effect per Example (OPUS FT vs OPUS Base)", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def _plot_opus_top_examples_by_delta(comparison_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()

    pair_order = [pair["pair_id"] for pair in OPUS_COMPARISON_PAIRS]
    pair_label_map = {pair["pair_id"]: pair["pair_label"] for pair in OPUS_COMPARISON_PAIRS}

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    top_rows: list[dict] = []

    for ax, pair_id in zip(axes, pair_order):
        pair_df = comparison_df[comparison_df["pair_id"] == pair_id].copy()
        if pair_df.empty:
            ax.set_visible(False)
            continue

        pair_df["abs_delta_comet"] = pair_df["delta_comet"].abs()
        top_df = pair_df.sort_values(by="abs_delta_comet", ascending=False).head(TOP_EXAMPLES_PER_PAIR)
        if top_df.empty:
            ax.set_visible(False)
            continue

        top_df = top_df.sort_values(by="delta_comet")
        labels = [f"{row.item_id} | {_truncate(row.source, 80)}" for row in top_df.itertuples()]
        colors = ["#D62728" if v < 0 else "#2CA02C" for v in top_df["delta_comet"]]

        ax.barh(labels, top_df["delta_comet"], color=colors)
        ax.axvline(0.0, color="#333333", linewidth=1)
        ax.set_title(f"{pair_label_map[pair_id]}: Top examples by |COMET delta|")
        ax.set_xlabel("COMET delta (FT - base)")
        ax.grid(axis="x", alpha=0.25)

        for row in top_df.itertuples():
            top_rows.append(
                {
                    "pair_id": pair_id,
                    "pair_label": pair_label_map[pair_id],
                    "item_id": row.item_id,
                    "delta_comet": row.delta_comet,
                    "delta_chrf": row.delta_chrf,
                    "delta_bleu": row.delta_bleu,
                    "source": row.source,
                    "reference": row.reference,
                    "base_translation": row.base_translated_value,
                    "ft_translation": row.ft_translated_value,
                }
            )

    fig.suptitle("Example-level Translation Changes After Fine-tuning", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)

    if not top_rows:
        return pd.DataFrame()

    top_df = pd.DataFrame(top_rows)
    top_df = top_df.sort_values(by=["pair_id", "delta_comet"], ascending=[True, False]).reset_index(drop=True)
    return top_df


def _plot_comparative_scatter(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return

    latency_col = "average_latency_ms"
    latency_axis_label = "Average profiled latency (ms)"
    if "average_pure_inference_latency_ms" in summary_df.columns:
        pure_vals = summary_df["average_pure_inference_latency_ms"].dropna()
        if not pure_vals.empty:
            latency_col = "average_pure_inference_latency_ms"
            latency_axis_label = "Average pure inference latency (ms)"

    agg_df = (
        summary_df.groupby(["model_short", "model_label", "family", "params_m"], as_index=False)
        .agg(
            {
                latency_col: "mean",
                "comet_mean": "mean",
            }
        )
        .dropna(subset=[latency_col, "comet_mean"])
    )
    if agg_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12.5, 8))

    size_scale = 0.22
    x_vals = agg_df[latency_col].to_numpy()
    y_vals = agg_df["comet_mean"].to_numpy()
    x_span = float(np.ptp(x_vals)) if len(x_vals) > 0 else 1.0
    y_span = float(np.ptp(y_vals)) if len(y_vals) > 0 else 1.0
    if x_span <= 0:
        x_span = max(float(np.max(x_vals)) * 0.01, 1.0) if len(x_vals) > 0 else 1.0
    if y_span <= 0:
        y_span = 0.01

    # Assign different text offsets for points that are close in x to reduce label overlap.
    close_x_threshold = 0.06 * x_span
    sorted_indices = np.argsort(x_vals)
    label_offsets_by_idx: dict[int, tuple[int, int]] = {}
    cluster: list[int] = []
    last_x: float | None = None
    offset_cycle = [(6, 10), (6, -12), (6, 22), (-52, 10), (-52, -12)]
    for idx in sorted_indices:
        x = x_vals[idx]
        if last_x is None or abs(x - last_x) <= close_x_threshold:
            cluster.append(int(idx))
        else:
            for pos, c_idx in enumerate(cluster):
                label_offsets_by_idx[c_idx] = offset_cycle[pos % len(offset_cycle)]
            cluster = [int(idx)]
        last_x = x
    for pos, c_idx in enumerate(cluster):
        label_offsets_by_idx[c_idx] = offset_cycle[pos % len(offset_cycle)]

    for i, row in enumerate(agg_df.itertuples()):
        size = max(45, float(row.params_m) * size_scale)
        color = FAMILY_COLORS.get(row.family, "#666666")
        latency_value = getattr(row, latency_col)
        ax.scatter(
            latency_value,
            row.comet_mean,
            s=size,
            color=color,
            marker="o",
            alpha=0.82,
            edgecolors="white",
            linewidth=0.8,
        )
        label = row.model_short
        dx, dy = label_offsets_by_idx.get(i, (6, 8))
        if row.comet_mean > (float(np.max(y_vals)) - 0.12 * y_span):
            dy = min(dy, -10)
        elif row.comet_mean < (float(np.min(y_vals)) + 0.12 * y_span):
            dy = max(dy, 10)
        if latency_value > (float(np.max(x_vals)) - 0.12 * x_span):
            dx = min(dx, -56)
        ax.annotate(
            label,
            (latency_value, row.comet_mean),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
        )

    latency_values = agg_df[latency_col].dropna()
    if not latency_values.empty and latency_values.max() > 0 and latency_values.min() > 0:
        if latency_values.max() / latency_values.min() > 8:
            ax.set_xscale("log")

    ax.set_title("Comparative Summary: Latency vs Accuracy (aggregated across EN->NO/PT/DE)")
    ax.set_xlabel(latency_axis_label)
    ax.set_ylabel("COMET mean")
    ax.grid(alpha=0.25)

    family_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=family)
        for family, color in FAMILY_COLORS.items()
        if family in set(agg_df["family"])
    ]
    size_handles = [
        plt.scatter([], [], s=max(45, p * size_scale), color="#999999", alpha=0.5, label=f"{p}M")
        for p in (100, 600, 1200)
    ]

    legend_family = ax.legend(
        handles=family_handles,
        title="Model family",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.add_artist(legend_family)
    ax.legend(
        handles=size_handles,
        title="Parameters",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)


def main() -> None:
    _apply_report_style()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    item_df = pd.DataFrame()
    if REUSE_EXISTING_RESULTS and SUMMARY_CSV_PATH.exists():
        summary_df = pd.read_csv(SUMMARY_CSV_PATH)
        if ITEM_LEVEL_CSV_PATH.exists():
            item_df = pd.read_csv(ITEM_LEVEL_CSV_PATH)
        print(f"Reusing existing evaluation outputs from: {RUN_DIR}")
    else:
        _http_json("GET", f"{API_BASE_URL}/models")

        items_by_ref = _load_items_by_reference_key()
        summary_rows: list[dict] = []
        item_rows: list[dict] = []

        for run in MODEL_RUNS:
            items = items_by_ref.get(run["reference_key"], [])
            if not items:
                continue
            resolved_model_id = _resolve_model_id(run["model_id"])

            payload = {
                "model_id": resolved_model_id,
                "src_lang": run["src_lang"],
                "tgt_lang": run["tgt_lang"],
                "metrics": METRICS,
                "items": items,
            }
            response = _http_json("POST", f"{API_BASE_URL}/evaluate", payload)

            raw_name = f"{run['run_id']}__{_slug(resolved_model_id)}.json"
            (RAW_DIR / raw_name).write_text(
                json.dumps(response, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            aggregates = response.get("aggregates") or {}
            summary_rows.append(
                {
                    "run_id": run["run_id"],
                    "model_id": resolved_model_id,
                    "model_label": run["model_label"],
                    "model_short": run["model_short"],
                    "family": run["family"],
                    "stage": run["stage"],
                    "params_m": run["params_m"],
                    "src_lang": run["src_lang"],
                    "tgt_lang": run["tgt_lang"],
                    "display_lang": run["display_lang"],
                    "reference_key": run["reference_key"],
                    "num_items": len(items),
                    "average_latency_ms": response.get("average_latency_ms"),
                    "average_pure_inference_latency_ms": response.get("average_pure_inference_latency_ms"),
                    "baseline_rss_mb": response.get("baseline_rss_mb"),
                    "bleu_mean": aggregates.get("bleu_mean"),
                    "chrf_mean": aggregates.get("chrf_mean"),
                    "ter_mean": aggregates.get("ter_mean"),
                    "comet_mean": aggregates.get("comet_mean"),
                    "cometkiwi_mean": aggregates.get("cometkiwi_mean"),
                    "cpu_percent_per_core_mean": aggregates.get("cpu_percent_per_core_mean"),
                    "ram_peak_mb_mean": aggregates.get("ram_peak_mb_mean"),
                    "total_tokens_per_second_mean": aggregates.get("total_tokens_per_second_mean"),
                    "output_tokens_per_second_mean": aggregates.get("output_tokens_per_second_mean"),
                    "latency_p95_ms": aggregates.get("latency_p95_ms"),
                    "latency_p99_ms": aggregates.get("latency_p99_ms"),
                    "pure_inference_latency_p95_ms": aggregates.get("pure_inference_latency_p95_ms"),
                    "pure_inference_latency_p99_ms": aggregates.get("pure_inference_latency_p99_ms"),
                }
            )

            for result in response.get("results", []):
                result_metrics = result.get("metrics") or {}
                item_rows.append(
                    {
                        "run_id": run["run_id"],
                        "model_id": resolved_model_id,
                        "model_label": run["model_label"],
                        "family": run["family"],
                        "stage": run["stage"],
                        "params_m": run["params_m"],
                        "display_lang": run["display_lang"],
                        "src_lang": run["src_lang"],
                        "tgt_lang": run["tgt_lang"],
                        "reference_key": run["reference_key"],
                        "item_id": result.get("item_id"),
                        "source": result.get("source"),
                        "reference": result.get("reference"),
                        "translated_value": result.get("translated_value"),
                        "latency_ms": result.get("latency_ms"),
                        "pure_inference_latency_ms": result.get("pure_inference_latency_ms"),
                        "cpu_percent_per_core": result.get("cpu_percent_per_core"),
                        "ram_peak_mb": result.get("ram_peak_mb"),
                        "bleu": result_metrics.get("bleu"),
                        "chrf": result_metrics.get("chrf"),
                        "ter": result_metrics.get("ter"),
                        "comet": result_metrics.get("comet"),
                        "cometkiwi": result_metrics.get("cometkiwi"),
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        if summary_df.empty:
            raise RuntimeError("No evaluation rows were produced.")

        summary_df = summary_df.sort_values(by=["display_lang", "stage", "model_label", "model_id"]).reset_index(drop=True)
        summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
        SUMMARY_JSON_PATH.write_text(
            summary_df.to_json(orient="records", indent=2, force_ascii=False),
            encoding="utf-8",
        )

        item_df = pd.DataFrame(item_rows)
        if not item_df.empty:
            item_df = item_df.sort_values(by=["display_lang", "model_id", "item_id"]).reset_index(drop=True)
            item_df.to_csv(ITEM_LEVEL_CSV_PATH, index=False)

    if summary_df.empty:
        raise RuntimeError(f"Summary is empty: {SUMMARY_CSV_PATH}")

    summary_df = summary_df.sort_values(by=["display_lang", "stage", "model_label", "model_id"]).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    SUMMARY_JSON_PATH.write_text(summary_df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")

    comparative_table_df = summary_df[
        [
            "model_label",
            "stage",
            "display_lang",
            "params_m",
            "comet_mean",
            "chrf_mean",
            "bleu_mean",
            "average_latency_ms",
            "average_pure_inference_latency_ms",
            "cpu_percent_per_core_mean",
            "ram_peak_mb_mean",
            "model_id",
        ]
    ].copy()
    comparative_table_df = comparative_table_df.sort_values(by=["display_lang", "stage", "comet_mean"], ascending=[True, True, False])
    comparative_table_df.to_csv(COMPARATIVE_TABLE_CSV_PATH, index=False)

    if not item_df.empty:
        item_df = item_df.sort_values(by=["display_lang", "model_id", "item_id"]).reset_index(drop=True)
        item_df.to_csv(ITEM_LEVEL_CSV_PATH, index=False)

    baseline_df = summary_df[summary_df["stage"] == "baseline"].copy()
    if not baseline_df.empty:
        baseline_df = baseline_df.sort_values(by=["display_lang", "model_label"]).reset_index(drop=True)
        baseline_df.to_csv(BASELINE_SUMMARY_CSV_PATH, index=False)

    _plot_baseline_report_figure(
        summary_df,
        PLOTS_DIR / "01_baseline_quality_comparison.png",
        BASELINE_QUALITY_METRICS,
        "Baseline Comparison: Translation Quality (Aggregated Across EN->NO/PT/DE)",
    )
    _plot_baseline_report_figure(
        summary_df,
        PLOTS_DIR / "02_baseline_performance_comparison.png",
        BASELINE_PERFORMANCE_METRICS,
        "Baseline Comparison: Runtime Performance (Aggregated Across EN->NO/PT/DE)",
    )
    _plot_opus_base_vs_finetuned_by_language(
        summary_df,
        PLOTS_DIR / "03_opus_base_vs_finetuned_by_language.png",
    )
    _plot_comparative_scatter(summary_df, PLOTS_DIR / "05_comparative_scatter_latency_comet_params.png")
    legacy_delta_plot = PLOTS_DIR / "03_finetune_delta_distributions.png"
    if legacy_delta_plot.exists():
        legacy_delta_plot.unlink()

    if not item_df.empty:
        opus_comparison_df = _build_opus_item_comparison(item_df)
        if not opus_comparison_df.empty:
            opus_comparison_df = opus_comparison_df.sort_values(by=["pair_id", "item_id"]).reset_index(drop=True)
            opus_comparison_df.to_csv(OPUS_COMPARISON_CSV_PATH, index=False)
            top_examples_df = _plot_opus_top_examples_by_delta(
                opus_comparison_df,
                PLOTS_DIR / "04_finetune_top_examples_by_comet_delta.png",
            )
            if not top_examples_df.empty:
                top_examples_df.to_csv(OPUS_TOP_EXAMPLES_CSV_PATH, index=False)

    print(f"Saved summary: {SUMMARY_CSV_PATH}")
    print(f"Saved baseline summary: {BASELINE_SUMMARY_CSV_PATH}")
    print(f"Saved comparative table: {COMPARATIVE_TABLE_CSV_PATH}")
    print(f"Saved item-level metrics: {ITEM_LEVEL_CSV_PATH}")
    print(f"Saved OPUS per-example comparison: {OPUS_COMPARISON_CSV_PATH}")
    print(f"Saved top examples: {OPUS_TOP_EXAMPLES_CSV_PATH}")
    print(f"Saved plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
