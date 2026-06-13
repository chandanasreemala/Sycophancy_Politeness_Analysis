# """
# Sample analysis script for creating plots from metrics.

# This template shows how to load the metrics CSV and create informative
# visualizations. Customize as needed for your analysis!

# Usage:
#     python create_plots.py --model-name google_gemma-2-2b-it --results-dir results
# """

# import argparse
# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# def load_metrics(model_name: str, results_dir: str) -> pd.DataFrame:
#     """Load metrics CSV for a specific model."""
#     metrics_file = Path(results_dir) / model_name / "metrics" / "metrics.csv"
#     if not metrics_file.exists():
#         raise FileNotFoundError(f"Metrics file not found at {metrics_file}")
#     return pd.read_csv(metrics_file)


# def plot_agreement_by_politeness(df: pd.DataFrame, output_dir: Path) -> None:
#     """Plot agreement rates by politeness level."""
#     # Filter out baseline
#     df_filtered = df[df["prompt_type"] != "baseline"].copy()
    
#     # Compute agreement rates
#     agreement_stats = df_filtered.groupby("politeness_level").apply(
#         lambda x: (x["judge_agreement"] == "agree").sum() / len(x) * 100
#     ).reindex(["impolite", "neutral", "polite"])
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     agreement_stats.plot(kind="bar", ax=ax, color=["#d62728", "#ff7f0e", "#2ca02c"])
#     ax.set_title("Agreement Rate by Politeness Level", fontsize=14, fontweight="bold")
#     ax.set_xlabel("Politeness Level", fontsize=12)
#     ax.set_ylabel("Agreement Rate (%)", fontsize=12)
#     ax.set_xticklabels(["Impolite", "Neutral", "Polite"], rotation=0)
#     ax.grid(axis="y", alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(output_dir / "01_agreement_by_politeness.png", dpi=150)
#     plt.close()
#     print(f"Saved: 01_agreement_by_politeness.png")


# def plot_sycophancy_cases(df: pd.DataFrame, output_dir: Path) -> None:
#     """Plot sycophancy cases (user wrong + model agrees) by politeness."""
#     # Filter to only user_wrong cases
#     df_wrong = df[df["answer_type"] == "incorrect"].copy()
    
#     # Count agreements by politeness
#     sycophancy_stats = df_wrong.groupby("politeness_level").apply(
#         lambda x: (x["judge_agreement"] == "agree").sum()
#     ).reindex(["impolite", "neutral", "polite"])
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sycophancy_stats.plot(kind="bar", ax=ax, color=["#d62728", "#ff7f0e", "#2ca02c"])
#     ax.set_title("Sycophancy Cases: Model Agrees with Wrong Claims", fontsize=14, fontweight="bold")
#     ax.set_xlabel("Politeness Level", fontsize=12)
#     ax.set_ylabel("Count of Sycophancy Cases", fontsize=12)
#     ax.set_xticklabels(["Impolite", "Neutral", "Polite"], rotation=0)
#     ax.grid(axis="y", alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(output_dir / "02_sycophancy_cases.png", dpi=150)
#     plt.close()
#     print(f"Saved: 02_sycophancy_cases.png")


# def plot_agreement_buckets(df: pd.DataFrame, output_dir: Path) -> None:
#     """Plot distribution of agreement buckets."""
#     # Filter out unknown and baseline
#     bucket_counts = df[
#         (df["agreement_bucket"] != "unknown") & 
#         (df["agreement_bucket"] != "baseline_no_claim")
#     ]["agreement_bucket"].value_counts()
    
#     bucket_order = [
#         "user_correct_model_agree",
#         "user_correct_model_disagree", 
#         "user_wrong_model_agree",
#         "user_wrong_model_disagree",
#     ]
#     bucket_counts = bucket_counts.reindex([b for b in bucket_order if b in bucket_counts.index])
    
#     labels = [
#         "User Correct\n+ Model Agrees",
#         "User Correct\n+ Model Disagrees",
#         "User Wrong\n+ Model Agrees",
#         "User Wrong\n+ Model Disagrees",
#     ]
    
#     fig, ax = plt.subplots(figsize=(12, 6))
#     colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
#     bucket_counts.plot(kind="bar", ax=ax, color=colors[:len(bucket_counts)])
#     ax.set_title("Distribution of Agreement Cases", fontsize=14, fontweight="bold")
#     ax.set_xlabel("Case Type", fontsize=12)
#     ax.set_ylabel("Count", fontsize=12)
#     ax.set_xticklabels(labels[:len(bucket_counts)], rotation=45, ha="right")
#     ax.grid(axis="y", alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(output_dir / "03_agreement_buckets.png", dpi=150)
#     plt.close()
#     print(f"Saved: 03_agreement_buckets.png")


# def plot_judge_politeness_distribution(df: pd.DataFrame, output_dir: Path) -> None:
#     """Plot distribution of judge's politeness assessments by actual politeness level."""
#     df_filtered = df[df["prompt_type"] != "baseline"].copy()
    
#     # Create crosstab
#     crosstab = pd.crosstab(df_filtered["politeness_level"], df_filtered["judge_politeness"])
#     crosstab = crosstab.reindex(["impolite", "neutral", "polite"], fill_value=0)
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     crosstab.plot(kind="bar", ax=ax, width=0.8)
#     ax.set_title("Judge's Politeness Assessment vs Actual Politeness", fontsize=14, fontweight="bold")
#     ax.set_xlabel("Actual Politeness Level (System Prompt)", fontsize=12)
#     ax.set_ylabel("Count", fontsize=12)
#     ax.set_xticklabels(["Impolite", "Neutral", "Polite"], rotation=0)
#     ax.legend(title="Judge's Assessment", bbox_to_anchor=(1.05, 1), loc="upper left")
#     ax.grid(axis="y", alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(output_dir / "04_judge_politeness_assessment.png", dpi=150)
#     plt.close()
#     print(f"Saved: 04_judge_politeness_assessment.png")


# def plot_metrics_comparison(df: pd.DataFrame, output_dir: Path) -> None:
#     """Plot answer validation metrics by politeness level."""
#     df_filtered = df[df["prompt_type"] != "baseline"].copy()
    
#     metrics_by_politeness = df_filtered.groupby("politeness_level")[
#         ["exact_match", "f1_score", "normalized_similarity"]
#     ].mean().reindex(["impolite", "neutral", "polite"])
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     metrics_by_politeness.plot(kind="bar", ax=ax, width=0.8)
#     ax.set_title("Answer Validation Metrics by Politeness Level", fontsize=14, fontweight="bold")
#     ax.set_xlabel("Politeness Level", fontsize=12)
#     ax.set_ylabel("Score", fontsize=12)
#     ax.set_xticklabels(["Impolite", "Neutral", "Polite"], rotation=0)
#     ax.legend(["Exact Match", "F1 Score (tokens)", "Normalized Similarity"], loc="upper left")
#     ax.set_ylim(0, 1)
#     ax.grid(axis="y", alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(output_dir / "05_answer_metrics.png", dpi=150)
#     plt.close()
#     print(f"Saved: 05_answer_metrics.png")


# def main():
#     parser = argparse.ArgumentParser(description="Create analysis plots from metrics CSV")
#     parser.add_argument(
#         "--model-name",
#         type=str,
#         required=True,
#         help="Model name (format: google_gemma-2-2b-it, etc.)",
#     )
#     parser.add_argument(
#         "--results-dir",
#         type=str,
#         default="results",
#         help="Results directory path",
#     )
    
#     args = parser.parse_args()
    
#     # Load metrics
#     print(f"Loading metrics for {args.model_name}...")
#     df = load_metrics(args.model_name, args.results_dir)
#     print(f"Loaded {len(df)} rows\n")
    
#     # Create output directory
#     output_dir = Path(args.results_dir) / args.model_name / "plots"
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Generate plots
#     print("Creating plots...")
#     plot_agreement_by_politeness(df, output_dir)
#     plot_sycophancy_cases(df, output_dir)
#     plot_agreement_buckets(df, output_dir)
#     plot_judge_politeness_distribution(df, output_dir)
#     plot_metrics_comparison(df, output_dir)
    
#     print(f"\n✅ All plots saved to: {output_dir}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Create focused plots for sycophancy / politeness analysis.

Expected input columns:
  - judge_politeness
  - judge_agreement
  - politeness_level
  - prompt_type

The script creates the exact requested views:
  01. user_correct/user_incorrect vs model agree/disagree
  02. user_correct + model agree by politeness_level
  03. user_correct + model disagree by politeness_level
  04. user_incorrect + model disagree by politeness_level
  05. user_incorrect + model agree by politeness_level
  06. baseline filter audit: only baseline + neutral is used as baseline
  07a. baseline vs user_correct/user_incorrect agreement/disagreement curves
  07b. baseline vs user_correct/user_incorrect model response politeness curves
  07. combined baseline comparison figure

Notes:
  - For agreement/disagreement rates, the default denominator is rows where
    judge_agreement is either agree or disagree. judge_agreement values like
    neutral/unknown are excluded from those rates but counted in diagnostics.
  - For baseline comparisons, only rows with prompt_type=baseline and
    politeness_level=neutral are used. Baseline + polite and baseline + impolite
    are ignored for baseline curves.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


REQUIRED_COLUMNS = {"judge_politeness", "judge_agreement", "politeness_level", "prompt_type"}

PROMPT_CORRECT = "user_correct_claim"
PROMPT_INCORRECT = "user_incorrect_claim"
PROMPT_BASELINE = "baseline"
PROMPT_ORDER = [PROMPT_CORRECT, PROMPT_INCORRECT]
PROMPT_LABELS = {
    PROMPT_CORRECT: "User correct claim",
    PROMPT_INCORRECT: "User incorrect claim",
    PROMPT_BASELINE: "Baseline",
}

POLITENESS_ORDER = ["impolite", "neutral", "polite"]
POLITENESS_LABELS = {
    "impolite": "Impolite",
    "neutral": "Neutral",
    "polite": "Polite",
}

AGREEMENT_ORDER = ["agree", "disagree"]
AGREEMENT_LABELS = {
    "agree": "Model agrees",
    "disagree": "Model disagrees",
}

RESPONSE_POLITENESS_ORDER = ["impolite", "neutral", "polite"]
RESPONSE_POLITENESS_LABELS = {
    "impolite": "Impolite response",
    "neutral": "Neutral response",
    "polite": "Polite response",
}


# Matplotlib settings. These are intentionally local to this script.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
})


COLOR_AGREE = "#2a9d8f"
COLOR_DISAGREE = "#e76f51"
COLOR_IMPOLITE = "#d62828"
COLOR_NEUTRAL = "#6c757d"
COLOR_POLITE = "#2a9d8f"
COLOR_BASELINE = "#4c78a8"
COLOR_RATE = "#5e60ce"
COLOR_GRID = "#e9ecef"

POLITENESS_COLORS = {
    "impolite": COLOR_IMPOLITE,
    "neutral": COLOR_NEUTRAL,
    "polite": COLOR_POLITE,
}
AGREEMENT_COLORS = {
    "agree": COLOR_AGREE,
    "disagree": COLOR_DISAGREE,
}


def normalize_column_name(name: str) -> str:
    """Make column names robust to spaces/case/hyphens."""
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text if text else "unknown"


def normalize_prompt_type(value: object) -> str:
    v = normalize_value(value)
    aliases = {
        "baseline": PROMPT_BASELINE,
        "base_line": PROMPT_BASELINE,
        "base": PROMPT_BASELINE,
        "neutral_baseline": PROMPT_BASELINE,
        "user_correct_claim": PROMPT_CORRECT,
        "correct_claim": PROMPT_CORRECT,
        "user_correct": PROMPT_CORRECT,
        "correct": PROMPT_CORRECT,
        "true_claim": PROMPT_CORRECT,
        "user_incorrect_claim": PROMPT_INCORRECT,
        "incorrect_claim": PROMPT_INCORRECT,
        "user_incorrect": PROMPT_INCORRECT,
        "incorrect": PROMPT_INCORRECT,
        "false_claim": PROMPT_INCORRECT,
    }
    return aliases.get(v, v)


def normalize_politeness(value: object) -> str:
    v = normalize_value(value)
    aliases = {
        "polite": "polite",
        "very_polite": "polite",
        "courteous": "polite",
        "neutral": "neutral",
        "neither": "neutral",
        "impolite": "impolite",
        "rude": "impolite",
        "discourteous": "impolite",
        "unknown": "unknown",
        "unclear": "unknown",
        "na": "unknown",
        "none": "unknown",
    }
    return aliases.get(v, v)


def normalize_agreement(value: object) -> str:
    v = normalize_value(value)
    aliases = {
        "agree": "agree",
        "agrees": "agree",
        "agreement": "agree",
        "model_agrees": "agree",
        "yes": "agree",
        "disagree": "disagree",
        "disagrees": "disagree",
        "disagreement": "disagree",
        "model_disagrees": "disagree",
        "no": "disagree",
        "neutral": "neutral",
        "neither": "neutral",
        "unknown": "unknown",
        "unclear": "unknown",
        "na": "unknown",
        "none": "unknown",
    }
    return aliases.get(v, v)


def pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * value:.1f}%"


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_and_prepare(input_file: str | Path) -> pd.DataFrame:
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Input CSV is missing required columns after normalization: "
            f"{sorted(missing)}. Available columns: {list(df.columns)}"
        )

    df["prompt_type"] = df["prompt_type"].map(normalize_prompt_type)
    df["politeness_level"] = df["politeness_level"].map(normalize_politeness)
    df["judge_politeness"] = df["judge_politeness"].map(normalize_politeness)
    df["judge_agreement"] = df["judge_agreement"].map(normalize_agreement)

    return df


def is_valid_agreement(series: pd.Series) -> pd.Series:
    return series.isin(AGREEMENT_ORDER)


def is_valid_response_politeness(series: pd.Series) -> pd.Series:
    return series.isin(RESPONSE_POLITENESS_ORDER)


def wilson_ci(success: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson confidence interval for a binomial proportion."""
    if total <= 0:
        return (np.nan, np.nan)
    phat = success / total
    denom = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denom
    half_width = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denom
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def condition_df_for_agreement(df: pd.DataFrame, prompt_type: str, politeness_level: Optional[str] = None) -> pd.DataFrame:
    sub = df[df["prompt_type"].eq(prompt_type)].copy()
    if politeness_level is not None:
        sub = sub[sub["politeness_level"].eq(politeness_level)]
    return sub[sub["judge_agreement"].isin(AGREEMENT_ORDER)].copy()


def add_percent_axis(ax: plt.Axes, y_max: float = 1.0) -> None:
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", color=COLOR_GRID, linewidth=1)


def save_fig(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def graph1_correctness_agreement_heatmap(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Graph 1: user_correct/user_incorrect vs model agree/disagree."""
    valid = df[
        df["prompt_type"].isin(PROMPT_ORDER)
        & df["judge_agreement"].isin(AGREEMENT_ORDER)
    ].copy()

    rows = []
    for prompt_type in PROMPT_ORDER:
        sub = valid[valid["prompt_type"].eq(prompt_type)]
        total = len(sub)
        for agreement in AGREEMENT_ORDER:
            count = int(sub["judge_agreement"].eq(agreement).sum())
            rate = count / total if total else np.nan
            rows.append({
                "prompt_type": prompt_type,
                "prompt_label": PROMPT_LABELS[prompt_type],
                "judge_agreement": agreement,
                "agreement_label": AGREEMENT_LABELS[agreement],
                "count": count,
                "valid_total": total,
                "rate_among_agree_disagree": rate,
            })
    summary = pd.DataFrame(rows)

    matrix = summary.pivot(index="prompt_label", columns="agreement_label", values="rate_among_agree_disagree")
    counts = summary.pivot(index="prompt_label", columns="agreement_label", values="count")
    totals = summary.groupby("prompt_label")["valid_total"].first()

    # Fixed order for readability.
    row_labels = [PROMPT_LABELS[p] for p in PROMPT_ORDER]
    col_labels = [AGREEMENT_LABELS[a] for a in AGREEMENT_ORDER]
    matrix = matrix.loc[row_labels, col_labels]
    counts = counts.loc[row_labels, col_labels]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(matrix.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    cbar.set_label("Share within valid agree/disagree rows")

    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title("Claim correctness vs model agreement/disagreement")
    ax.set_xlabel("Model agreement judgement")
    ax.set_ylabel("User claim type")

    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            rate = matrix.loc[row, col]
            count = int(counts.loc[row, col])
            total = int(totals.loc[row])
            ax.text(j, i, f"{pct(rate)}\n{count}/{total}", ha="center", va="center", fontsize=11)

    ax.text(
        0.0,
        -0.22,
        "Denominator excludes judge_agreement values outside agree/disagree.",
        transform=ax.transAxes,
        fontsize=9,
        color="dimgray",
    )
    save_fig(fig, output_dir / "01_claim_correctness_vs_model_agreement_heatmap.png")
    return summary


def build_politeness_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_type in PROMPT_ORDER:
        for politeness_level in POLITENESS_ORDER:
            sub = condition_df_for_agreement(df, prompt_type, politeness_level)
            total = len(sub)
            for agreement in AGREEMENT_ORDER:
                success = int(sub["judge_agreement"].eq(agreement).sum())
                rate = success / total if total else np.nan
                lower, upper = wilson_ci(success, total)
                rows.append({
                    "prompt_type": prompt_type,
                    "prompt_label": PROMPT_LABELS[prompt_type],
                    "politeness_level": politeness_level,
                    "politeness_label": POLITENESS_LABELS[politeness_level],
                    "target_agreement": agreement,
                    "target_label": AGREEMENT_LABELS[agreement],
                    "count": success,
                    "valid_total": total,
                    "rate_among_agree_disagree": rate,
                    "wilson_lower": lower,
                    "wilson_upper": upper,
                })
    return pd.DataFrame(rows)


def plot_condition_rate(
    summary: pd.DataFrame,
    prompt_type: str,
    target_agreement: str,
    title: str,
    output_path: Path,
) -> None:
    sub = summary[
        summary["prompt_type"].eq(prompt_type)
        & summary["target_agreement"].eq(target_agreement)
    ].copy()
    sub["politeness_level"] = pd.Categorical(sub["politeness_level"], POLITENESS_ORDER, ordered=True)
    sub = sub.sort_values("politeness_level")

    x = np.arange(len(POLITENESS_ORDER))
    y = sub["rate_among_agree_disagree"].to_numpy(dtype=float)
    lower = sub["wilson_lower"].to_numpy(dtype=float)
    upper = sub["wilson_upper"].to_numpy(dtype=float)
    yerr = np.vstack([y - lower, upper - y])

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    color = AGREEMENT_COLORS[target_agreement]
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        markersize=9,
        linewidth=2.5,
        capsize=5,
        color=color,
        ecolor=color,
    )
    ax.plot(x, y, linewidth=2.5, color=color, alpha=0.75)
    ax.fill_between(x, lower, upper, color=color, alpha=0.10)

    for xi, row in zip(x, sub.itertuples(index=False)):
        if pd.isna(row.rate_among_agree_disagree):
            label = "NA"
        else:
            label = f"{pct(row.rate_among_agree_disagree)}\n{row.count}/{row.valid_total}"
        ax.annotate(label, (xi, row.rate_among_agree_disagree if not pd.isna(row.rate_among_agree_disagree) else 0),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)

    ax.set_xticks(x, [POLITENESS_LABELS[p] for p in POLITENESS_ORDER])
    ax.set_xlabel("Prompt politeness level")
    ax.set_ylabel(f"Share: {AGREEMENT_LABELS[target_agreement].lower()}")
    add_percent_axis(ax)
    ax.set_title(title)
    ax.text(
        0.0,
        -0.18,
        "Point labels are count/valid_total. Shaded band is Wilson 95% confidence interval.",
        transform=ax.transAxes,
        fontsize=9,
        color="dimgray",
    )
    save_fig(fig, output_path)


def graph2_to_5_politeness_condition_plots(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = build_politeness_condition_summary(df)

    plot_condition_rate(
        summary,
        prompt_type=PROMPT_CORRECT,
        target_agreement="agree",
        title="User correct claim + model agrees, by prompt politeness",
        output_path=output_dir / "02_user_correct_model_agree_by_politeness.png",
    )
    plot_condition_rate(
        summary,
        prompt_type=PROMPT_CORRECT,
        target_agreement="disagree",
        title="User correct claim + model disagrees, by prompt politeness",
        output_path=output_dir / "03_user_correct_model_disagree_by_politeness.png",
    )
    plot_condition_rate(
        summary,
        prompt_type=PROMPT_INCORRECT,
        target_agreement="disagree",
        title="User incorrect claim + model disagrees, by prompt politeness",
        output_path=output_dir / "04_user_incorrect_model_disagree_by_politeness.png",
    )
    # The user repeated user_correct + agree in bullet 5. To cover all four logical combinations,
    # this plot is user_incorrect + agree.
    plot_condition_rate(
        summary,
        prompt_type=PROMPT_INCORRECT,
        target_agreement="agree",
        title="User incorrect claim + model agrees, by prompt politeness",
        output_path=output_dir / "05_user_incorrect_model_agree_by_politeness.png",
    )
    return summary


def baseline_filter_audit(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    baseline = df[df["prompt_type"].eq(PROMPT_BASELINE)].copy()
    rows = []
    for level in POLITENESS_ORDER:
        count = int(baseline["politeness_level"].eq(level).sum())
        rows.append({
            "prompt_type": PROMPT_BASELINE,
            "politeness_level": level,
            "count": count,
            "used_as_baseline": level == "neutral",
            "baseline_rule": "kept" if level == "neutral" else "ignored",
        })
    audit = pd.DataFrame(rows)

    # Plot as a small audit matrix/table: kept vs ignored.
    plot_data = np.array([
        [int(audit.query("politeness_level == 'impolite'")["count"].iloc[0]), 0],
        [0, int(audit.query("politeness_level == 'neutral'")["count"].iloc[0])],
        [int(audit.query("politeness_level == 'polite'")["count"].iloc[0]), 0],
    ], dtype=float)
    row_labels = [POLITENESS_LABELS[p] for p in POLITENESS_ORDER]
    col_labels = ["Ignored for baseline", "Used as baseline"]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    im = ax.imshow(plot_data, cmap="Greys", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Rows")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title("Baseline rule audit: only baseline + neutral is used")
    ax.set_xlabel("Baseline treatment")
    ax.set_ylabel("politeness_level among prompt_type=baseline")

    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            value = int(plot_data[i, j])
            if value > 0:
                ax.text(j, i, f"{value}", ha="center", va="center", fontsize=12, fontweight="bold")
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=11, color="gray")

    ax.text(
        0.0,
        -0.22,
        "All baseline+polite and baseline+impolite rows are excluded from baseline comparison curves.",
        transform=ax.transAxes,
        fontsize=9,
        color="dimgray",
    )
    save_fig(fig, output_dir / "06_baseline_filter_audit.png")
    return audit


def get_comparison_subsets(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    baseline_neutral = df[
        df["prompt_type"].eq(PROMPT_BASELINE)
        & df["politeness_level"].eq("neutral")
    ].copy()
    user_correct = df[df["prompt_type"].eq(PROMPT_CORRECT)].copy()
    user_incorrect = df[df["prompt_type"].eq(PROMPT_INCORRECT)].copy()
    return [
        ("Baseline\n(neutral only)", baseline_neutral),
        ("User correct\nclaim", user_correct),
        ("User incorrect\nclaim", user_incorrect),
    ]


def build_baseline_agreement_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_label, sub in get_comparison_subsets(df):
        valid = sub[sub["judge_agreement"].isin(AGREEMENT_ORDER)]
        total = len(valid)
        for agreement in AGREEMENT_ORDER:
            count = int(valid["judge_agreement"].eq(agreement).sum())
            rate = count / total if total else np.nan
            lower, upper = wilson_ci(count, total)
            rows.append({
                "condition": condition_label.replace("\n", " "),
                "judge_agreement": agreement,
                "agreement_label": AGREEMENT_LABELS[agreement],
                "count": count,
                "valid_total": total,
                "rate_among_agree_disagree": rate,
                "wilson_lower": lower,
                "wilson_upper": upper,
            })
    return pd.DataFrame(rows)


def build_baseline_politeness_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_label, sub in get_comparison_subsets(df):
        valid = sub[sub["judge_politeness"].isin(RESPONSE_POLITENESS_ORDER)]
        total = len(valid)
        for style in RESPONSE_POLITENESS_ORDER:
            count = int(valid["judge_politeness"].eq(style).sum())
            rate = count / total if total else np.nan
            lower, upper = wilson_ci(count, total)
            rows.append({
                "condition": condition_label.replace("\n", " "),
                "judge_politeness": style,
                "politeness_label": RESPONSE_POLITENESS_LABELS[style],
                "count": count,
                "valid_total": total,
                "rate_among_valid_politeness": rate,
                "wilson_lower": lower,
                "wilson_upper": upper,
            })
    return pd.DataFrame(rows)


def plot_baseline_agreement_curves(summary: pd.DataFrame, output_path: Path, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
    else:
        fig = None

    condition_order = ["Baseline (neutral only)", "User correct claim", "User incorrect claim"]
    x_labels = ["Baseline\n(neutral only)", "User correct\nclaim", "User incorrect\nclaim"]
    x = np.arange(len(condition_order))

    for agreement in AGREEMENT_ORDER:
        sub = summary[summary["judge_agreement"].eq(agreement)].copy()
        sub["condition"] = pd.Categorical(sub["condition"], condition_order, ordered=True)
        sub = sub.sort_values("condition")
        y = sub["rate_among_agree_disagree"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=8,
            linewidth=2.5,
            label=AGREEMENT_LABELS[agreement],
            color=AGREEMENT_COLORS[agreement],
        )
        # Put agree labels below and disagree labels above to avoid overlap
        # at the middle condition where the two rates can be close.
        label_offset = -32 if agreement == "agree" else 12
        for xi, row in zip(x, sub.itertuples(index=False)):
            ax.annotate(
                f"{pct(row.rate_among_agree_disagree)}\n{row.count}/{row.valid_total}",
                (xi, row.rate_among_agree_disagree),
                textcoords="offset points",
                xytext=(0, label_offset),
                ha="center",
                fontsize=8.5,
            )

    ax.set_xticks(x, x_labels)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Share among valid agree/disagree rows")
    ax.set_title("Baseline vs claim conditions: agreement/disagreement")
    add_percent_axis(ax)
    ax.legend(loc="best")

    if own_fig:
        ax.text(
            0.0,
            -0.24,
            "Baseline uses only prompt_type=baseline and politeness_level=neutral.",
            transform=ax.transAxes,
            fontsize=9,
            color="dimgray",
        )
        save_fig(fig, output_path)
        return fig
    return None


def plot_baseline_politeness_curves(summary: pd.DataFrame, output_path: Path, ax: Optional[plt.Axes] = None) -> Optional[plt.Figure]:
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
    else:
        fig = None

    condition_order = ["Baseline (neutral only)", "User correct claim", "User incorrect claim"]
    x_labels = ["Baseline\n(neutral only)", "User correct\nclaim", "User incorrect\nclaim"]
    x = np.arange(len(condition_order))

    for style in RESPONSE_POLITENESS_ORDER:
        sub = summary[summary["judge_politeness"].eq(style)].copy()
        sub["condition"] = pd.Categorical(sub["condition"], condition_order, ordered=True)
        sub = sub.sort_values("condition")
        y = sub["rate_among_valid_politeness"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=8,
            linewidth=2.5,
            label=RESPONSE_POLITENESS_LABELS[style],
            color=POLITENESS_COLORS[style],
        )
        for xi, row in zip(x, sub.itertuples(index=False)):
            # Use smaller labels to avoid crowding.
            offset = 8
            if style == "impolite":
                offset = -22
            elif style == "neutral":
                offset = 12
            ax.annotate(
                f"{pct(row.rate_among_valid_politeness)}",
                (xi, row.rate_among_valid_politeness),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=8.5,
            )

    ax.set_xticks(x, x_labels)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Share among valid judge_politeness rows")
    ax.set_title("Baseline vs claim conditions: model response politeness")
    add_percent_axis(ax)
    ax.legend(loc="best")

    if own_fig:
        ax.text(
            0.0,
            -0.24,
            "Baseline uses only prompt_type=baseline and politeness_level=neutral.",
            transform=ax.transAxes,
            fontsize=9,
            color="dimgray",
        )
        save_fig(fig, output_path)
        return fig
    return None


def graph7_baseline_comparisons(df: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agreement_summary = build_baseline_agreement_summary(df)
    politeness_summary = build_baseline_politeness_summary(df)

    plot_baseline_agreement_curves(
        agreement_summary,
        output_dir / "07a_baseline_vs_claims_agreement_curves.png",
    )
    plot_baseline_politeness_curves(
        politeness_summary,
        output_dir / "07b_baseline_vs_claims_response_politeness_curves.png",
    )

    # Combined figure for side-by-side comparison.
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8))
    plot_baseline_agreement_curves(agreement_summary, output_dir / "unused.png", ax=axes[0])
    plot_baseline_politeness_curves(politeness_summary, output_dir / "unused.png", ax=axes[1])
    fig.suptitle("Baseline vs user-correct/user-incorrect claims", fontsize=14, fontweight="bold", y=1.02)
    fig.text(
        0.01,
        -0.03,
        "Baseline is strictly prompt_type=baseline + politeness_level=neutral; baseline+polite and baseline+impolite are ignored.",
        fontsize=9,
        color="dimgray",
    )
    save_fig(fig, output_dir / "07_baseline_vs_claims_combined_curves.png")

    return agreement_summary, politeness_summary


def diagnostics_tables(df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    diagnostics: Dict[str, pd.DataFrame] = {}

    diagnostics["prompt_type_counts"] = df["prompt_type"].value_counts(dropna=False).rename_axis("prompt_type").reset_index(name="count")
    diagnostics["politeness_level_counts"] = df["politeness_level"].value_counts(dropna=False).rename_axis("politeness_level").reset_index(name="count")
    diagnostics["judge_agreement_counts"] = df["judge_agreement"].value_counts(dropna=False).rename_axis("judge_agreement").reset_index(name="count")
    diagnostics["judge_politeness_counts"] = df["judge_politeness"].value_counts(dropna=False).rename_axis("judge_politeness").reset_index(name="count")

    invalid_agreement = df[~df["judge_agreement"].isin(AGREEMENT_ORDER)].copy()
    diagnostics["excluded_from_agreement_rates"] = (
        invalid_agreement.groupby(["prompt_type", "politeness_level", "judge_agreement"])
        .size()
        .reset_index(name="count")
        .sort_values(["prompt_type", "politeness_level", "judge_agreement"])
    )

    invalid_politeness = df[~df["judge_politeness"].isin(RESPONSE_POLITENESS_ORDER)].copy()
    diagnostics["excluded_from_politeness_rates"] = (
        invalid_politeness.groupby(["prompt_type", "politeness_level", "judge_politeness"])
        .size()
        .reset_index(name="count")
        .sort_values(["prompt_type", "politeness_level", "judge_politeness"])
    )

    for name, table in diagnostics.items():
        table.to_csv(output_dir / f"diagnostic_{name}.csv", index=False)
    return diagnostics


def first_rate(summary: pd.DataFrame, **filters: str) -> Optional[float]:
    sub = summary.copy()
    for col, value in filters.items():
        sub = sub[sub[col].eq(value)]
    if sub.empty:
        return None
    return float(sub.iloc[0]["rate_among_agree_disagree"])


def condition_rate(summary: pd.DataFrame, prompt_type: str, politeness_level: str, agreement: str) -> Optional[float]:
    sub = summary[
        summary["prompt_type"].eq(prompt_type)
        & summary["politeness_level"].eq(politeness_level)
        & summary["target_agreement"].eq(agreement)
    ]
    if sub.empty:
        return None
    return float(sub.iloc[0]["rate_among_agree_disagree"])


def write_report(
    df: pd.DataFrame,
    output_dir: Path,
    graph1_summary: pd.DataFrame,
    politeness_condition_summary: pd.DataFrame,
    baseline_audit_table: pd.DataFrame,
    baseline_agreement_summary: pd.DataFrame,
    baseline_politeness_summary: pd.DataFrame,
) -> None:
    # Pull key values for report.
    correct_agree = first_rate(graph1_summary, prompt_type=PROMPT_CORRECT, judge_agreement="agree")
    correct_disagree = first_rate(graph1_summary, prompt_type=PROMPT_CORRECT, judge_agreement="disagree")
    incorrect_agree = first_rate(graph1_summary, prompt_type=PROMPT_INCORRECT, judge_agreement="agree")
    incorrect_disagree = first_rate(graph1_summary, prompt_type=PROMPT_INCORRECT, judge_agreement="disagree")

    correct_agree_impolite = condition_rate(politeness_condition_summary, PROMPT_CORRECT, "impolite", "agree")
    correct_agree_neutral = condition_rate(politeness_condition_summary, PROMPT_CORRECT, "neutral", "agree")
    correct_agree_polite = condition_rate(politeness_condition_summary, PROMPT_CORRECT, "polite", "agree")
    incorrect_disagree_impolite = condition_rate(politeness_condition_summary, PROMPT_INCORRECT, "impolite", "disagree")
    incorrect_disagree_neutral = condition_rate(politeness_condition_summary, PROMPT_INCORRECT, "neutral", "disagree")
    incorrect_disagree_polite = condition_rate(politeness_condition_summary, PROMPT_INCORRECT, "polite", "disagree")
    incorrect_agree_impolite = condition_rate(politeness_condition_summary, PROMPT_INCORRECT, "impolite", "agree")
    incorrect_agree_neutral = condition_rate(politeness_condition_summary, PROMPT_INCORRECT, "neutral", "agree")
    incorrect_agree_polite = condition_rate(politeness_condition_summary, PROMPT_INCORRECT, "polite", "agree")

    baseline_kept = int(baseline_audit_table[baseline_audit_table["used_as_baseline"]]["count"].sum())
    baseline_ignored = int(baseline_audit_table[~baseline_audit_table["used_as_baseline"]]["count"].sum())

    excluded_agreement = int((~df["judge_agreement"].isin(AGREEMENT_ORDER)).sum())
    excluded_politeness = int((~df["judge_politeness"].isin(RESPONSE_POLITENESS_ORDER)).sum())

    # Baseline comparison highlights.
    def baseline_rate_agreement(condition: str, agreement: str) -> Optional[float]:
        sub = baseline_agreement_summary[
            baseline_agreement_summary["condition"].eq(condition)
            & baseline_agreement_summary["judge_agreement"].eq(agreement)
        ]
        return None if sub.empty else float(sub.iloc[0]["rate_among_agree_disagree"])

    def baseline_rate_politeness(condition: str, style: str) -> Optional[float]:
        sub = baseline_politeness_summary[
            baseline_politeness_summary["condition"].eq(condition)
            & baseline_politeness_summary["judge_politeness"].eq(style)
        ]
        return None if sub.empty else float(sub.iloc[0]["rate_among_valid_politeness"])

    baseline_agree = baseline_rate_agreement("Baseline (neutral only)", "agree")
    baseline_disagree = baseline_rate_agreement("Baseline (neutral only)", "disagree")
    correct_claim_agree = baseline_rate_agreement("User correct claim", "agree")
    incorrect_claim_agree = baseline_rate_agreement("User incorrect claim", "agree")
    baseline_polite_resp = baseline_rate_politeness("Baseline (neutral only)", "polite")
    correct_polite_resp = baseline_rate_politeness("User correct claim", "polite")
    incorrect_polite_resp = baseline_rate_politeness("User incorrect claim", "polite")

    report = f"""
# Focused politeness/agreement plot report

Input rows: **{len(df):,}**

## Denominator rules

- Agreement/disagreement plots use only rows where `judge_agreement` is `agree` or `disagree`.
- Rows with other agreement labels are excluded from agreement rates: **{excluded_agreement:,}** rows.
- Response-politeness plots use only rows where `judge_politeness` is `polite`, `neutral`, or `impolite`.
- Rows with other response-politeness labels are excluded from response-politeness rates: **{excluded_politeness:,}** rows.
- Baseline is strictly `prompt_type=baseline` and `politeness_level=neutral`: **{baseline_kept:,}** baseline rows kept, **{baseline_ignored:,}** baseline+polite/impolite rows ignored for baseline comparisons.

## Graphs created

1. `01_claim_correctness_vs_model_agreement_heatmap.png`  
   User-correct/user-incorrect claims vs model agree/disagree.

2. `02_user_correct_model_agree_by_politeness.png`  
   User-correct + model-agree rate across prompt politeness levels.

3. `03_user_correct_model_disagree_by_politeness.png`  
   User-correct + model-disagree rate across prompt politeness levels.

4. `04_user_incorrect_model_disagree_by_politeness.png`  
   User-incorrect + model-disagree rate across prompt politeness levels.

5. `05_user_incorrect_model_agree_by_politeness.png`  
   User-incorrect + model-agree rate across prompt politeness levels.  
   Note: your bullet 5 repeated user-correct + model-agree, so this script uses the missing fourth logical combination: user-incorrect + model-agree.

6. `06_baseline_filter_audit.png`  
   Confirms that only baseline+neutral is used as baseline and baseline+polite/impolite are ignored.

7. `07a_baseline_vs_claims_agreement_curves.png` and `07b_baseline_vs_claims_response_politeness_curves.png`  
   Baseline-vs-claim comparison curves for agreement/disagreement and response politeness.

   `07_baseline_vs_claims_combined_curves.png` combines both graph 7 views into one figure.

## Main observations

- For `user_correct_claim`, the model agrees **{pct(correct_agree)}** and disagrees **{pct(correct_disagree)}** among valid agree/disagree rows.
- For `user_incorrect_claim`, the model agrees **{pct(incorrect_agree)}** and disagrees **{pct(incorrect_disagree)}** among valid agree/disagree rows.
- When the user is correct, model agreement changes sharply with prompt politeness: impolite **{pct(correct_agree_impolite)}**, neutral **{pct(correct_agree_neutral)}**, polite **{pct(correct_agree_polite)}**.
- When the user is incorrect, model disagreement is strongest for impolite prompts: impolite **{pct(incorrect_disagree_impolite)}**, neutral **{pct(incorrect_disagree_neutral)}**, polite **{pct(incorrect_disagree_polite)}**.
- The risky sycophancy-like case, `user_incorrect_claim + model agrees`, is lowest for impolite prompts at **{pct(incorrect_agree_impolite)}**, but higher for neutral **{pct(incorrect_agree_neutral)}** and polite **{pct(incorrect_agree_polite)}** prompts.
- Baseline-neutral agreement is **{pct(baseline_agree)}** and disagreement is **{pct(baseline_disagree)}**. Compared with this baseline, user-correct claims raise agreement to **{pct(correct_claim_agree)}**, while user-incorrect claims reduce agreement to **{pct(incorrect_claim_agree)}**.
- Model responses are polite more often in claim prompts than in the neutral baseline: baseline **{pct(baseline_polite_resp)}**, user-correct **{pct(correct_polite_resp)}**, user-incorrect **{pct(incorrect_polite_resp)}**.

## Tables created

- `01_claim_correctness_vs_model_agreement_summary.csv`
- `02_to_05_politeness_condition_summary.csv`
- `06_baseline_filter_audit.csv`
- `07a_baseline_vs_claims_agreement_summary.csv`
- `07b_baseline_vs_claims_response_politeness_summary.csv`
- `diagnostic_*.csv`
"""
    report = textwrap.dedent(report).strip() + "\n"
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def run_analysis(input_file: str | Path, output_dir: str | Path) -> Path:
    output_dir = ensure_output_dir(output_dir)
    df = load_and_prepare(input_file)

    graph1_summary = graph1_correctness_agreement_heatmap(df, output_dir)
    graph1_summary.to_csv(output_dir / "01_claim_correctness_vs_model_agreement_summary.csv", index=False)

    politeness_condition_summary = graph2_to_5_politeness_condition_plots(df, output_dir)
    politeness_condition_summary.to_csv(output_dir / "02_to_05_politeness_condition_summary.csv", index=False)

    baseline_audit_table = baseline_filter_audit(df, output_dir)
    baseline_audit_table.to_csv(output_dir / "06_baseline_filter_audit.csv", index=False)

    baseline_agreement_summary, baseline_politeness_summary = graph7_baseline_comparisons(df, output_dir)
    baseline_agreement_summary.to_csv(output_dir / "07a_baseline_vs_claims_agreement_summary.csv", index=False)
    baseline_politeness_summary.to_csv(output_dir / "07b_baseline_vs_claims_response_politeness_summary.csv", index=False)

    diagnostics_tables(df, output_dir)

    write_report(
        df=df,
        output_dir=output_dir,
        graph1_summary=graph1_summary,
        politeness_condition_summary=politeness_condition_summary,
        baseline_audit_table=baseline_audit_table,
        baseline_agreement_summary=baseline_agreement_summary,
        baseline_politeness_summary=baseline_politeness_summary,
    )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create focused politeness/agreement plots for sycophancy analysis."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to CSV with judge_politeness, judge_agreement, politeness_level, prompt_type.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/plots/requested_politeness_analysis",
        help="Directory where plots, summaries, and report will be saved.",
    )
    args = parser.parse_args()

    output_dir = run_analysis(args.input_file, args.output_dir)
    print(f"Done. Outputs saved to: {output_dir}")
    print(f"Report: {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
