"""
LESSON 10: FINAL RESULTS ANALYSIS AND THESIS-READY OUTPUTS

Purpose
-------
Summarise the untouched five-fold outer-test evaluation produced by
09_model_evaluation.py. This script does NOT train models, predict new
samples, select thresholds, change selective cut-offs, or modify any
Lesson 01-09 output.

Required inputs in 04_Results/tables/
-------------------------------------
evaluation_probability_metrics_pooled.csv
evaluation_probability_summary_mean_std.csv
evaluation_probability_all_folds.csv
evaluation_threshold_metrics_pooled.csv
evaluation_threshold_summary_mean_std.csv
evaluation_thresholds_all_folds.csv
evaluation_selective_metrics_pooled.csv
evaluation_selective_summary_mean_std.csv
evaluation_selective_all_folds.csv
test_predictions_all_folds.csv

Also validates:
01_Data/raw/water_potability.csv

Outputs (all new files prefixed with 10_)
-----------------------------------------
04_Results/tables/
    10_probability_quality.csv            pooled + fold mean ± SD + nominal bootstrap 95% CI
    10_threshold_comparison.csv           pooled + fold mean ± SD + nominal exact 95% CI
    10_selective_comparison.csv           pooled + fold mean ± SD + nominal exact 95% CI
    10_operating_points_meeting_target.csv  every operating point, flagged (no ranking)
    10_metric_leaders.csv                 metric-wise leaders, NOT a composite winner
    10_thesis_ready_model_comparison.csv  one row per model
    10_final_results_report.md
    10_final_results_manifest.json        SHA-256 of every input and output

04_Results/figures/   (PNG 300 dpi + PDF)
    10_probability_quality
    10_average_precision
    10_threshold_false_safe_vs_recall
    10_selective_false_safe_vs_recall
    10_selective_risk_vs_coverage
    10_selective_actual_vs_target_coverage
    10_false_safe_rate_intervals
    10_pooled_roc_and_reliability

Usage
-----
From the project root:
    python 02_Code/10_final_results_analysis.py

Optional:
    python 02_Code/10_final_results_analysis.py --overwrite
    python 02_Code/10_final_results_analysis.py --project-dir "C:/path/to/project"

Research safeguards
-------------------
* Requires all five outer-test folds and every raw row exactly once.
* Checks row_index and y_true against the original raw dataset.
* Distinguishes pooled metrics from fold mean +/- standard deviation.
* Uses the primary false-safe definition from Lesson 09:
      FSR = FP / (FP + TN)
  and verifies it from the saved confusion counts.
* Reports nominal descriptive intervals (bootstrap for ROC-AUC / Brier,
  Clopper-Pearson for FSR and recall); they do not capture full model-refit
  uncertainty or provide formal paired model-comparison tests.
* Does not calculate a single composite model score or declare a winner.
* A metric-wise leader on a SAFETY metric must be a model that ranks
  better than chance (bootstrap ROC-AUC 95% CI above 0.5) and declares
  at least one sample potable; otherwise a low FSR only reflects
  declaring (almost) nothing potable.
* Never edits source CSVs, model files, thresholds, calibrators, or rules.
* Requires complete five-fold Lesson 09 selective outputs; partial runs are
  rejected rather than silently reported as five-fold evidence.
* If this script's own 10_ outputs exist, it stops unless --overwrite is given;
  --overwrite replaces only the outputs listed by this script.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import beta as beta_distribution
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss,
    roc_auc_score, roc_curve,
)


# ============================================================
# 2. FIXED PROJECT SETTINGS
# ============================================================

TARGET = "Potability"
EXPECTED_FOLDS = {1, 2, 3, 4, 5}

TARGET_FSR = 0.05
CI_LEVEL = 0.95
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
N_RELIABILITY_BINS = 10

MODELS = [
    "majority_baseline",
    "logistic_regression",
    "random_forest",
    "mlp",
]

NON_BASELINE_MODELS = [
    "logistic_regression",
    "random_forest",
    "mlp",
]

REFERENCE_ONLY_MODELS = {"majority_baseline"}

PROBABILITY_METHODS = [
    "uncalibrated",
    "sigmoid",
    "isotonic",
]

THRESHOLD_STRATEGIES = [
    "fixed_0.50",
    "empirical",
    "clopper_pearson",
    "bonferroni",
]

TARGET_COVERAGES = [0.25, 0.50, 0.75, 1.00]

INPUT_FILES = [
    "evaluation_probability_metrics_pooled.csv",
    "evaluation_probability_summary_mean_std.csv",
    "evaluation_probability_all_folds.csv",
    "evaluation_threshold_metrics_pooled.csv",
    "evaluation_threshold_summary_mean_std.csv",
    "evaluation_thresholds_all_folds.csv",
    "evaluation_selective_metrics_pooled.csv",
    "evaluation_selective_summary_mean_std.csv",
    "evaluation_selective_all_folds.csv",
    "test_predictions_all_folds.csv",
]

OUTPUT_TABLES = [
    "10_probability_quality.csv",
    "10_threshold_comparison.csv",
    "10_selective_comparison.csv",
    "10_operating_points_meeting_target.csv",
    "10_metric_leaders.csv",
    "10_thesis_ready_model_comparison.csv",
    "10_final_results_report.md",
    "10_final_results_manifest.json",
]

OUTPUT_FIGURES = [
    "10_probability_quality",
    "10_average_precision",
    "10_threshold_false_safe_vs_recall",
    "10_selective_false_safe_vs_recall",
    "10_selective_risk_vs_coverage",
    "10_selective_actual_vs_target_coverage",
    "10_false_safe_rate_intervals",
    "10_pooled_roc_and_reliability",
]

# ---------- Display settings (thesis figures) ----------

MODEL_LABELS = {
    "majority_baseline": "Majority baseline",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "mlp": "MLP",
}

STRATEGY_LABELS = {
    "fixed_0.50": "Fixed 0.50",
    "empirical": "Empirical",
    "clopper_pearson": "Clopper–Pearson",
    "bonferroni": "Bonferroni",
}

# Fixed colour per model (validated for colour-vision deficiency), so a
# model keeps the same colour in every figure.
MODEL_COLORS = {
    "logistic_regression": "#2a78d6",
    "random_forest": "#eb6834",
    "mlp": "#1baf7a",
    "majority_baseline": "#8a8985",
}

# Marker shape is a second encoding so identity never relies on colour.
STRATEGY_MARKERS = {
    "fixed_0.50": "o",
    "empirical": "s",
    "clopper_pearson": "D",
    "bonferroni": "^",
}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
TARGET_LINE_COLOR = "#d03b3b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": TEXT_PRIMARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "savefig.facecolor": "white",
})


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def require_file(path, purpose=""):
    path = Path(path)
    if not path.is_file():
        message = f"Required input file not found:\n{path}"
        if purpose:
            message += f"\n{purpose}"
        raise FileNotFoundError(message)
    return path


def require_columns(df, required, filename):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def sha256_file(path):
    """Return the SHA-256 fingerprint of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(table_dir, filename, required_columns):
    path = require_file(
        table_dir / filename,
        "Run `python 02_Code/09_model_evaluation.py --all` first.",
    )
    df = pd.read_csv(path)
    require_columns(df, required_columns, filename)
    if df.empty:
        raise ValueError(f"{filename} is empty.")
    return df


def check_five_fold_ids(df, filename):
    require_columns(df, ["fold"], filename)
    numeric = pd.to_numeric(df["fold"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{filename} contains non-integer or non-finite fold IDs.")
    folds = set(numeric.astype(int).tolist())
    if folds != EXPECTED_FOLDS:
        raise AssertionError(
            f"{filename} must contain folds {sorted(EXPECTED_FOLDS)}, "
            f"but contains {sorted(folds)}."
        )


def check_complete_grid(df, filename, keys, expected_values):
    """Check that every expected N-dimensional combination exists exactly once."""
    if len(keys) != len(expected_values):
        raise ValueError("Grid keys and expected-value dimensions do not match.")

    if df.duplicated(keys).any():
        duplicate_rows = df.loc[df.duplicated(keys, keep=False), keys]
        raise AssertionError(
            f"{filename} contains duplicate combinations:\\n"
            f"{duplicate_rows.head(10).to_string(index=False)}"
        )

    expected = set(itertools.product(*expected_values))
    observed = set(map(tuple, df[keys].itertuples(index=False, name=None)))
    missing = expected - observed
    extra = observed - expected

    if missing or extra:
        raise AssertionError(
            f"{filename} has an incomplete or unexpected result grid.\\n"
            f"Missing (first 10): {list(missing)[:10]}\\n"
            f"Unexpected (first 10): {list(extra)[:10]}"
        )


def clopper_pearson_interval(successes, trials, level=CI_LEVEL):
    """
    Exact two-sided Clopper-Pearson interval for a binomial proportion.

    Used only to DESCRIBE uncertainty in outer-test proportions.
    It is not used to choose anything.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("Clopper–Pearson requires 0 <= successes <= trials.")
    if not 0 < level < 1:
        raise ValueError("Confidence level must be strictly between 0 and 1.")
    if trials == 0:
        return np.nan, np.nan
    alpha = 1.0 - level
    lower = (
        0.0 if successes == 0
        else beta_distribution.ppf(alpha / 2, successes, trials - successes + 1)
    )
    upper = (
        1.0 if successes == trials
        else beta_distribution.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return float(lower), float(upper)


def percent(value, digits=1):
    if pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.{digits}f}%"


def percent_ci(value, lower, upper, digits=1):
    if pd.isna(value):
        return "NA"
    return (
        f"{100 * value:.{digits}f}% "
        f"[{100 * lower:.{digits}f}, {100 * upper:.{digits}f}]"
    )


def mean_sd(mean, sd, digits=3, as_percent=False):
    if pd.isna(mean):
        return "NA"
    sd = 0.0 if pd.isna(sd) else sd
    if as_percent:
        return f"{100 * mean:.1f} ± {100 * sd:.1f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def fmt(value, digits=4):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def dataframe_to_markdown(df):
    """Render a compact Markdown table without requiring the tabulate package."""
    headers = [str(column) for column in df.columns]

    def cell_text(value):
        if isinstance(value, float) and pd.isna(value):
            return "—"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [[cell_text(value) for value in row]
            for row in df.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def order_rows(df, by_model=True, second=None, second_order=None):
    """Sort rows in the fixed thesis order (never by performance)."""
    work = df.copy()
    sort_keys = []
    if by_model:
        work["_model_order"] = work["model"].map(MODELS.index)
        sort_keys.append("_model_order")
    if second is not None:
        if second_order is not None:
            work["_second_order"] = work[second].map(second_order.index)
            sort_keys.append("_second_order")
        else:
            sort_keys.append(second)
    work = work.sort_values(sort_keys).drop(
        columns=[c for c in ["_model_order", "_second_order"] if c in work]
    )
    return work.reset_index(drop=True)


# ============================================================
# 4. OUTPUT PROTECTION
# ============================================================

def check_existing_outputs(table_dir, figure_dir, overwrite):
    """
    Never touch Lesson 01-09 files. If this script's own 10_ outputs
    already exist, stop unless --overwrite was given.
    """
    existing = [
        table_dir / name for name in OUTPUT_TABLES
        if (table_dir / name).exists()
    ] + [
        figure_dir / f"{name}.{ext}"
        for name in OUTPUT_FIGURES for ext in ("png", "pdf")
        if (figure_dir / f"{name}.{ext}").exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"{len(existing)} Lesson 10 output file(s) already exist, e.g.\n"
            f"{existing[0]}\n"
            "Nothing was changed. Re-run with --overwrite to replace only "
            "the 10_ files created by this script."
        )


# ============================================================
# 5. INPUT AND PROVENANCE VALIDATION
# ============================================================

def validate_prediction_coverage(project_dir, table_dir):
    raw_path = require_file(
        project_dir / "01_Data" / "raw" / "water_potability.csv",
        "The original raw dataset is required for row/label validation.",
    )
    predictions_path = require_file(
        table_dir / "test_predictions_all_folds.csv",
        "Run Lesson 09 with --all to create combined outer-test predictions.",
    )

    raw = pd.read_csv(raw_path)
    if TARGET not in raw.columns:
        raise KeyError(f"Raw dataset has no '{TARGET}' column.")
    if raw[TARGET].isna().any():
        raise ValueError("Raw target contains missing values.")
    if set(pd.to_numeric(raw[TARGET], errors="raise").unique()) != {0, 1}:
        raise ValueError("Raw Potability target must contain labels 0 and 1.")

    predictions = pd.read_csv(predictions_path)
    require_columns(
        predictions, ["fold", "row_index", "y_true"],
        "test_predictions_all_folds.csv",
    )
    check_five_fold_ids(predictions, "test_predictions_all_folds.csv")

    if predictions["row_index"].isna().any():
        raise ValueError("Combined outer-test predictions contain missing row_index.")

    row_ids_numeric = pd.to_numeric(
        predictions["row_index"], errors="raise"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(row_ids_numeric).all()
        or not np.equal(row_ids_numeric, np.floor(row_ids_numeric)).all()
    ):
        raise ValueError("row_index must contain finite integer values.")
    row_ids = row_ids_numeric.astype(int)

    if len(np.unique(row_ids)) != len(row_ids):
        raise AssertionError(
            "A raw dataset row appears more than once in outer-test predictions."
        )

    expected_ids = set(range(len(raw)))
    observed_ids = set(row_ids.tolist())
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise AssertionError(
            "Outer-test folds do not cover every raw row exactly once.\n"
            f"Missing row IDs (first 20): {missing[:20]}\n"
            f"Unexpected row IDs (first 20): {extra[:20]}"
        )

    y_saved = pd.to_numeric(predictions["y_true"], errors="raise").to_numpy(dtype=int)
    y_raw = pd.to_numeric(raw[TARGET], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(y_saved, y_raw[row_ids]):
        raise AssertionError(
            "Saved y_true values do not match raw dataset labels at row_index."
        )

    required_prediction_columns = [
        f"{model}_{method}_probability"
        for model in MODELS for method in PROBABILITY_METHODS
    ] + [
        f"{model}_{strategy}_prediction"
        for model in MODELS for strategy in THRESHOLD_STRATEGIES
    ] + [
        f"{model}_{strategy}_threshold"
        for model in MODELS for strategy in THRESHOLD_STRATEGIES
    ]
    require_columns(
        predictions, required_prediction_columns,
        "test_predictions_all_folds.csv",
    )

    # Validate saved probabilities and prove that every stored threshold
    # decision can be reproduced from that row's sigmoid probability and
    # its fold-specific saved threshold. This does not reselect anything.
    for model in MODELS:
        for method in PROBABILITY_METHODS:
            column = f"{model}_{method}_probability"
            values = pd.to_numeric(predictions[column], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"{column} contains invalid probabilities.")

        sigmoid = predictions[f"{model}_sigmoid_probability"].to_numpy(dtype=float)
        for strategy in THRESHOLD_STRATEGIES:
            threshold_column = f"{model}_{strategy}_threshold"
            prediction_column = f"{model}_{strategy}_prediction"
            thresholds = pd.to_numeric(
                predictions[threshold_column], errors="raise"
            ).to_numpy(dtype=float)
            observed_numeric = pd.to_numeric(
                predictions[prediction_column], errors="raise"
            ).to_numpy(dtype=float)
            if (
                not np.isfinite(observed_numeric).all()
                or not np.equal(observed_numeric, np.floor(observed_numeric)).all()
            ):
                raise ValueError(f"{prediction_column} must contain finite integer labels.")
            observed = observed_numeric.astype(int)

            if not np.isfinite(thresholds).all():
                raise ValueError(f"{threshold_column} contains non-finite thresholds.")
            if not np.isin(observed, [0, 1]).all():
                raise ValueError(f"{prediction_column} must contain only 0/1 labels.")

            # Lesson 09 preserves the >1 reject-all sentinel exactly.
            expected = (sigmoid >= thresholds).astype(int)
            if not np.array_equal(observed, expected):
                raise AssertionError(
                    f"{prediction_column} does not match saved sigmoid "
                    f"probabilities and fold-specific thresholds."
                )

            # Each fold has one saved threshold per model/strategy.
            threshold_check = pd.DataFrame({
                "fold": predictions["fold"].to_numpy(),
                "threshold": thresholds,
            })
            if (threshold_check.groupby("fold")["threshold"].nunique() != 1).any():
                raise AssertionError(
                    f"{threshold_column} changes within an outer-test fold."
                )

    fold_counts = (
        predictions.groupby("fold")["row_index"]
        .count()
        .rename("outer_test_rows")
        .reset_index()
    )

    return raw, raw_path, predictions, fold_counts


def expected_calibration_error(y_true, probabilities, n_bins=N_RELIABILITY_BINS):
    """Match Lesson 09's equal-width-bin ECE implementation exactly."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for bin_number in range(n_bins):
        mask = bin_ids == bin_number
        if mask.any():
            ece += mask.mean() * abs(
                y_true[mask].mean() - probabilities[mask].mean()
            )
    return float(ece)


def verify_pooled_probability_metrics(probability_pooled, predictions):
    """Recalculate Lesson 09 pooled probability metrics from saved OOF rows."""
    y_true = predictions["y_true"].to_numpy(dtype=int)
    for row in probability_pooled.itertuples(index=False):
        column = f"{row.model}_{row.probability_method}_probability"
        p = predictions[column].to_numpy(dtype=float)
        p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        calculated = {
            "roc_auc": roc_auc_score(y_true, p),
            "average_precision": average_precision_score(y_true, p),
            "brier_score": brier_score_loss(y_true, p),
            "log_loss": log_loss(y_true, p_clipped, labels=[0, 1]),
            "ece": expected_calibration_error(y_true, p),
        }
        for metric, value in calculated.items():
            if not np.isclose(
                float(getattr(row, metric)), float(value),
                rtol=1e-7, atol=1e-8
            ):
                raise AssertionError(
                    f"Pooled {metric} for {row.model}/{row.probability_method} "
                    "does not match the saved row-level probabilities."
                )


def load_evaluation_outputs(table_dir, predictions):
    probability_pooled = load_csv(
        table_dir, "evaluation_probability_metrics_pooled.csv",
        ["model", "probability_method", "roc_auc", "average_precision",
         "brier_score", "log_loss", "ece"],
    )
    probability_mean_std = load_csv(
        table_dir, "evaluation_probability_summary_mean_std.csv",
        ["model", "probability_method", "roc_auc_mean", "roc_auc_std",
         "average_precision_mean", "average_precision_std",
         "brier_score_mean", "brier_score_std",
         "log_loss_mean", "log_loss_std", "ece_mean", "ece_std"],
    )
    probability_folds = load_csv(
        table_dir, "evaluation_probability_all_folds.csv",
        ["fold", "model", "probability_method", "roc_auc"],
    )
    threshold_pooled = load_csv(
        table_dir, "evaluation_threshold_metrics_pooled.csv",
        ["model", "strategy", "n_folds", "mean_applied_threshold",
         "tn", "fp_false_safe", "fn_false_unsafe", "tp",
         "n_actual_non_potable", "n_actual_potable", "n_declared_potable",
         "false_safe_rate", "unsafe_share_among_declared_potable",
         "potable_recall", "positive_precision", "specificity", "accuracy",
         "balanced_accuracy", "f1_score", "safe_prediction_share"],
    )
    threshold_mean_std = load_csv(
        table_dir, "evaluation_threshold_summary_mean_std.csv",
        ["model", "strategy", "threshold_mean", "threshold_std",
         "false_safe_rate_mean", "false_safe_rate_std",
         "potable_recall_mean", "potable_recall_std",
         "positive_precision_mean", "positive_precision_std",
         "f1_score_mean", "f1_score_std"],
    )
    threshold_folds = load_csv(
        table_dir, "evaluation_thresholds_all_folds.csv",
        ["fold", "model", "strategy", "tn", "fp_false_safe",
         "fn_false_unsafe", "tp", "n_actual_non_potable",
         "n_actual_potable", "false_safe_rate", "potable_recall"],
    )
    selective_pooled = load_csv(
        table_dir, "evaluation_selective_metrics_pooled.csv",
        ["model", "target_coverage", "n_folds", "n_samples", "n_decided",
         "n_needs_review", "actual_coverage", "review_share",
         "selective_accuracy", "accepted_error_rate", "n_false_safe",
         "n_actual_non_potable", "false_safe_rate_all_non_potable",
         "unsafe_share_among_declared_safe", "potable_recall_all",
         "precision_among_decided", "f1_among_decided"],
    )
    selective_mean_std = load_csv(
        table_dir, "evaluation_selective_summary_mean_std.csv",
        ["model", "target_coverage", "actual_coverage_mean",
         "actual_coverage_std", "accepted_error_rate_mean",
         "accepted_error_rate_std",
         "false_safe_rate_all_non_potable_mean",
         "false_safe_rate_all_non_potable_std",
         "potable_recall_all_mean", "potable_recall_all_std"],
    )
    selective_folds = load_csv(
        table_dir, "evaluation_selective_all_folds.csv",
        ["fold", "model", "target_coverage", "n_samples", "n_decided",
         "n_needs_review", "n_correct_decided", "n_safe", "n_false_safe",
         "n_actual_non_potable", "n_actual_potable", "decided_tn",
         "decided_fp_false_safe", "decided_fn_false_unsafe", "decided_tp",
         "actual_coverage", "accepted_error_rate",
         "false_safe_rate_all_non_potable", "potable_recall_all"],
    )

    # ---- Fold-level tables: five folds, complete grids, no duplicates ----
    fold_grid_specs = [
        (
            "evaluation_probability_all_folds.csv",
            probability_folds,
            ["fold", "model", "probability_method"],
            [sorted(EXPECTED_FOLDS), MODELS, PROBABILITY_METHODS],
        ),
        (
            "evaluation_thresholds_all_folds.csv",
            threshold_folds,
            ["fold", "model", "strategy"],
            [sorted(EXPECTED_FOLDS), MODELS, THRESHOLD_STRATEGIES],
        ),
        (
            "evaluation_selective_all_folds.csv",
            selective_folds,
            ["fold", "model", "target_coverage"],
            [sorted(EXPECTED_FOLDS), NON_BASELINE_MODELS, TARGET_COVERAGES],
        ),
    ]
    for filename, frame, keys, expected_values in fold_grid_specs:
        check_five_fold_ids(frame, filename)
        check_complete_grid(frame, filename, keys, expected_values)

    # ---- Pooled and mean/SD tables: complete grids ----
    for filename, frame in [
        ("evaluation_probability_metrics_pooled.csv", probability_pooled),
        ("evaluation_probability_summary_mean_std.csv", probability_mean_std),
    ]:
        check_complete_grid(frame, filename, ["model", "probability_method"],
                            [MODELS, PROBABILITY_METHODS])
    for filename, frame in [
        ("evaluation_threshold_metrics_pooled.csv", threshold_pooled),
        ("evaluation_threshold_summary_mean_std.csv", threshold_mean_std),
    ]:
        check_complete_grid(frame, filename, ["model", "strategy"],
                            [MODELS, THRESHOLD_STRATEGIES])
    # Selective tables intentionally exclude the reference-only baseline.
    for filename, frame in [
        ("evaluation_selective_metrics_pooled.csv", selective_pooled),
        ("evaluation_selective_summary_mean_std.csv", selective_mean_std),
    ]:
        check_complete_grid(frame, filename, ["model", "target_coverage"],
                            [NON_BASELINE_MODELS, TARGET_COVERAGES])

    verify_pooled_probability_metrics(probability_pooled, predictions)

    if not (threshold_pooled["n_folds"].astype(int) == 5).all():
        raise AssertionError("Pooled threshold results do not all report five folds.")
    if not (selective_pooled["n_folds"].astype(int) == 5).all():
        raise AssertionError("Pooled selective results do not all report five folds.")
    if not (selective_pooled["n_samples"].astype(int) == len(predictions)).all():
        raise AssertionError("Pooled selective sample counts do not match the dataset.")

    # ---- The primary FSR really is FP / (FP + TN) ----
    recomputed = threshold_pooled["fp_false_safe"] / (
        threshold_pooled["fp_false_safe"] + threshold_pooled["tn"]
    )
    if not np.allclose(recomputed, threshold_pooled["false_safe_rate"]):
        raise AssertionError("Pooled threshold false_safe_rate is not FP / (FP + TN).")

    # Cross-check every fold-level confusion matrix against row-level predictions.
    for row in threshold_folds.itertuples(index=False):
        fold_rows = predictions.loc[predictions["fold"] == row.fold]
        y_fold = fold_rows["y_true"].to_numpy(dtype=int)
        predicted_fold = fold_rows[
            f"{row.model}_{row.strategy}_prediction"
        ].to_numpy(dtype=int)
        tn_fold = int(((y_fold == 0) & (predicted_fold == 0)).sum())
        fp_fold = int(((y_fold == 0) & (predicted_fold == 1)).sum())
        fn_fold = int(((y_fold == 1) & (predicted_fold == 0)).sum())
        tp_fold = int(((y_fold == 1) & (predicted_fold == 1)).sum())
        if (
            int(row.tn), int(row.fp_false_safe),
            int(row.fn_false_unsafe), int(row.tp)
        ) != (tn_fold, fp_fold, fn_fold, tp_fold):
            raise AssertionError(
                f"Fold-level confusion counts for {row.model}/{row.strategy}, "
                f"fold {row.fold}, do not match row-level predictions."
            )

    # Cross-check every pooled threshold confusion matrix against the
    # combined row-level outer-test predictions, not merely against its own FSR.
    y_all = predictions["y_true"].to_numpy(dtype=int)
    for row in threshold_pooled.itertuples(index=False):
        prediction_column = f"{row.model}_{row.strategy}_prediction"
        predicted = predictions[prediction_column].to_numpy(dtype=int)
        tn = int(((y_all == 0) & (predicted == 0)).sum())
        fp = int(((y_all == 0) & (predicted == 1)).sum())
        fn = int(((y_all == 1) & (predicted == 0)).sum())
        tp = int(((y_all == 1) & (predicted == 1)).sum())
        observed_counts = (int(row.tn), int(row.fp_false_safe),
                           int(row.fn_false_unsafe), int(row.tp))
        expected_counts = (tn, fp, fn, tp)
        if observed_counts != expected_counts:
            raise AssertionError(
                f"Pooled confusion counts for {row.model}/{row.strategy} "
                f"do not match test_predictions_all_folds.csv: "
                f"{observed_counts} != {expected_counts}"
            )

    n_non_potable = int((predictions["y_true"] == 0).sum())
    if not (threshold_pooled["n_actual_non_potable"] == n_non_potable).all():
        raise AssertionError("Threshold non-potable counts do not match the dataset.")
    if not (selective_pooled["n_actual_non_potable"] == n_non_potable).all():
        raise AssertionError("Selective non-potable counts do not match the dataset.")

    selective_fold_count_columns = [
        "n_samples", "n_decided", "n_needs_review", "n_correct_decided",
        "n_safe", "n_false_safe", "n_actual_non_potable", "n_actual_potable",
        "decided_tn", "decided_fp_false_safe", "decided_fn_false_unsafe",
        "decided_tp",
    ]
    selective_fold_totals = (
        selective_folds.groupby(["model", "target_coverage"])[selective_fold_count_columns]
        .sum()
        .reset_index()
    )
    selective_count_check = selective_pooled.merge(
        selective_fold_totals,
        on=["model", "target_coverage"],
        suffixes=("_pooled", "_fold_sum"),
        validate="one_to_one",
    )
    pooled_count_columns = [
        "n_samples", "n_decided", "n_needs_review",
        "n_false_safe", "n_actual_non_potable",
    ]
    for column in pooled_count_columns:
        if not np.array_equal(
            selective_count_check[f"{column}_pooled"].to_numpy(dtype=int),
            selective_count_check[f"{column}_fold_sum"].to_numpy(dtype=int),
        ):
            raise AssertionError(
                f"Selective pooled {column} does not equal the sum of five fold counts."
            )

    # Lesson 09's pooled selective table omits several integer confusion
    # counts. Add their exact five-fold sums for confidence intervals and
    # thesis tables; do not reconstruct counts from rounded percentages.
    selective_extra_counts = selective_fold_totals[
        [
            "model", "target_coverage", "n_correct_decided", "n_safe",
            "n_actual_potable", "decided_tn", "decided_fp_false_safe",
            "decided_fn_false_unsafe", "decided_tp",
        ]
    ]
    selective_pooled = selective_pooled.merge(
        selective_extra_counts,
        on=["model", "target_coverage"],
        validate="one_to_one",
    )
    if not np.array_equal(
        selective_pooled["n_safe"].to_numpy(dtype=int),
        (
            selective_pooled["decided_tp"]
            + selective_pooled["decided_fp_false_safe"]
        ).to_numpy(dtype=int),
    ):
        raise AssertionError("Selective n_safe does not equal decided TP + decided FP.")
    if not np.array_equal(
        selective_pooled["n_false_safe"].to_numpy(dtype=int),
        selective_pooled["decided_fp_false_safe"].to_numpy(dtype=int),
    ):
        raise AssertionError("Selective false-safe counts do not equal decided FP.")
    if not np.array_equal(
        selective_pooled["n_actual_potable"].to_numpy(dtype=int),
        (
            selective_pooled["n_samples"]
            - selective_pooled["n_actual_non_potable"]
        ).to_numpy(dtype=int),
    ):
        raise AssertionError("Selective class counts do not sum to n_samples.")
    if not np.array_equal(
        (
            selective_pooled["n_decided"]
            + selective_pooled["n_needs_review"]
        ).to_numpy(dtype=int),
        selective_pooled["n_samples"].to_numpy(dtype=int),
    ):
        raise AssertionError("Selective decided + review counts do not equal n_samples.")

    decided = selective_pooled["n_decided"].to_numpy(dtype=float)
    correct = selective_pooled["n_correct_decided"].to_numpy(dtype=float)
    expected_accuracy = np.divide(
        correct, decided, out=np.full_like(correct, np.nan), where=decided > 0
    )
    expected_error = np.divide(
        decided - correct, decided,
        out=np.full_like(correct, np.nan), where=decided > 0
    )
    if not np.allclose(
        expected_accuracy,
        selective_pooled["selective_accuracy"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise AssertionError("Selective accuracy does not match exact decided counts.")
    if not np.allclose(
        expected_error,
        selective_pooled["accepted_error_rate"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise AssertionError("Selective accepted error does not match exact decided counts.")

    recomputed_selective = (
        selective_pooled["n_false_safe"] / selective_pooled["n_actual_non_potable"]
    )
    if not np.allclose(
        recomputed_selective, selective_pooled["false_safe_rate_all_non_potable"]
    ):
        raise AssertionError("Selective FSR is not FP / all actual non-potable.")

    return {
        "probability_pooled": probability_pooled,
        "probability_mean_std": probability_mean_std,
        "probability_folds": probability_folds,
        "threshold_pooled": threshold_pooled,
        "threshold_mean_std": threshold_mean_std,
        "threshold_folds": threshold_folds,
        "selective_pooled": selective_pooled,
        "selective_mean_std": selective_mean_std,
        "selective_folds": selective_folds,
    }


# ============================================================
# 6. BUILD THESIS-READY TABLES
# ============================================================

def bootstrap_roc_auc_and_brier(y_true, probabilities):
    """
    Percentile bootstrap 95% CI for pooled ROC-AUC and Brier score.

    Resampling is stratified by class, and the same seed is used for
    every model, so all models are compared on identical resamples.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.ndim != 1 or len(y_true) != len(probabilities):
        raise ValueError("Bootstrap labels and probabilities must be aligned 1-D arrays.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Bootstrap probabilities must be finite and lie in [0, 1].")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Stratified ROC-AUC bootstrap requires both labels 0 and 1.")
    positive_idx = np.flatnonzero(y_true == 1)
    negative_idx = np.flatnonzero(y_true == 0)

    auc_values = np.empty(N_BOOTSTRAP)
    brier_values = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sample = np.concatenate([
            rng.choice(positive_idx, len(positive_idx), replace=True),
            rng.choice(negative_idx, len(negative_idx), replace=True),
        ])
        auc_values[b] = roc_auc_score(y_true[sample], probabilities[sample])
        brier_values[b] = brier_score_loss(y_true[sample], probabilities[sample])

    q = [100 * (1 - CI_LEVEL) / 2, 100 * (1 - (1 - CI_LEVEL) / 2)]
    return tuple(np.percentile(auc_values, q)), tuple(np.percentile(brier_values, q))


def build_probability_table(data, predictions):
    table = data["probability_pooled"].merge(
        data["probability_mean_std"][[
            "model", "probability_method",
            "roc_auc_mean", "roc_auc_std",
            "average_precision_mean", "average_precision_std",
            "brier_score_mean", "brier_score_std",
            "log_loss_mean", "log_loss_std",
            "ece_mean", "ece_std",
        ]],
        on=["model", "probability_method"],
        validate="one_to_one",
    )

    ci_rows = []
    for model in MODELS:
        for method in PROBABILITY_METHODS:
            (auc_lo, auc_hi), (brier_lo, brier_hi) = bootstrap_roc_auc_and_brier(
                predictions["y_true"], predictions[f"{model}_{method}_probability"]
            )
            ci_rows.append({
                "model": model, "probability_method": method,
                "roc_auc_ci_lower": auc_lo, "roc_auc_ci_upper": auc_hi,
                "brier_ci_lower": brier_lo, "brier_ci_upper": brier_hi,
            })
    table = table.merge(pd.DataFrame(ci_rows),
                        on=["model", "probability_method"],
                        validate="one_to_one")

    table["reference_only"] = table["model"].isin(REFERENCE_ONLY_MODELS)
    table["better_than_chance"] = table["roc_auc_ci_lower"] > 0.5
    table["roc_auc_pooled_ci"] = table.apply(
        lambda r: f"{r['roc_auc']:.3f} [{r['roc_auc_ci_lower']:.3f}, "
                  f"{r['roc_auc_ci_upper']:.3f}]", axis=1)
    table["roc_auc_fold_mean_sd"] = table.apply(
        lambda r: mean_sd(r["roc_auc_mean"], r["roc_auc_std"]), axis=1)
    table["average_precision_fold_mean_sd"] = table.apply(
        lambda r: mean_sd(
            r["average_precision_mean"], r["average_precision_std"]
        ), axis=1)
    table["log_loss_fold_mean_sd"] = table.apply(
        lambda r: mean_sd(r["log_loss_mean"], r["log_loss_std"]), axis=1)
    table["ece_fold_mean_sd"] = table.apply(
        lambda r: mean_sd(r["ece_mean"], r["ece_std"]), axis=1)
    table["brier_pooled_ci"] = table.apply(
        lambda r: f"{r['brier_score']:.3f} [{r['brier_ci_lower']:.3f}, "
                  f"{r['brier_ci_upper']:.3f}]", axis=1)
    table["brier_fold_mean_sd"] = table.apply(
        lambda r: mean_sd(r["brier_score_mean"], r["brier_score_std"]), axis=1)

    return order_rows(table, second="probability_method",
                      second_order=PROBABILITY_METHODS)


def build_threshold_table(data, discrimination_ok):
    threshold_fold_summary = data["threshold_mean_std"][[
        "model", "strategy", "threshold_mean", "threshold_std",
        "false_safe_rate_mean", "false_safe_rate_std",
        "potable_recall_mean", "potable_recall_std",
        "positive_precision_mean", "positive_precision_std",
        "f1_score_mean", "f1_score_std",
    ]].rename(columns={
        "threshold_mean": "threshold_fold_mean",
        "threshold_std": "threshold_fold_std",
    })
    table = data["threshold_pooled"].merge(
        threshold_fold_summary,
        on=["model", "strategy"],
        validate="one_to_one",
    )

    fsr_ci = table.apply(lambda r: clopper_pearson_interval(
        int(r["fp_false_safe"]), int(r["n_actual_non_potable"])), axis=1)
    recall_ci = table.apply(lambda r: clopper_pearson_interval(
        int(r["tp"]), int(r["n_actual_potable"])), axis=1)
    table["fsr_ci_lower"] = [lo for lo, _ in fsr_ci]
    table["fsr_ci_upper"] = [hi for _, hi in fsr_ci]
    table["recall_ci_lower"] = [lo for lo, _ in recall_ci]
    table["recall_ci_upper"] = [hi for _, hi in recall_ci]

    folds = data["threshold_folds"]
    folds_ok = (
        folds.assign(meets=folds["false_safe_rate"] <= TARGET_FSR)
        .groupby(["model", "strategy"])["meets"].sum()
        .rename("folds_meeting_target_fsr").reset_index()
    )
    table = table.merge(folds_ok, on=["model", "strategy"], validate="one_to_one")

    table["reference_only"] = table["model"].isin(REFERENCE_ONLY_MODELS)
    table["better_than_chance"] = table["model"].map(discrimination_ok)
    table["meets_target_fsr_pooled"] = table["false_safe_rate"] <= TARGET_FSR
    table["fsr_pooled_ci"] = table.apply(lambda r: percent_ci(
        r["false_safe_rate"], r["fsr_ci_lower"], r["fsr_ci_upper"]), axis=1)
    table["fsr_fold_mean_sd"] = table.apply(lambda r: mean_sd(
        r["false_safe_rate_mean"], r["false_safe_rate_std"], as_percent=True), axis=1)
    table["recall_pooled_ci"] = table.apply(lambda r: percent_ci(
        r["potable_recall"], r["recall_ci_lower"], r["recall_ci_upper"]), axis=1)

    def note(row):
        if row["reference_only"]:
            return "Reference only: constant score, declares nothing potable."
        if row["n_declared_potable"] == 0:
            return "Declares nothing potable: FSR of 0 is degenerate."
        if not row["better_than_chance"]:
            return (
                f"Ranking ability not better than chance; only "
                f"{int(row['n_declared_potable'])} rows declared potable. "
                "Low FSR does not indicate useful safety."
            )
        return ""

    table["note"] = table.apply(note, axis=1)
    return order_rows(table, second="strategy", second_order=THRESHOLD_STRATEGIES)


def build_selective_table(data, discrimination_ok):
    table = data["selective_pooled"].merge(
        data["selective_mean_std"][[
            "model", "target_coverage",
            "actual_coverage_mean", "actual_coverage_std",
            "accepted_error_rate_mean", "accepted_error_rate_std",
            "false_safe_rate_all_non_potable_mean",
            "false_safe_rate_all_non_potable_std",
            "potable_recall_all_mean", "potable_recall_all_std",
        ]],
        on=["model", "target_coverage"],
        validate="one_to_one",
    )

    n_potable = table["n_actual_potable"].to_numpy(dtype=int)
    if (n_potable <= 0).any():
        raise AssertionError("Selective pooled rows must contain actual potable samples.")

    # Use exact integer counts aggregated from Lesson 09's five fold-level
    # selective outputs; never infer counts from rounded recall percentages.
    table["n_true_safe"] = table["decided_tp"].astype(int)
    table["n_declared_safe"] = table["n_safe"].astype(int)
    if not np.array_equal(
        table["n_true_safe"].to_numpy(dtype=int),
        (table["potable_recall_all"] * n_potable).round().to_numpy(dtype=int),
    ):
        raise AssertionError(
            "Selective pooled potable_recall_all does not match exact decided TP counts."
        )

    fsr_ci = table.apply(lambda r: clopper_pearson_interval(
        int(r["n_false_safe"]), int(r["n_actual_non_potable"])), axis=1)
    table["fsr_ci_lower"] = [lo for lo, _ in fsr_ci]
    table["fsr_ci_upper"] = [hi for _, hi in fsr_ci]
    recall_ci = [
        clopper_pearson_interval(int(tp), int(n))
        for tp, n in zip(table["n_true_safe"], n_potable)
    ]
    table["recall_ci_lower"] = [lo for lo, _ in recall_ci]
    table["recall_ci_upper"] = [hi for _, hi in recall_ci]

    selective_fold_target = (
        data["selective_folds"]
        .assign(meets=lambda d: d["false_safe_rate_all_non_potable"] <= TARGET_FSR)
        .groupby(["model", "target_coverage"])["meets"]
        .sum()
        .rename("folds_meeting_target_fsr")
        .reset_index()
    )
    table = table.merge(
        selective_fold_target,
        on=["model", "target_coverage"],
        validate="one_to_one",
    )
    table["better_than_chance"] = table["model"].map(discrimination_ok)
    table["meets_target_fsr_pooled"] = (
        table["false_safe_rate_all_non_potable"] <= TARGET_FSR
    )
    table["coverage_gap_actual_minus_target"] = (
        table["actual_coverage"] - table["target_coverage"]
    )
    table["fsr_pooled_ci"] = table.apply(lambda r: percent_ci(
        r["false_safe_rate_all_non_potable"],
        r["fsr_ci_lower"], r["fsr_ci_upper"]), axis=1)
    table["fsr_fold_mean_sd"] = table.apply(lambda r: mean_sd(
        r["false_safe_rate_all_non_potable_mean"],
        r["false_safe_rate_all_non_potable_std"], as_percent=True), axis=1)

    def note(row):
        if row["n_declared_safe"] == 0:
            return "Declares nothing potable: FSR of 0 is degenerate."
        if not row["better_than_chance"]:
            return "Ranking ability not better than chance."
        return ""

    table["note"] = table.apply(note, axis=1)
    return order_rows(table, second="target_coverage")


def build_operating_points(threshold_table, selective_table):
    """
    Every learned-model operating point, flagged against the 5% target.
    Rows are kept in the fixed thesis order; nothing is ranked.
    """
    threshold_points = threshold_table.loc[~threshold_table["reference_only"]].assign(
        approach="threshold",
        setting=lambda d: d["strategy"].map(STRATEGY_LABELS),
        decision_coverage=1.0,
        n_declared_safe=lambda d: d["n_declared_potable"],
        fsr=lambda d: d["false_safe_rate"],
        recall_all=lambda d: d["potable_recall"],
    )
    selective_points = selective_table.assign(
        approach="selective",
        setting=lambda d: d["target_coverage"].map(lambda c: f"Target coverage {c:.0%}"),
        decision_coverage=lambda d: d["actual_coverage"],
        fsr=lambda d: d["false_safe_rate_all_non_potable"],
        recall_all=lambda d: d["potable_recall_all"],
    )
    columns = [
        "model", "approach", "setting", "decision_coverage", "n_declared_safe",
        "fsr", "fsr_ci_lower", "fsr_ci_upper", "recall_all",
        "recall_ci_lower", "recall_ci_upper", "better_than_chance",
        "meets_target_fsr_pooled", "note",
    ]
    points = pd.concat([threshold_points[columns], selective_points[columns]],
                       ignore_index=True)
    points["fsr_upper_ci_within_target"] = points["fsr_ci_upper"] <= TARGET_FSR
    points["meaningful_and_meets_target"] = (
        points["meets_target_fsr_pooled"]
        & points["better_than_chance"]
        & (points["n_declared_safe"] > 0)
    )
    points["model_order"] = points["model"].map(MODELS.index)
    points["approach_order"] = points["approach"].map(["threshold", "selective"].index)
    return (
        points.sort_values(["model_order", "approach_order"], kind="stable")
        .drop(columns=["model_order", "approach_order"])
        .reset_index(drop=True)
    )


def leader_rows(df, metric, direction, label, eligible_note,
                ci_lower=None, ci_upper=None):
    """
    Metric-wise leaders; intentionally no composite ranking.

    If the leader's 95% interval overlaps the runner-up's, the result is
    marked as not clearly separated.
    """
    work = df.loc[df[metric].notna()].copy()
    if work.empty:
        return []

    ascending = direction == "min"
    work = work.sort_values(metric, ascending=ascending)
    best = work[metric].iloc[0]
    winners = work[np.isclose(work[metric], best, rtol=1e-8, atol=1e-10)]
    others = work.drop(index=winners.index)

    rows = []
    for _, row in winners.iterrows():
        separation = "Only one eligible model"
        runner_up = ""
        if not others.empty:
            second = others.iloc[0]
            runner_up = f"{MODEL_LABELS[second['model']]} ({fmt(second[metric])})"
            if ci_lower and ci_upper:
                overlap = (
                    row[ci_lower] <= second[ci_upper]
                    and second[ci_lower] <= row[ci_upper]
                )
                separation = (
                    "95% intervals overlap (descriptive only)"
                    if overlap else "95% intervals do not overlap (not a formal test)"
                )
            else:
                separation = "No interval available"
        rows.append({
            "comparison": label,
            "metric": metric,
            "direction": "Higher is better" if direction == "max" else "Lower is better",
            "leader_model": row["model"],
            "strategy": row.get("strategy", ""),
            "target_coverage": row.get("target_coverage", np.nan),
            "value": float(best),
            "runner_up": runner_up,
            "separation": separation,
            "eligibility": eligible_note,
        })
    return rows


def build_metric_leaders(probability_table, threshold_table, selective_table):
    """Independent metric-wise leaders, not a composite winner."""
    rows = []
    sigmoid = probability_table.loc[
        (probability_table["probability_method"] == "sigmoid")
        & ~probability_table["reference_only"]
    ]
    all_models = "Learned models (baseline is reference-only)"
    rows += leader_rows(sigmoid, "roc_auc", "max", "Pooled sigmoid ROC-AUC",
                        all_models, "roc_auc_ci_lower", "roc_auc_ci_upper")
    rows += leader_rows(sigmoid, "average_precision", "max",
                        "Pooled sigmoid Average Precision", all_models)
    # A model that predicts close to the base rate for everyone is
    # trivially well calibrated, so calibration leaders must also rank
    # better than chance.
    informative = sigmoid.loc[sigmoid["better_than_chance"]]
    calibration_note = "Learned models better than chance (ROC-AUC CI > 0.5)"
    rows += leader_rows(informative, "brier_score", "min",
                        "Pooled sigmoid Brier score", calibration_note,
                        "brier_ci_lower", "brier_ci_upper")
    rows += leader_rows(informative, "log_loss", "min",
                        "Pooled sigmoid log loss", calibration_note)
    rows += leader_rows(informative, "ece", "min",
                        "Pooled sigmoid ECE", calibration_note)

    safety_note = (
        "Models better than chance (ROC-AUC CI > 0.5) that declare "
        "at least one sample potable"
    )
    threshold = threshold_table.loc[~threshold_table["reference_only"]]
    for strategy in THRESHOLD_STRATEGIES:
        subset = threshold.loc[threshold["strategy"] == strategy]
        safety_subset = subset.loc[
            subset["better_than_chance"] & (subset["n_declared_potable"] > 0)
        ]
        label = STRATEGY_LABELS[strategy]
        rows += leader_rows(safety_subset, "false_safe_rate", "min",
                            f"Threshold FSR — {label}", safety_note,
                            "fsr_ci_lower", "fsr_ci_upper")
        rows += leader_rows(subset, "potable_recall", "max",
                            f"Threshold potable recall — {label}", all_models,
                            "recall_ci_lower", "recall_ci_upper")

    for coverage in TARGET_COVERAGES:
        subset = selective_table.loc[np.isclose(selective_table["target_coverage"], coverage)]
        safety_subset = subset.loc[
            subset["better_than_chance"] & (subset["n_declared_safe"] > 0)
        ]
        label = f"{int(coverage * 100)}% target coverage"
        rows += leader_rows(safety_subset, "false_safe_rate_all_non_potable", "min",
                            f"Selective FSR — {label}", safety_note,
                            "fsr_ci_lower", "fsr_ci_upper")
        rows += leader_rows(subset, "accepted_error_rate", "min",
                            f"Selective accepted error — {label}", all_models)
        rows += leader_rows(subset, "potable_recall_all", "max",
                            f"Selective potable recall — {label}", all_models,
                            "recall_ci_lower", "recall_ci_upper")

    return pd.DataFrame(rows)


def build_thesis_ready_comparison(probability_table, threshold_table, selective_table):
    """One row per model. No composite score is calculated."""
    sigmoid = probability_table.loc[
        probability_table["probability_method"] == "sigmoid"
    ].set_index("model")

    rows = []
    for model in MODELS:
        p = sigmoid.loc[model]
        row = {
            "model": model,
            "reference_only": model in REFERENCE_ONLY_MODELS,
            "sigmoid_roc_auc_pooled_ci": p["roc_auc_pooled_ci"],
            "sigmoid_roc_auc_fold_mean_sd": p["roc_auc_fold_mean_sd"],
            "sigmoid_average_precision": round(p["average_precision"], 4),
            "sigmoid_average_precision_fold_mean_sd": p["average_precision_fold_mean_sd"],
            "sigmoid_brier_pooled_ci": p["brier_pooled_ci"],
            "sigmoid_brier_fold_mean_sd": p["brier_fold_mean_sd"],
            "sigmoid_log_loss": round(p["log_loss"], 4),
            "sigmoid_log_loss_fold_mean_sd": p["log_loss_fold_mean_sd"],
            "sigmoid_ece": round(p["ece"], 4),
            "sigmoid_ece_fold_mean_sd": p["ece_fold_mean_sd"],
            "better_than_chance": bool(p["better_than_chance"]),
        }
        for strategy in THRESHOLD_STRATEGIES:
            t = threshold_table.loc[
                (threshold_table["model"] == model)
                & (threshold_table["strategy"] == strategy)
            ].iloc[0]
            prefix = f"threshold_{strategy}"
            row[f"{prefix}_fsr"] = t["fsr_pooled_ci"]
            row[f"{prefix}_recall"] = percent(t["potable_recall"])
            row[f"{prefix}_precision"] = percent(t["positive_precision"])
            row[f"{prefix}_f1"] = round(float(t["f1_score"]), 4)
            row[f"{prefix}_n_declared_potable"] = int(t["n_declared_potable"])
            row[f"{prefix}_folds_meeting_target_fsr"] = int(
                t["folds_meeting_target_fsr"]
            )
            row[f"{prefix}_fold_threshold_mean"] = round(
                float(t["threshold_fold_mean"]), 6
            )
            row[f"{prefix}_fold_threshold_sd"] = round(
                float(t["threshold_fold_std"]), 6
            )
        for coverage in TARGET_COVERAGES:
            s = selective_table.loc[
                (selective_table["model"] == model)
                & np.isclose(selective_table["target_coverage"], coverage)
            ]
            prefix = f"selective_{int(coverage * 100)}pct"
            if len(s) == 1:
                s = s.iloc[0]
                row[f"{prefix}_actual_coverage"] = percent(s["actual_coverage"])
                row[f"{prefix}_review_share"] = percent(s["review_share"])
                row[f"{prefix}_accepted_error"] = percent(s["accepted_error_rate"])
                row[f"{prefix}_fsr"] = s["fsr_pooled_ci"]
                row[f"{prefix}_recall_all"] = percent(s["potable_recall_all"])
                row[f"{prefix}_precision_decided"] = percent(
                    s["precision_among_decided"]
                )
                row[f"{prefix}_n_declared_safe"] = int(s["n_declared_safe"])
                row[f"{prefix}_folds_meeting_target_fsr"] = int(
                    s["folds_meeting_target_fsr"]
                )
            else:
                row[f"{prefix}_actual_coverage"] = "reference only"
                row[f"{prefix}_review_share"] = "reference only"
                row[f"{prefix}_accepted_error"] = "reference only"
                row[f"{prefix}_fsr"] = "reference only"
                row[f"{prefix}_recall_all"] = "reference only"
                row[f"{prefix}_precision_decided"] = "reference only"
                row[f"{prefix}_n_declared_safe"] = "reference only"
                row[f"{prefix}_folds_meeting_target_fsr"] = "reference only"
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 7. CREATE FIGURES
# ============================================================

def save_figure(fig, figure_dir, name):
    """Save one figure as PNG (300 dpi) and PDF; return both paths."""
    paths = [figure_dir / f"{name}.png", figure_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def add_target_line(ax, orientation="vertical"):
    if orientation == "vertical":
        ax.axvline(100 * TARGET_FSR, color=TARGET_LINE_COLOR,
                   linestyle="--", linewidth=1.2, zorder=1)
    else:
        ax.axhline(100 * TARGET_FSR, color=TARGET_LINE_COLOR,
                   linestyle="--", linewidth=1.2, zorder=1)


def plot_probability_quality(probability_table, figure_dir):
    """ROC-AUC, Brier and ECE as dot plots with 95% intervals."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
    offsets = {"uncalibrated": -0.22, "sigmoid": 0.0, "isotonic": 0.22}
    markers = {"uncalibrated": "o", "sigmoid": "D", "isotonic": "s"}

    for model_index, model in enumerate(MODELS):
        for method in PROBABILITY_METHODS:
            r = probability_table.loc[
                (probability_table["model"] == model)
                & (probability_table["probability_method"] == method)
            ].iloc[0]
            y = model_index + offsets[method]
            color = MODEL_COLORS[model]
            panels = [
                (axes[0], r["roc_auc"], r["roc_auc_ci_lower"], r["roc_auc_ci_upper"]),
                (axes[1], r["brier_score"], r["brier_ci_lower"], r["brier_ci_upper"]),
                (axes[2], r["ece"], np.nan, np.nan),
            ]
            for ax, value, lo, hi in panels:
                if not pd.isna(lo):
                    ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, alpha=0.7)
                ax.scatter(value, y, marker=markers[method], s=26, color=color,
                           edgecolor="white", linewidth=1, zorder=3)

    axes[0].axvline(0.5, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
    axes[0].text(0.503, -0.45, "Chance", fontsize=7.5, color=TEXT_SECONDARY)
    axes[0].set_xlabel("ROC-AUC (higher is better)")
    axes[1].set_xlabel("Brier score (lower is better)")
    axes[2].set_xlabel("ECE, 10 bins (lower is better)")
    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], marker=markers[m], linestyle="", color=TEXT_SECONDARY,
                   label=m.capitalize())
        for m in PROBABILITY_METHODS
    ]
    axes[2].legend(handles=handles, loc="lower right", title="Probabilities",
                   title_fontsize=8)
    fig.suptitle("Probability quality on pooled outer test "
                 "(lines = bootstrap 95% CI, 2000 resamples)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, figure_dir, "10_probability_quality")


def plot_average_precision(probability_table, figure_dir):
    """Show fold mean +/- SD and pooled OOF Average Precision separately."""
    sigmoid = probability_table.loc[
        probability_table["probability_method"] == "sigmoid"
    ].copy()
    sigmoid = sigmoid.set_index("model").loc[MODELS].reset_index()

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y_positions = np.arange(len(MODELS))

    for i, row in sigmoid.iterrows():
        model = row["model"]
        mean_value = float(row["average_precision_mean"])
        sd_value = 0.0 if pd.isna(row["average_precision_std"]) else float(
            row["average_precision_std"]
        )
        pooled_value = float(row["average_precision"])

        ax.errorbar(
            mean_value, y_positions[i],
            xerr=sd_value,
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            capsize=3,
            markersize=6,
            label="Fold mean ± SD" if i == 0 else None,
        )
        ax.scatter(
            pooled_value, y_positions[i],
            marker="D",
            s=52,
            facecolor="white",
            edgecolor=MODEL_COLORS[model],
            linewidth=1.5,
            zorder=4,
            label="Pooled OOF" if i == 0 else None,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average Precision (higher is better)")
    ax.set_title(
        "Average Precision: fold distribution vs pooled outer-test value",
        loc="left",
    )
    ax.legend(loc="lower right")
    return save_figure(fig, figure_dir, "10_average_precision")


def plot_threshold_risk_utility(threshold_table, figure_dir):
    """Plot discrete threshold operating points without implying interpolation."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    add_target_line(ax)
    ax.text(
        100 * TARGET_FSR + 0.3,
        97,
        f"Target FSR {100 * TARGET_FSR:.0f}%",
        color=TARGET_LINE_COLOR,
        fontsize=8,
        va="top",
    )

    for model in NON_BASELINE_MODELS:
        subset = threshold_table.loc[
            threshold_table["model"] == model
        ].set_index("strategy").loc[THRESHOLD_STRATEGIES]

        for strategy in THRESHOLD_STRATEGIES:
            row = subset.loc[strategy]
            x = 100 * float(row["false_safe_rate"])
            y = 100 * float(row["potable_recall"])
            ax.scatter(
                x,
                y,
                s=62,
                marker=STRATEGY_MARKERS[strategy],
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )

    model_handles = [
        plt.Line2D(
            [], [], color=MODEL_COLORS[m], marker="o", linestyle="",
            markersize=6, label=MODEL_LABELS[m]
        )
        for m in NON_BASELINE_MODELS
    ]
    strategy_handles = [
        plt.Line2D(
            [], [], color=TEXT_SECONDARY, marker=STRATEGY_MARKERS[s],
            linestyle="", markersize=6, label=STRATEGY_LABELS[s]
        )
        for s in THRESHOLD_STRATEGIES
    ]
    ax.legend(handles=model_handles + strategy_handles, loc="best", ncol=2)

    x_values = 100 * threshold_table.loc[
        threshold_table["model"].isin(NON_BASELINE_MODELS), "false_safe_rate"
    ].to_numpy(dtype=float)
    y_values = 100 * threshold_table.loc[
        threshold_table["model"].isin(NON_BASELINE_MODELS), "potable_recall"
    ].to_numpy(dtype=float)
    xmax = min(100.0, max(10.0, float(np.nanmax(x_values)) * 1.12 + 1.0))
    ymax = min(100.0, max(10.0, float(np.nanmax(y_values)) * 1.12 + 1.0))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("False-safe rate, FP / (FP + TN) (%)  ← safer")
    ax.set_ylabel("Potable recall (%)  ↑ more useful")
    ax.set_title(
        "Risk–utility trade-off of threshold strategies\\n"
        "(pooled outer test, five folds)",
        loc="left",
    )
    return save_figure(fig, figure_dir, "10_threshold_false_safe_vs_recall")


def plot_selective_risk_utility(selective_table, threshold_table, figure_dir):
    """Compare all selective operating points with each model's no-abstention rule."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    add_target_line(ax)
    ax.text(
        100 * TARGET_FSR + 0.3,
        97,
        f"Target FSR {100 * TARGET_FSR:.0f}%",
        color=TARGET_LINE_COLOR,
        fontsize=8,
        va="top",
    )

    plotted_models = []
    for model in NON_BASELINE_MODELS:
        subset = selective_table.loc[
            selective_table["model"] == model
        ].sort_values("target_coverage")
        if subset.empty:
            continue

        plotted_models.append(model)
        x = 100 * subset["false_safe_rate_all_non_potable"].to_numpy(dtype=float)
        y = 100 * subset["potable_recall_all"].to_numpy(dtype=float)

        ax.plot(
            x, y,
            color=MODEL_COLORS[model],
            linewidth=1.8,
            alpha=0.75,
            label=f"{MODEL_LABELS[model]} (selective)",
        )

        for _, row in subset.iterrows():
            marker = "o" if int(row["n_declared_safe"]) > 0 else "x"
            ax.scatter(
                100 * row["false_safe_rate_all_non_potable"],
                100 * row["potable_recall_all"],
                marker=marker,
                s=58,
                color=MODEL_COLORS[model],
                edgecolor="white" if marker == "o" else MODEL_COLORS[model],
                linewidth=1.0,
                zorder=3,
            )
            ax.annotate(
                f"{row['target_coverage']:.0%}",
                (
                    100 * row["false_safe_rate_all_non_potable"],
                    100 * row["potable_recall_all"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=TEXT_SECONDARY,
            )

        # Diamond = Clopper–Pearson threshold strategy (no abstention).
        cp = threshold_table.loc[
            (threshold_table["model"] == model)
            & (threshold_table["strategy"] == "clopper_pearson")
        ]
        if len(cp) == 1:
            cp = cp.iloc[0]
            ax.scatter(
                100 * cp["false_safe_rate"],
                100 * cp["potable_recall"],
                marker="D",
                s=54,
                facecolor="white",
                edgecolor=MODEL_COLORS[model],
                linewidth=1.6,
                zorder=4,
            )

    handles, _ = ax.get_legend_handles_labels()
    handles.append(
        plt.Line2D(
            [], [], marker="D", linestyle="", markerfacecolor="white",
            markeredgecolor=TEXT_SECONDARY,
            label="Clopper–Pearson threshold (no abstention)",
        )
    )
    handles.append(
        plt.Line2D(
            [], [], marker="x", linestyle="", color=TEXT_SECONDARY,
            label="No potable declarations at that setting",
        )
    )
    ax.legend(handles=handles, loc="best", fontsize=7.5)

    x_values = 100 * selective_table["false_safe_rate_all_non_potable"].to_numpy(dtype=float)
    y_values = 100 * selective_table["potable_recall_all"].to_numpy(dtype=float)
    xmax = min(100.0, max(10.0, float(np.nanmax(x_values)) * 1.12 + 1.0))
    ymax = min(100.0, max(10.0, float(np.nanmax(y_values)) * 1.12 + 1.0))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("False-safe rate, FP / all actual non-potable (%)  ← safer")
    ax.set_ylabel("Potable recall over all rows (%)  ↑ more useful")
    ax.set_title("Risk–utility trade-off of selective prediction", loc="left")
    return save_figure(fig, figure_dir, "10_selective_false_safe_vs_recall")


def plot_selective_risk_coverage(selective_table, majority_accuracy, figure_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for model in NON_BASELINE_MODELS:
        subset = selective_table.loc[selective_table["model"] == model]
        coverage = 100 * subset["actual_coverage"].to_numpy()
        style = dict(color=MODEL_COLORS[model], linewidth=2, marker="o",
                     markersize=5, markeredgecolor="white",
                     markeredgewidth=1.2, label=MODEL_LABELS[model])
        axes[0].plot(coverage, 100 * subset["false_safe_rate_all_non_potable"], **style)
        axes[1].plot(coverage, 100 * subset["accepted_error_rate"], **style)

    add_target_line(axes[0], orientation="horizontal")
    axes[0].text(2, 100 * TARGET_FSR + 0.4, "Target 5%",
                 color=TARGET_LINE_COLOR, fontsize=7.5)
    axes[0].set_ylabel("False-safe rate (%)")
    axes[0].set_title("(a) False-safe risk rises with coverage", loc="left")

    majority_error = 1.0 - majority_accuracy
    axes[1].axhline(100 * majority_error, color=TEXT_SECONDARY, linewidth=1,
                    linestyle=":")
    axes[1].text(2, 100 * majority_error - 1.8,
                 f"Majority-class error {100 * majority_error:.1f}%",
                 color=TEXT_SECONDARY, fontsize=7.5)
    axes[1].set_ylabel("Error among decided rows (%)")
    axes[1].set_title("(b) Accepted error rises with coverage", loc="left")
    for ax in axes:
        ax.set_xlabel("Actual decision coverage on test (%)")
        ax.set_xlim(0, 105)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, figure_dir, "10_selective_risk_vs_coverage")


def plot_actual_vs_target_coverage(selective_table, figure_dir):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 105], [0, 105], color=TEXT_SECONDARY, linestyle=":",
            linewidth=1, label="Actual = target")
    for model in NON_BASELINE_MODELS:
        subset = selective_table.loc[selective_table["model"] == model]
        ax.plot(100 * subset["target_coverage"], 100 * subset["actual_coverage"],
                color=MODEL_COLORS[model], linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.2,
                label=MODEL_LABELS[model])
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Target coverage, chosen on calibration data (%)")
    ax.set_ylabel("Actual outer-test coverage (%)")
    ax.set_title("Target vs achieved decision coverage", loc="left")
    ax.legend(loc="upper left")
    return save_figure(fig, figure_dir, "10_selective_actual_vs_target_coverage")


def plot_fsr_intervals(threshold_table, selective_table, figure_dir):
    rows = []
    for model in NON_BASELINE_MODELS:
        for strategy in THRESHOLD_STRATEGIES:
            r = threshold_table.loc[
                (threshold_table["model"] == model)
                & (threshold_table["strategy"] == strategy)
            ].iloc[0]
            rows.append((model, f"Threshold: {STRATEGY_LABELS[strategy]}",
                         r["false_safe_rate"], r["fsr_ci_lower"],
                         r["fsr_ci_upper"], r["n_declared_potable"]))
        for _, r in selective_table.loc[selective_table["model"] == model].iterrows():
            rows.append((model, f"Selective: {r['target_coverage']:.0%} coverage",
                         r["false_safe_rate_all_non_potable"], r["fsr_ci_lower"],
                         r["fsr_ci_upper"], r["n_declared_safe"]))

    fig, ax = plt.subplots(figsize=(6.8, 7.6))
    y, y_positions, labels = 0, [], []
    for i, (model, setting, fsr, lo, hi, n_safe) in enumerate(rows):
        if i > 0 and rows[i - 1][0] != model:
            y += 1.0
        y_positions.append(y)
        labels.append(setting)
        ax.plot([100 * lo, 100 * hi], [y, y], color=MODEL_COLORS[model],
                linewidth=2, solid_capstyle="round")
        ax.scatter(100 * fsr, y, color=MODEL_COLORS[model], s=30,
                   edgecolor="white", linewidth=1.2, zorder=3)
        if n_safe < 10:
            ax.text(100 * hi + 0.4, y,
                    "no row declared safe" if n_safe == 0
                    else f"only {int(n_safe)} rows declared safe",
                    fontsize=7, color=TEXT_SECONDARY, va="center")
        y += 1

    for model in NON_BASELINE_MODELS:
        first = next(i for i, r in enumerate(rows) if r[0] == model)
        ax.text(-0.5, y_positions[first] - 0.9, MODEL_LABELS[model],
                fontsize=9, fontweight="bold", color=TEXT_PRIMARY)

    add_target_line(ax)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    interval_max = max(float(r[4]) for r in rows) * 100 if rows else 0.0
    xmax = min(100.0, max(10.0, interval_max * 1.15 + 1.0))
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Pooled outer-test false-safe rate with exact 95% CI (%)")
    ax.set_title("False-safe rate of every operating point "
                 "(dashed line = 5% target)", loc="left")
    return save_figure(fig, figure_dir, "10_false_safe_rate_intervals")


def plot_roc_and_reliability(predictions, figure_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    y_true = predictions["y_true"].to_numpy()
    for model in NON_BASELINE_MODELS:
        p = predictions[f"{model}_sigmoid_probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        axes[0].plot(fpr, tpr, color=MODEL_COLORS[model], linewidth=2,
                     label=f"{MODEL_LABELS[model]} ({auc:.3f})")
        observed, predicted = calibration_curve(
            y_true, p, n_bins=N_RELIABILITY_BINS, strategy="quantile")
        axes[1].plot(predicted, observed, color=MODEL_COLORS[model], linewidth=2,
                     marker="o", markersize=4, markeredgecolor="white",
                     label=MODEL_LABELS[model])

    for ax in axes:
        ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linestyle=":", linewidth=1)
    axes[0].set_xlabel("False-positive rate (= FSR)")
    axes[0].set_ylabel("Potable recall")
    axes[0].set_title("(a) Pooled ROC, sigmoid (AUC)", loc="left")
    axes[0].legend(loc="lower right")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Mean predicted probability (quantile bins)")
    axes[1].set_ylabel("Observed potable share")
    axes[1].set_title("(b) Reliability, sigmoid", loc="left")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    return save_figure(fig, figure_dir, "10_pooled_roc_and_reliability")


# ============================================================
# 8. GENERATE A DATA-DRIVEN RESULTS REPORT
# ============================================================

def build_markdown_report(raw, fold_counts, probability_table, threshold_table,
                          selective_table, operating_points, leaders):
    sigmoid = probability_table.loc[probability_table["probability_method"] == "sigmoid"]
    rf = sigmoid.loc[sigmoid["model"] == "random_forest"].iloc[0]
    mlp = sigmoid.loc[sigmoid["model"] == "mlp"].iloc[0]
    auc_overlap = (
        rf["roc_auc_ci_lower"] <= mlp["roc_auc_ci_upper"]
        and mlp["roc_auc_ci_lower"] <= rf["roc_auc_ci_upper"]
    )
    not_better = sigmoid.loc[
        ~sigmoid["better_than_chance"] & ~sigmoid["reference_only"], "model"
    ].map(MODEL_LABELS).tolist()

    probability_view = sigmoid[[
        "model", "roc_auc_pooled_ci", "roc_auc_fold_mean_sd",
        "average_precision", "average_precision_fold_mean_sd",
        "brier_pooled_ci", "brier_fold_mean_sd",
        "log_loss", "log_loss_fold_mean_sd", "ece", "ece_fold_mean_sd",
    ]].round(4)
    threshold_view = threshold_table[[
        "model", "strategy", "mean_applied_threshold",
        "threshold_fold_mean", "threshold_fold_std",
        "n_declared_potable", "fsr_pooled_ci", "fsr_fold_mean_sd",
        "folds_meeting_target_fsr", "potable_recall",
        "positive_precision", "f1_score",
    ]].round(4)
    selective_view = selective_table[[
        "model", "target_coverage", "actual_coverage", "review_share",
        "n_declared_safe", "fsr_pooled_ci", "folds_meeting_target_fsr",
        "potable_recall_all", "accepted_error_rate",
    ]].round(4)
    target_view = operating_points.loc[
        operating_points["meaningful_and_meets_target"],
        ["model", "approach", "setting", "decision_coverage", "fsr",
         "fsr_ci_upper", "fsr_upper_ci_within_target", "recall_all"],
    ].round(4)

    lines = [
        "# Final Results Analysis — Water Potability Thesis",
        "",
        "## Evaluation scope",
        "",
        f"- Raw dataset rows: **{len(raw):,}**",
        f"- Outer-test folds: **{len(EXPECTED_FOLDS)}**",
        "- Evaluation basis: untouched outer-test predictions from Lesson 09",
        "- Positive class: Potable (1); negative class: Non-potable (0)",
        "- Primary false-safe rate: `FP / (FP + TN)` (verified from saved counts)",
        f"- Target false-safe rate used in Lesson 07: **{100 * TARGET_FSR:.0f}%**",
        "- No model, threshold, calibrator, or selective cut-off was modified.",
        "- Nominal 95% intervals: stratified sample bootstrap (ROC-AUC, Brier; "
        "2000 resamples, seed 42) and exact Clopper–Pearson (FSR, recall).",
        "- These intervals are descriptive summaries conditional on the pooled "
        "out-of-fold predictions. They do not capture the full variation from "
        "retraining across folds and are not formal paired model-comparison tests.",
        "",
        "### Outer-test rows by fold",
        "",
        dataframe_to_markdown(fold_counts),
        "",
        "## 1. Probability model comparison (sigmoid)",
        "",
        dataframe_to_markdown(probability_view),
        "",
        f"- Random forest ROC-AUC {rf['roc_auc_pooled_ci']} vs MLP "
        f"{mlp['roc_auc_pooled_ci']}: "
        + ("their descriptive 95% intervals overlap."
           if auc_overlap else "their descriptive 95% intervals do not overlap."),
        "- Interval overlap is only a visual descriptive comparison; it is not "
        "a formal paired test of model differences.",
        f"- Models whose ROC-AUC interval includes 0.5 (no clear evidence of ranking "
        f"ability): {', '.join(not_better) if not_better else 'None'}.",
        "",
        "## 2. Threshold-selection results",
        "",
        dataframe_to_markdown(threshold_view),
        "",
        "`folds_meeting_target_fsr` counts how many of the five outer-test folds "
        "kept FSR at or below the target. A pooled FSR below the target does not "
        "guarantee every fold met it.",
        "",
        "## 3. Selective-prediction results",
        "",
        dataframe_to_markdown(selective_view),
        "",
        "Target coverage is chosen on calibration data; actual coverage is measured "
        "on the untouched outer test, so they differ slightly. Recall counts rows "
        "sent to Needs Review as missed, and review workload must be reported "
        "alongside error. At 100% coverage nothing is abstained.",
        "",
        "## 4. Operating points meeting the 5% target",
        "",
        "Learned-model operating points with pooled FSR ≤ 5%, ranking ability "
        "better than chance and at least one potable declaration. Listed in fixed "
        "model order, not ranked. `fsr_upper_ci_within_target` separately shows "
        "whether the descriptive 95% interval's upper limit is also at or below 5%; "
        "this is stricter than the pooled point-estimate target.",
        "",
        dataframe_to_markdown(target_view),
        "",
        "## 5. Metric-wise leaders",
        "",
        "Best observed value for each metric separately. This is not a composite "
        "ranking or a declaration of one universally best model. `separation` "
        "summarises descriptive interval overlap; it is not a formal paired test.",
        "",
        dataframe_to_markdown(leaders.drop(columns=["eligibility"]).round(4)),
        "",
        "## 6. Methodological interpretation and limitations",
        "",
        "1. Pooled metrics and fold mean ± SD answer different questions: pooled "
        "metrics combine all outer-test predictions; mean ± SD describes variation "
        "across the five fold-level estimates.",
        "2. The five outer-test folds together cover the raw dataset exactly once.",
        "3. Thresholds and selective cut-offs were not selected using outer-test labels.",
        "4. ECE is descriptive and depends on the binning scheme used in Lesson 09.",
        "5. The bootstrap and Clopper–Pearson intervals are nominal descriptive "
        "intervals for pooled out-of-fold predictions. They do not model the "
        "dependence induced by overlapping outer-fold training sets or include "
        "full model-refitting uncertainty; interval overlap is not a formal paired test.",
        "6. A low false-safe rate alone does not establish a useful classifier. "
        "A model that cannot rank better than chance, or declares (almost) nothing "
        "potable, reaches a low FSR trivially. Such models are excluded from the "
        "FSR leader rows and flagged in every table.",
        "7. `unsafe_share_among_declared_potable` = FP / (TP + FP) is a different "
        "denominator and is reported only as a supplementary quantity.",
        "8. These are model-based estimates on one public dataset and split design. "
        "They do not certify real-world water safety or guarantee future risk.",
        "9. Any final model recommendation should follow the thesis objective and "
        "pre-specified safety/utility priorities, not a post-hoc composite score.",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# 9. MAIN WORKFLOW
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create thesis-ready summary tables and figures from the "
            "five-fold outer-test evaluation. Does not refit or retune."
        )
    )
    parser.add_argument(
        "--project-dir", type=str, default=None,
        help="Optional project root. Default: parent of this script's 02_Code folder.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace this script's own earlier 10_ outputs.",
    )
    args = parser.parse_args()

    project_dir = (
        Path(args.project_dir).expanduser().resolve()
        if args.project_dir else Path(__file__).resolve().parents[1]
    )
    table_dir = project_dir / "04_Results" / "tables"
    figure_dir = project_dir / "04_Results" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("LESSON 10 — FINAL RESULTS ANALYSIS")
    print("=" * 72)
    print("Project:", project_dir)
    print("Reading Lesson 09 outputs; no model fitting or threshold selection.")
    print("Primary FSR: FP / (FP + TN); target:", TARGET_FSR)

    check_existing_outputs(table_dir, figure_dir, args.overwrite)

    raw, raw_path, predictions, fold_counts = validate_prediction_coverage(
        project_dir, table_dir)
    data = load_evaluation_outputs(table_dir, predictions)

    input_paths = [raw_path] + [table_dir / name for name in INPUT_FILES]
    hashes_before = {str(p.relative_to(project_dir)): sha256_file(p)
                     for p in input_paths}

    print("\n===== INPUT VALIDATION =====")
    print("Raw dataset rows:", len(raw))
    print("Outer-test prediction rows:", len(predictions))
    print("Fold IDs:", sorted(predictions["fold"].astype(int).unique().tolist()))
    print("Saved labels match raw dataset: PASSED")
    print("Five-fold full row coverage: PASSED")
    print("Probability / threshold / selective result grids: PASSED")
    print("Primary FSR = FP / (FP + TN) verified from counts: PASSED")

    # ---------- Tables ----------
    probability_table = build_probability_table(data, predictions)
    sigmoid = probability_table.loc[probability_table["probability_method"] == "sigmoid"]
    discrimination_ok = dict(zip(sigmoid["model"], sigmoid["better_than_chance"]))

    threshold_table = build_threshold_table(data, discrimination_ok)
    selective_table = build_selective_table(data, discrimination_ok)
    operating_points = build_operating_points(threshold_table, selective_table)
    leaders = build_metric_leaders(probability_table, threshold_table, selective_table)
    thesis_comparison = build_thesis_ready_comparison(
        probability_table, threshold_table, selective_table)

    tables = {
        "10_probability_quality.csv": probability_table,
        "10_threshold_comparison.csv": threshold_table,
        "10_selective_comparison.csv": selective_table,
        "10_operating_points_meeting_target.csv": operating_points,
        "10_metric_leaders.csv": leaders,
        "10_thesis_ready_model_comparison.csv": thesis_comparison,
    }
    output_paths = []
    for filename, frame in tables.items():
        path = table_dir / filename
        frame.to_csv(path, index=False)
        output_paths.append(path)

    # ---------- Figures ----------
    class_shares = predictions["y_true"].value_counts(normalize=True)
    majority_accuracy = float(class_shares.max())
    output_paths += plot_probability_quality(probability_table, figure_dir)
    output_paths += plot_average_precision(probability_table, figure_dir)
    output_paths += plot_threshold_risk_utility(threshold_table, figure_dir)
    output_paths += plot_selective_risk_utility(selective_table, threshold_table, figure_dir)
    output_paths += plot_selective_risk_coverage(selective_table, majority_accuracy, figure_dir)
    output_paths += plot_actual_vs_target_coverage(selective_table, figure_dir)
    output_paths += plot_fsr_intervals(threshold_table, selective_table, figure_dir)
    output_paths += plot_roc_and_reliability(predictions, figure_dir)

    # ---------- Report ----------
    report = build_markdown_report(raw, fold_counts, probability_table,
                                   threshold_table, selective_table,
                                   operating_points, leaders)
    report_path = table_dir / "10_final_results_report.md"
    report_path.write_text(report, encoding="utf-8")
    output_paths.append(report_path)

    # ---------- Input integrity and audit manifest ----------
    hashes_after = {str(p.relative_to(project_dir)): sha256_file(p)
                    for p in input_paths}
    if hashes_before != hashes_after:
        raise AssertionError("An input file changed during the analysis.")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_fsr_definition": "FP / (FP + TN)",
        "target_fsr": TARGET_FSR,
        "models_retrained": False,
        "thresholds_or_cutoffs_reselected": False,
        "composite_ranking": False,
        "reference_only_models": sorted(REFERENCE_ONLY_MODELS),
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                      "stratified_by_class": True},
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": hashes_after,
        "output_sha256": {str(p.relative_to(project_dir)): sha256_file(p)
                          for p in output_paths},
    }
    manifest_path = table_dir / "10_final_results_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    # ---------- Console summary ----------
    print("\n===== PROBABILITY QUALITY (SIGMOID, POOLED) =====")
    print(sigmoid[["model", "roc_auc_pooled_ci", "brier_pooled_ci",
                   "better_than_chance"]].to_string(index=False))
    print("\n===== THRESHOLD STRATEGIES (POOLED) =====")
    print(threshold_table[["model", "strategy", "n_declared_potable",
                           "fsr_pooled_ci", "potable_recall"]]
          .round(4).to_string(index=False))
    print("\n===== SELECTIVE PREDICTION (POOLED) =====")
    print(selective_table[["model", "target_coverage", "actual_coverage",
                           "fsr_pooled_ci", "potable_recall_all"]]
          .round(4).to_string(index=False))
    print("\n===== METRIC-WISE LEADERS (NOT A COMPOSITE RANKING) =====")
    print(leaders[["comparison", "leader_model", "value", "separation"]]
          .round(4).to_string(index=False))

    print("\n===== OUTPUTS CREATED =====")
    for path in output_paths + [manifest_path]:
        print(path.relative_to(project_dir))

    print("\nInput SHA-256 verification: PASSED")
    print("\n" + "=" * 72)
    print("LESSON 10 COMPLETED")
    print("=" * 72)
    print("All five outer-test folds were validated.")
    print("No model was trained or modified.")
    print("No threshold or selective cut-off was changed.")
    print("All Lesson 01-09 files were read-only.")
    print("No single composite model winner was imposed.")
    print("New outputs use the 10_ prefix.")


if __name__ == "__main__":
    main()
