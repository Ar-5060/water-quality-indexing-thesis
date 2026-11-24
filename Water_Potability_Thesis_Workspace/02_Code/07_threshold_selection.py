"""
LESSON 08: THRESHOLD SELECTION — FINAL THESIS-CONSISTENT VERSION

Purpose
-------
Select potable-decision thresholds using ONLY out-of-fold sigmoid-calibrated
probabilities from each outer fold's calibration subset.

Positive class: Potable = 1
Negative class: Non-potable = 0
False-safe event: actual 0 predicted 1

PRIMARY THESIS RISK METRIC
--------------------------
False-safe rate = FP / (all actual non-potable)
                 = FP / (FP + TN)

This is the false-positive rate among actual non-potable samples.

SUPPLEMENTARY METRIC
--------------------
Unsafe share among declared-potable predictions = FP / (TP + FP)
This is 1 - positive precision (also called the false discovery proportion).
It is reported separately and is NOT used for threshold selection in this
thesis version, because the governing thesis handover defines the primary
false-safe denominator as all actual non-potable samples.

Research design
---------------
1. Read calibrated_predictions_fold_k.csv.
2. Use only *_sigmoid_oof columns. Never use *_sigmoid_full_fit here.
3. Compare:
   a. Fixed threshold 0.50 (reference only; no risk constraint).
   b. Empirical threshold: lowest candidate with observed primary FSR <= target.
   c. Clopper-Pearson: lowest candidate with one-sided 95% upper bound <= target.
   d. Bonferroni: same, using alpha / 202 for the 202-candidate threshold family.
4. Main target: 5%. Sensitivity targets: 10%, 20%, 30%.
5. Candidate family: 201 cutoffs from 0.000 to 1.000 in 0.005 steps,
   plus one reject-all cutoff just above 1.000 (202 total).
6. The majority-class baseline is reference-only. Because its probability
   is constant, risk-based threshold optimization can only produce an
   all-negative or all-positive decision. Its numeric 0.50 reference
   threshold is retained for downstream schema compatibility.
7. If no candidate satisfies a confidence-bound rule for a non-baseline
   model, save the reject-all cutoff as a conservative operational fallback
   and mark feasible=False.
8. The outer test fold is never loaded or used. The saved main-target
   thresholds are intended for 09_model_evaluation.py.

Statistical scope
-----------------
Bonferroni correction covers the 202 candidate thresholds within EACH model's
threshold family. It does not simultaneously correct across all four models.
A calibration-side <=5% result is not a guarantee of <=5% future test risk.
The interpretation also relies on valid labels, model/calibration separation,
and exchangeability between calibration and future observations.

Outputs
-------
03_Experiments/fold_k/thresholds.json
    Downstream-compatible numeric threshold dictionary for 09_model_evaluation.py

04_Results/tables/threshold_grid_fold_k.csv
    All 202 candidates per model, counts, rates, bounds and pass indicators

04_Results/tables/threshold_selection_fold_k.csv
    Selected strategies for all sensitivity targets

04_Results/tables/threshold_selection_summary.csv
    Created with --all

Usage
-----
python 02_Code/07_threshold_selection.py
python 02_Code/07_threshold_selection.py --fold 3
python 02_Code/07_threshold_selection.py --all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_DIR / "03_Experiments"
TABLE_DIR = PROJECT_DIR / "04_Results" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PREDECLARED RESEARCH SETTINGS
# ============================================================

N_OUTER_SPLITS = 5

# Primary thesis target and sensitivity targets.
TARGET_FALSE_SAFE = 0.05
SENSITIVITY_TARGETS = [0.05, 0.10, 0.20, 0.30]

# One-sided confidence level = 1 - FAMILYWISE_ALPHA.
FAMILYWISE_ALPHA = 0.05

FIXED_THRESHOLD = 0.50

# 201 ordinary cutoffs: 0.000, 0.005, ..., 1.000.
N_ORDINARY_THRESHOLDS = 201

# One additional cutoff just above 1.0 predicts every sample as non-potable.
N_CANDIDATES = 202

BONFERRONI_ALPHA = FAMILYWISE_ALPHA / N_CANDIDATES

ORDINARY_THRESHOLDS = np.linspace(
    0.0, 1.0, N_ORDINARY_THRESHOLDS
)
REJECT_ALL_THRESHOLD = np.nextafter(1.0, np.inf)

CANDIDATE_THRESHOLDS = np.concatenate([
    ORDINARY_THRESHOLDS,
    np.array([REJECT_ALL_THRESHOLD]),
])

assert len(CANDIDATE_THRESHOLDS) == N_CANDIDATES

PROBABILITY_SUFFIX = "_sigmoid_oof"

MODELS = [
    "majority_baseline",
    "logistic_regression",
    "random_forest",
    "mlp",
]

STRATEGIES = [
    "fixed_0.50",
    "empirical",
    "clopper_pearson",
    "bonferroni",
]


# ============================================================
# 3. VALIDATION HELPERS
# ============================================================

def validate_target(values):
    """Require a non-empty binary target containing both classes."""
    numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy()

    if len(numeric) == 0:
        raise ValueError("The calibration prediction file is empty.")

    if not np.isfinite(numeric).all():
        raise ValueError("Target contains NaN or infinite values.")

    if not np.isin(numeric, [0, 1]).all():
        raise ValueError("Target must contain only binary labels 0 and 1.")

    y = numeric.astype(int)

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "Calibration data must contain both classes (0 and 1)."
        )

    return y


def validate_probabilities(values, model_name):
    """Require finite probability values in the closed interval [0, 1]."""
    p = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(
        dtype=float
    )

    if not np.isfinite(p).all():
        raise ValueError(
            f"{model_name}: probabilities contain NaN or infinity."
        )

    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError(
            f"{model_name}: probabilities must be between 0 and 1."
        )

    return p


# ============================================================
# 4. LOAD ONE FOLD AND CHECK DATA SEPARATION
# ============================================================

def load_fold_data(fold):
    """Load calibration OOF probabilities and verify test-fold separation."""
    prediction_path = (
        TABLE_DIR / f"calibrated_predictions_fold_{fold}.csv"
    )
    manifest_path = (
        EXPERIMENT_DIR / f"fold_{fold}" / "split_manifest.json"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Calibrated predictions not found:\n{prediction_path}\n"
            f"Run 06_probability_calibration.py for fold {fold} first."
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found:\n{manifest_path}"
        )

    predictions = pd.read_csv(prediction_path)

    required_columns = ["row_index", "y_true"] + [
        f"{model}{PROBABILITY_SUFFIX}" for model in MODELS
    ]

    missing_columns = [
        column for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            "Run 06_probability_calibration.py and check that the "
            "*_sigmoid_oof columns were saved."
        )

    if predictions[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in required target/probability columns."
        )

    if predictions["row_index"].duplicated().any():
        raise ValueError("Duplicate row_index values found.")

    row_indices = pd.to_numeric(
        predictions["row_index"], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(row_indices).all():
        raise ValueError("row_index contains NaN or infinite values.")

    if not np.equal(row_indices, np.floor(row_indices)).all():
        raise ValueError("row_index must contain integer row identifiers.")

    calibration_ids = set(row_indices.astype(int).tolist())
    y_true = validate_target(predictions["y_true"].to_numpy())

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if "test_indices" not in manifest:
        raise KeyError(
            "split_manifest.json does not contain 'test_indices'."
        )

    test_indices = manifest["test_indices"]

    if not isinstance(test_indices, list):
        raise TypeError("'test_indices' in split_manifest.json must be a list.")

    test_ids = set(int(value) for value in test_indices)

    overlap = calibration_ids.intersection(test_ids)

    if overlap:
        raise AssertionError(
            "Calibration rows overlap the untouched outer test fold. "
            f"Example overlapping IDs: {list(overlap)[:5]}"
        )

    return predictions, y_true


# ============================================================
# 5. EXACT ONE-SIDED CLOPPER-PEARSON UPPER BOUND
# ============================================================

def clopper_pearson_upper(errors, denominator, alpha):
    """
    One-sided exact Clopper-Pearson upper confidence bound for a binomial
    proportion: errors / denominator.

    Here:
      errors     = false-safe count (FP)
      denominator = all actual non-potable calibration rows (FP + TN)

    The denominator is not the number predicted potable.
    """
    errors = int(errors)
    denominator = int(denominator)

    if denominator <= 0:
        raise ValueError(
            "Clopper-Pearson denominator must be greater than zero."
        )

    if not 0 <= errors <= denominator:
        raise ValueError(
            "Error count must be between zero and the denominator."
        )

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")

    if errors == denominator:
        return 1.0

    return float(
        beta.ppf(
            1.0 - alpha,
            errors + 1,
            denominator - errors,
        )
    )


# ============================================================
# 6. METRICS FOR ONE THRESHOLD
# ============================================================

def threshold_metrics(y_true, probabilities, threshold):
    """
    Apply a binary threshold and calculate primary and supplementary metrics.

    Primary FSR:
        FP / all actual non-potable = FP / (FP + TN)

    Supplementary unsafe share among declared potable:
        FP / (TP + FP) = 1 - positive precision
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    predicted = (probabilities >= threshold).astype(int)

    fp = int(((y_true == 0) & (predicted == 1)).sum())
    tn = int(((y_true == 0) & (predicted == 0)).sum())
    tp = int(((y_true == 1) & (predicted == 1)).sum())
    fn = int(((y_true == 1) & (predicted == 0)).sum())

    n_actual_non_potable = int((y_true == 0).sum())
    n_actual_potable = int((y_true == 1).sum())
    n_total = int(len(y_true))

    n_declared_potable = tp + fp

    primary_false_safe_rate = (
        fp / n_actual_non_potable
        if n_actual_non_potable > 0
        else np.nan
    )

    unsafe_share_declared_potable = (
        fp / n_declared_potable
        if n_declared_potable > 0
        else np.nan
    )

    potable_recall = (
        tp / n_actual_potable
        if n_actual_potable > 0
        else np.nan
    )

    positive_precision = (
        tp / n_declared_potable
        if n_declared_potable > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "false_safe_count_fp": fp,
        "true_negative_tn": tn,
        "true_positive_tp": tp,
        "false_negative_fn": fn,
        "n_actual_non_potable": n_actual_non_potable,
        "n_actual_potable": n_actual_potable,
        "n_declared_potable": n_declared_potable,
        "n_total": n_total,
        "false_safe_rate": float(primary_false_safe_rate),
        "unsafe_share_among_declared_potable": float(
            unsafe_share_declared_potable
        ),
        "potable_recall": float(potable_recall),
        "positive_precision": float(positive_precision),
        "safe_prediction_share": (
            n_declared_potable / n_total if n_total else np.nan
        ),
        "decision_coverage": 1.0,
        "review_workload": 0.0,
        "accepted_error_rate": (
            (fp + fn) / n_total if n_total else np.nan
        ),
    }


# ============================================================
# 7. BUILD THE PREDECLARED CANDIDATE GRID
# ============================================================

def target_key(target):
    """Convert a target such as 0.05 to a stable column suffix such as 5pct."""
    return f"{int(round(target * 100))}pct"


def evaluate_candidate_grid(model_name, y_true, probabilities):
    """Evaluate all 202 predeclared thresholds for one model."""
    rows = []

    for index, threshold in enumerate(CANDIDATE_THRESHOLDS):
        metrics = threshold_metrics(
            y_true,
            probabilities,
            threshold,
        )

        cp_upper = clopper_pearson_upper(
            metrics["false_safe_count_fp"],
            metrics["n_actual_non_potable"],
            FAMILYWISE_ALPHA,
        )

        bonferroni_upper = clopper_pearson_upper(
            metrics["false_safe_count_fp"],
            metrics["n_actual_non_potable"],
            BONFERRONI_ALPHA,
        )

        is_reject_all = index == N_CANDIDATES - 1

        row = {
            "model": model_name,
            "candidate_index": index + 1,
            "candidate_type": (
                "REJECT_ALL" if is_reject_all else "ORDINARY_CUTOFF"
            ),
            **metrics,
            "clopper_pearson_upper_95": cp_upper,
            "bonferroni_clopper_pearson_upper": bonferroni_upper,
        }

        for target in SENSITIVITY_TARGETS:
            key = target_key(target)
            row[f"passes_empirical_{key}"] = (
                metrics["false_safe_rate"] <= target
            )
            row[f"passes_clopper_pearson_{key}"] = (
                cp_upper <= target
            )
            row[f"passes_bonferroni_{key}"] = (
                bonferroni_upper <= target
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 8. SELECT ONE THRESHOLD
# ============================================================

def select_threshold(grid, strategy, target):
    """
    Select the LOWEST candidate threshold that satisfies the risk rule.

    Lower cutoffs declare more samples potable and therefore generally
    increase coverage/recall. The cutoff is selected using calibration data
    only. This is a calibration-side selection, not a future-risk guarantee.
    """
    key = target_key(target)

    if strategy == "empirical":
        pass_column = f"passes_empirical_{key}"
    elif strategy == "clopper_pearson":
        pass_column = f"passes_clopper_pearson_{key}"
    elif strategy == "bonferroni":
        pass_column = f"passes_bonferroni_{key}"
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")

    eligible = grid.loc[grid[pass_column]].sort_values(
        "threshold",
        ascending=True,
        kind="stable",
    )

    if not eligible.empty:
        selected = eligible.iloc[0]

        status = (
            "REJECT_ALL_SELECTED"
            if selected["candidate_type"] == "REJECT_ALL"
            else "SELECTED"
        )

        return float(selected["threshold"]), True, status

    # If even the reject-all candidate cannot satisfy a confidence bound,
    # retain reject-all as the conservative operational fallback, but mark
    # the statistical target as not demonstrated on this calibration sample.
    return float(REJECT_ALL_THRESHOLD), False, "REJECT_ALL_FALLBACK"


# ============================================================
# 9. CREATE A RESULT ROW FOR A SELECTED RULE
# ============================================================

def make_result_row(
    fold,
    model_name,
    target,
    strategy,
    threshold,
    feasible,
    selection_status,
    y_true,
    probabilities,
):
    metrics = threshold_metrics(
        y_true,
        probabilities,
        threshold,
    )

    cp_upper = clopper_pearson_upper(
        metrics["false_safe_count_fp"],
        metrics["n_actual_non_potable"],
        FAMILYWISE_ALPHA,
    )

    bonferroni_upper = clopper_pearson_upper(
        metrics["false_safe_count_fp"],
        metrics["n_actual_non_potable"],
        BONFERRONI_ALPHA,
    )

    if strategy == "bonferroni":
        alpha_used = BONFERRONI_ALPHA
        selected_upper = bonferroni_upper
    elif strategy == "clopper_pearson":
        alpha_used = FAMILYWISE_ALPHA
        selected_upper = cp_upper
    else:
        alpha_used = np.nan
        selected_upper = np.nan

    return {
        "fold": fold,
        "target_false_safe_rate": target,
        "model": model_name,
        "strategy": strategy,
        "selection_status": selection_status,
        "feasible": feasible,
        "threshold": float(threshold),
        "alpha_used": alpha_used,
        **metrics,
        "clopper_pearson_upper_95": cp_upper,
        "bonferroni_clopper_pearson_upper": bonferroni_upper,
        "upper_bound_used_for_selection": selected_upper,
    }


# ============================================================
# 10. RUN ONE OUTER FOLD
# ============================================================

def run_fold(fold):
    print("\n" + "=" * 72)
    print(f"THRESHOLD SELECTION — OUTER FOLD {fold}")
    print("=" * 72)

    predictions, y_true = load_fold_data(fold)

    print("Calibration rows:", len(y_true))
    print("Actual non-potable (0):", int((y_true == 0).sum()))
    print("Actual potable (1):", int((y_true == 1).sum()))
    print("Probability source:", f"*{PROBABILITY_SUFFIX}")
    print("Primary false-safe denominator: all actual non-potable rows")
    print("Main risk target:", f"{TARGET_FALSE_SAFE:.0%}")
    print("Sensitivity targets:", SENSITIVITY_TARGETS)
    print("Candidate thresholds per model:", N_CANDIDATES)
    print("Bonferroni alpha per candidate:", f"{BONFERRONI_ALPHA:.8f}")

    all_grid_frames = []
    all_result_rows = []
    downstream_thresholds = {}

    for model_name in MODELS:
        probabilities = validate_probabilities(
            predictions[f"{model_name}{PROBABILITY_SUFFIX}"].to_numpy(),
            model_name,
        )

        grid = evaluate_candidate_grid(
            model_name,
            y_true,
            probabilities,
        )
        all_grid_frames.append(grid)

        downstream_thresholds[model_name] = {}

        # Fixed threshold is a comparison reference and is not risk-selected.
        fixed_metrics = threshold_metrics(
            y_true,
            probabilities,
            FIXED_THRESHOLD,
        )

        fixed_row_base = {
            "fold": fold,
            "model": model_name,
            "strategy": "fixed_0.50",
            "selection_status": "PREDEFINED_REFERENCE",
            "feasible": np.nan,
            "threshold": FIXED_THRESHOLD,
            "alpha_used": np.nan,
            **fixed_metrics,
            "clopper_pearson_upper_95": clopper_pearson_upper(
                fixed_metrics["false_safe_count_fp"],
                fixed_metrics["n_actual_non_potable"],
                FAMILYWISE_ALPHA,
            ),
            "bonferroni_clopper_pearson_upper": clopper_pearson_upper(
                fixed_metrics["false_safe_count_fp"],
                fixed_metrics["n_actual_non_potable"],
                BONFERRONI_ALPHA,
            ),
            "upper_bound_used_for_selection": np.nan,
        }

        for target in SENSITIVITY_TARGETS:
            fixed_row = dict(fixed_row_base)
            fixed_row["target_false_safe_rate"] = target
            all_result_rows.append(fixed_row)

        # The majority baseline is a constant-probability reference model.
        # Its threshold cannot meaningfully trade off recall and false-safe
        # risk: it predicts all samples as one class or all as the other.
        # Therefore, do NOT present a threshold found on its flat score
        # distribution as a useful risk-controlled selection.
        if model_name == "majority_baseline":
            for target in SENSITIVITY_TARGETS:
                for strategy in [
                    "empirical",
                    "clopper_pearson",
                    "bonferroni",
                ]:
                    result_row = make_result_row(
                        fold=fold,
                        model_name=model_name,
                        target=target,
                        strategy=strategy,
                        threshold=FIXED_THRESHOLD,
                        feasible=np.nan,
                        selection_status="REFERENCE_ONLY_CONSTANT_BASELINE",
                        y_true=y_true,
                        probabilities=probabilities,
                    )

                    # These bounds are descriptive only; they were not used
                    # to select a threshold for the constant baseline.
                    result_row["alpha_used"] = np.nan
                    result_row["upper_bound_used_for_selection"] = np.nan
                    all_result_rows.append(result_row)

                    if target == TARGET_FALSE_SAFE:
                        # Keep numeric keys expected by the downstream
                        # evaluation script. All baseline entries intentionally
                        # point to the fixed reference threshold.
                        downstream_thresholds[model_name][strategy] = float(
                            FIXED_THRESHOLD
                        )

        else:
            # Select risk-controlled thresholds for predictive models only.
            for target in SENSITIVITY_TARGETS:
                for strategy in [
                    "empirical",
                    "clopper_pearson",
                    "bonferroni",
                ]:
                    threshold, feasible, status = select_threshold(
                        grid,
                        strategy,
                        target,
                    )

                    result_row = make_result_row(
                        fold=fold,
                        model_name=model_name,
                        target=target,
                        strategy=strategy,
                        threshold=threshold,
                        feasible=feasible,
                        selection_status=status,
                        y_true=y_true,
                        probabilities=probabilities,
                    )
                    all_result_rows.append(result_row)

                    # The downstream model-evaluation script uses the main
                    # 5% target only. Keep its expected nested numeric structure.
                    if target == TARGET_FALSE_SAFE:
                        downstream_thresholds[model_name][strategy] = float(
                            threshold
                        )

        downstream_thresholds[model_name]["fixed_0.50"] = float(
            FIXED_THRESHOLD
        )

    grid_df = pd.concat(all_grid_frames, ignore_index=True)
    results_df = pd.DataFrame(all_result_rows)

    # --------------------------------------------------------
    # 10.1 Save the full candidate grid
    # --------------------------------------------------------

    grid_path = TABLE_DIR / f"threshold_grid_fold_{fold}.csv"
    grid_df.to_csv(grid_path, index=False)

    # --------------------------------------------------------
    # 10.2 Save selected strategies and sensitivity analysis
    # --------------------------------------------------------

    result_path = TABLE_DIR / f"threshold_selection_fold_{fold}.csv"
    results_df.to_csv(result_path, index=False)

    # --------------------------------------------------------
    # 10.3 Save downstream-compatible thresholds.json
    # --------------------------------------------------------

    fold_dir = EXPERIMENT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    threshold_path = fold_dir / "thresholds.json"

    threshold_manifest = {
        "fold": fold,
        "probability_columns": f"*{PROBABILITY_SUFFIX}",
        "positive_class": 1,
        "positive_class_name": "potable",
        "negative_class": 0,
        "negative_class_name": "non_potable",
        "false_safe_definition": "actual 0 predicted 1",
        "primary_false_safe_metric": "FP / (FP + TN)",
        "supplementary_metric": (
            "FP / (TP + FP), reported as unsafe share among declared-potable "
            "predictions; not used for threshold selection"
        ),
        "target_false_safe_rate": TARGET_FALSE_SAFE,
        "sensitivity_targets": SENSITIVITY_TARGETS,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "one_sided_confidence_level": 1.0 - FAMILYWISE_ALPHA,
        "candidate_count_per_model": N_CANDIDATES,
        "ordinary_thresholds": (
            "0.000 to 1.000 in increments of 0.005"
        ),
        "reject_all_threshold": float(REJECT_ALL_THRESHOLD),
        "bonferroni_alpha_per_candidate": BONFERRONI_ALPHA,
        "bonferroni_scope": (
            "202 candidate thresholds within each model family; "
            "not a simultaneous correction across all four models"
        ),
        "majority_baseline_handling": (
            "Reference-only constant-probability model. No risk-based "
            "threshold optimization is performed. Numeric 0.50 values are "
            "retained for downstream schema compatibility and must not be "
            "interpreted as selected risk-controlled thresholds."
        ),
        "outer_test_used_for_selection": False,
        "thresholds": downstream_thresholds,
    }

    with open(threshold_path, "w", encoding="utf-8") as file:
        json.dump(
            threshold_manifest,
            file,
            indent=4,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # 10.4 Print the main 5% results
    # --------------------------------------------------------

    main_results = results_df[
        results_df["target_false_safe_rate"] == TARGET_FALSE_SAFE
    ].copy()

    display_columns = [
        "model",
        "strategy",
        "selection_status",
        "threshold",
        "false_safe_count_fp",
        "n_actual_non_potable",
        "false_safe_rate",
        "unsafe_share_among_declared_potable",
        "potable_recall",
        "safe_prediction_share",
        "clopper_pearson_upper_95",
        "bonferroni_clopper_pearson_upper",
    ]

    print("\n===== MAIN TARGET RESULTS (5%) =====")
    print(
        main_results[display_columns]
        .round(5)
        .to_string(index=False)
    )

    print("\nSaved full candidate grid:", grid_path)
    print("Saved selected-results table:", result_path)
    print("Saved downstream thresholds:", threshold_path)

    return results_df


# ============================================================
# 11. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Thesis-consistent water potability threshold selection."
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Run one outer fold (1-5). Default: 1.",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Run all five outer folds.",
    )

    args = parser.parse_args()

    if args.all:
        folds = list(range(1, N_OUTER_SPLITS + 1))
    else:
        if not 1 <= args.fold <= N_OUTER_SPLITS:
            raise ValueError("Fold number must be between 1 and 5.")
        folds = [args.fold]

    all_results = pd.concat(
        [run_fold(fold) for fold in folds],
        ignore_index=True,
    )

    if args.all:
        summary = (
            all_results
            .groupby(
                ["target_false_safe_rate", "model", "strategy"],
                sort=False,
                dropna=False,
            )
            .agg(
                threshold_mean=("threshold", "mean"),
                feasible_folds=("feasible", "sum"),
                safe_prediction_share_mean=("safe_prediction_share", "mean"),
                false_safe_rate_mean=("false_safe_rate", "mean"),
                unsafe_share_declared_potable_mean=(
                    "unsafe_share_among_declared_potable", "mean"
                ),
                potable_recall_mean=("potable_recall", "mean"),
            )
            .round(4)
            .reset_index()
        )

        summary_path = TABLE_DIR / "threshold_selection_summary.csv"
        summary.to_csv(summary_path, index=False)

        print("\n===== ALL-FOLD SUMMARY — MAIN 5% TARGET =====")
        print(
            summary[
                summary["target_false_safe_rate"] == TARGET_FALSE_SAFE
            ].to_string(index=False)
        )
        print("Saved all-fold summary:", summary_path)

    print("\n" + "=" * 72)
    print("THRESHOLD SELECTION COMPLETED")
    print("=" * 72)
    print("Only *_sigmoid_oof probabilities were used.")
    print("Majority baseline was retained as a reference-only model.")
    print("Outer test folds were not loaded or used.")
    print("Main threshold target:", f"{TARGET_FALSE_SAFE:.0%}")
    print(
        "The calibration-side target is not a guarantee "
        "of future outer-test risk."
    )
    print("Thresholds are saved in the format expected by 09_model_evaluation.py.")


if __name__ == "__main__":
    main()
