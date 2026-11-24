"""
LESSON 08: SELECTIVE PREDICTION WITH ABSTENTION

Purpose
-------
Choose confidence cut-offs on each fold's calibration data so that the
system can either make a decision or abstain with "Needs Review".

Research design
---------------
1. Input:
   04_Results/tables/calibrated_predictions_fold_k.csv

2. Only out-of-fold sigmoid probabilities are used:
   {model}_sigmoid_oof

   Do NOT use {model}_sigmoid_full_fit or any outer-test probability.
   OOF probabilities prevent each calibration row from being scored by
   a sigmoid calibrator fitted on that same row.

3. Confidence:
       confidence = abs(sigmoid_oof_probability - 0.50)

4. Target decision coverages:
       25%, 50%, 75%, 100%

   For each target, retain at least the requested proportion of rows
   with the highest confidence. Tied confidence values are all retained,
   so actual calibration coverage can be slightly higher than target.

   For 100% coverage, the saved cut-off is exactly 0.0 so every future
   test row is decided. Actual outer-test coverage at 25%, 50% and 75%
   is not guaranteed to equal the calibration target.

5. Decision rule among retained rows:
       probability >= 0.50 -> Safe (predicted potable)
       probability <  0.50 -> Unsafe (predicted non-potable)
       otherwise           -> Needs Review

6. The majority baseline is reference-only. Its constant score cannot
   meaningfully rank uncertainty, so no selective cut-offs are created
   for it.

7. Outer-test rows are checked against split_manifest.json and are never
   loaded for cut-off selection or used in this script.

8. Calibration-side performance metrics are descriptive diagnostics only.
   Final performance must be reported from the untouched outer-test folds
   in 09_model_evaluation.py.

Outputs
-------
03_Experiments/fold_k/selective_rules.json
03_Experiments/fold_k/selective_manifest.json (input + output SHA-256 fingerprints)

04_Results/tables/selective_prediction_fold_k.csv
04_Results/tables/selective_calibration_predictions_fold_k.csv

With --all:
04_Results/tables/selective_prediction_summary_all_folds.csv

Usage
-----
python 02_Code/08_selective_prediction.py
python 02_Code/08_selective_prediction.py --fold 3
python 02_Code/08_selective_prediction.py --all
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 2. PROJECT PATHS AND FIXED SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "01_Data" / "raw" / "water_potability.csv"
EXPERIMENT_DIR = PROJECT_DIR / "03_Experiments"
TABLE_DIR = PROJECT_DIR / "04_Results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Potability"
N_OUTER_SPLITS = 5

DECISION_THRESHOLD = 0.50
TARGET_COVERAGES = [0.25, 0.50, 0.75, 1.00]
CONFIDENCE_TOLERANCE = 1e-9

MODELS = [
    "majority_baseline",
    "logistic_regression",
    "random_forest",
    "mlp",
]

REFERENCE_ONLY_MODELS = {"majority_baseline"}


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def require_file(path, hint=""):
    """Stop with a clear message if a required file is missing."""
    path = Path(path)

    if not path.is_file():
        message = f"Required file not found:\n{path}"

        if hint:
            message += f"\n{hint}"

        raise FileNotFoundError(message)

    return path


def sha256_file(path):
    """Return the SHA-256 fingerprint of a file."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def safe_divide(numerator, denominator):
    """Return NaN when the denominator is zero."""
    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def get_manifest_indices(manifest, key, description):
    """Read a strictly integer, unique index list from the fold manifest."""
    if key not in manifest:
        raise KeyError(
            f"'{key}' is missing from split_manifest.json "
            f"({description})."
        )

    raw_indices = np.asarray(manifest[key], dtype=float)

    if raw_indices.ndim != 1 or len(raw_indices) == 0:
        raise ValueError(f"{description} must be a non-empty 1-D list.")

    if not np.isfinite(raw_indices).all():
        raise ValueError(f"{description} contains non-finite indices.")

    if not np.equal(raw_indices, np.floor(raw_indices)).all():
        raise ValueError(f"{description} contains non-integer indices.")

    indices = raw_indices.astype(int)

    if len(indices) != len(np.unique(indices)):
        raise AssertionError(f"Duplicate indices found in {description}.")

    return indices


def validate_probabilities(probabilities, label):
    """Check that probabilities are finite and lie within [0, 1]."""
    values = np.asarray(probabilities, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional array.")

    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinite values.")

    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"{label} contains values outside [0, 1].")

    return values


# ============================================================
# 4. LOAD AND VALIDATE THE RAW DATASET
# ============================================================

def load_raw_labels():
    """Load labels only, to verify calibration-row provenance."""
    require_file(
        DATA_PATH,
        "Place water_potability.csv inside 01_Data/raw/.",
    )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' was not found.")

    y = pd.to_numeric(df[TARGET], errors="raise")

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Potability must contain only labels 0 and 1.")

    return y.astype(int).to_numpy()


# ============================================================
# 5. VALIDATE FOLD SPLIT AND CALIBRATION PREDICTIONS
# ============================================================

def load_fold_calibration_data(fold, raw_y):
    """
    Load this fold's OOF sigmoid probabilities and verify that the
    rows are exactly the calibration rows in split_manifest.json.
    """
    fold_dir = EXPERIMENT_DIR / f"fold_{fold}"

    manifest_path = require_file(
        fold_dir / "split_manifest.json",
        "Run 05_model_training.py for this fold first.",
    )

    predictions_path = require_file(
        TABLE_DIR / f"calibrated_predictions_fold_{fold}.csv",
        "Run 06_probability_calibration.py for this fold first.",
    )

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    saved_fold = manifest.get("fold")

    if saved_fold is not None and int(saved_fold) != fold:
        raise AssertionError(
            f"Split manifest belongs to fold {saved_fold}, not fold {fold}."
        )

    calibration_indices = get_manifest_indices(
        manifest,
        "calibration_indices",
        "calibration indices",
    )

    test_indices = get_manifest_indices(
        manifest,
        "test_indices",
        "outer-test indices",
    )

    train_indices = get_manifest_indices(
        manifest,
        "train_indices",
        "training indices",
    )

    train_set = set(train_indices.tolist())
    calibration_set = set(calibration_indices.tolist())
    test_set = set(test_indices.tolist())

    if train_set & calibration_set:
        raise AssertionError("Training and calibration rows overlap.")

    if train_set & test_set:
        raise AssertionError("Training and outer-test rows overlap.")

    if calibration_set & test_set:
        raise AssertionError("Calibration and outer-test rows overlap.")

    all_split_indices = train_set | calibration_set | test_set

    if max(all_split_indices) >= len(raw_y) or min(all_split_indices) < 0:
        raise IndexError("Split manifest contains indices outside the dataset.")

    if all_split_indices != set(range(len(raw_y))):
        missing = sorted(set(range(len(raw_y))) - all_split_indices)
        extra = sorted(all_split_indices - set(range(len(raw_y))))
        raise AssertionError(
            "Train/calibration/test splits do not partition the raw "
            "dataset exactly once. "
            f"Missing (first 10): {missing[:10]}; "
            f"unexpected (first 10): {extra[:10]}"
        )

    predictions = pd.read_csv(predictions_path)

    required_columns = {"row_index", "y_true"}

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise KeyError(
            "Calibration predictions are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if predictions["row_index"].isna().any():
        raise ValueError("row_index contains missing values.")

    if predictions["row_index"].duplicated().any():
        raise AssertionError(
            "Duplicate row_index values found in calibration predictions."
        )

    raw_row_indices = pd.to_numeric(
        predictions["row_index"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(raw_row_indices).all():
        raise ValueError("Calibration row_index contains non-finite values.")

    if not np.equal(raw_row_indices, np.floor(raw_row_indices)).all():
        raise ValueError("Calibration row_index must contain integers.")

    row_indices = raw_row_indices.astype(int)

    if ((row_indices < 0) | (row_indices >= len(raw_y))).any():
        raise IndexError("Calibration prediction row_index is out of range.")

    observed_calibration_set = set(row_indices.tolist())

    if observed_calibration_set != calibration_set:
        missing = sorted(calibration_set - observed_calibration_set)
        extra = sorted(observed_calibration_set - calibration_set)

        raise AssertionError(
            "Calibration predictions do not match the fold's "
            "calibration_indices.\n"
            f"Missing row IDs (first 10): {missing[:10]}\n"
            f"Unexpected row IDs (first 10): {extra[:10]}"
        )

    if observed_calibration_set & test_set:
        raise AssertionError(
            "Outer-test rows were found in calibration predictions."
        )

    raw_y_true = pd.to_numeric(
        predictions["y_true"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(raw_y_true).all():
        raise ValueError("Calibration y_true contains non-finite values.")

    if not np.equal(raw_y_true, np.floor(raw_y_true)).all():
        raise ValueError("Calibration y_true must contain integer labels.")

    y_true = raw_y_true.astype(int)

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Calibration labels must contain only 0 and 1.")

    expected_y = raw_y[row_indices]

    if not np.array_equal(y_true, expected_y):
        raise AssertionError(
            "y_true in calibration predictions does not match "
            "the raw dataset labels at row_index."
        )

    required_oof_columns = [
        f"{model_name}_sigmoid_oof"
        for model_name in MODELS
        if model_name not in REFERENCE_ONLY_MODELS
    ]
    missing_oof_columns = [
        column for column in required_oof_columns
        if column not in predictions.columns
    ]

    if missing_oof_columns:
        raise KeyError(
            "The calibrated predictions file does not contain the required "
            "out-of-fold sigmoid columns: "
            f"{missing_oof_columns}. "
            "Re-run the corrected 06_probability_calibration.py. "
            "Do NOT substitute *_sigmoid full-fit columns: those calibrators "
            "were fitted on the same calibration rows."
        )

    if predictions[required_oof_columns].isna().any().any():
        raise ValueError(
            "One or more required OOF probability columns contain NaN."
        )

    print(f"\n===== OUTER FOLD {fold}: DATA VALIDATION =====")
    print("Calibration rows:", len(predictions))
    print("Training/calibration/test separation: PASSED")
    print("Calibration row IDs match split manifest: PASSED")
    print("Calibration labels match raw dataset: PASSED")
    print("Outer-test rows used: 0")

    return predictions, row_indices, y_true, manifest_path, predictions_path


# ============================================================
# 6. SELECT A CONFIDENCE CUTOFF FOR A TARGET COVERAGE
# ============================================================

def select_confidence_cutoff(confidence, target_coverage):
    """
    Retain at least ceil(target_coverage * n) highest-confidence rows.

    If several rows share the cutoff confidence, all tied rows are
    retained. Therefore actual coverage may exceed target coverage.
    """
    confidence = np.asarray(confidence, dtype=float)

    if len(confidence) == 0:
        raise ValueError("Cannot select a cutoff from an empty array.")

    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("Target coverage must be in (0, 1].")

    # 100% means every future row must receive a decision. A cutoff equal
    # to the calibration minimum is not sufficient: a future row may have
    # even lower confidence. Zero is the correct universal cutoff because
    # confidence = abs(p - 0.5) is always non-negative.
    if target_coverage == 1.0:
        return 0.0, np.ones(len(confidence), dtype=bool)

    n_rows = len(confidence)
    n_to_retain = max(1, int(math.ceil(target_coverage * n_rows)))

    sorted_confidence = np.sort(confidence)

    cutoff_index = n_rows - n_to_retain
    cutoff = float(sorted_confidence[cutoff_index])

    decided = (
        confidence >= cutoff - CONFIDENCE_TOLERANCE
    )

    return cutoff, decided


# ============================================================
# 7. CALCULATE SELECTIVE-PREDICTION METRICS
# ============================================================

def calculate_selective_metrics(y_true, probabilities, decided):
    """
    Calculate selective classification metrics.

    Potable = 1
    Non-potable = 0

    Primary false-safe denominator:
        all actual non-potable rows, including rows sent to review.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    decided = np.asarray(decided, dtype=bool)

    if not (
        len(y_true) == len(probabilities) == len(decided)
    ):
        raise ValueError("Metric input arrays have different lengths.")

    safe = decided & (probabilities >= DECISION_THRESHOLD)
    unsafe = decided & (probabilities < DECISION_THRESHOLD)
    needs_review = ~decided

    actual_non_potable = y_true == 0
    actual_potable = y_true == 1

    false_safe = safe & actual_non_potable
    false_unsafe = unsafe & actual_potable

    true_safe = safe & actual_potable
    true_unsafe = unsafe & actual_non_potable

    n_total = len(y_true)
    n_decided = int(decided.sum())

    n_safe = int(safe.sum())
    n_unsafe = int(unsafe.sum())
    n_review = int(needs_review.sum())

    n_actual_non_potable = int(actual_non_potable.sum())
    n_actual_potable = int(actual_potable.sum())

    n_false_safe = int(false_safe.sum())
    n_false_unsafe = int(false_unsafe.sum())

    n_correct_decisions = int(true_safe.sum() + true_unsafe.sum())
    n_accepted_errors = n_false_safe + n_false_unsafe

    decided_non_potable = int(
        (decided & actual_non_potable).sum()
    )

    return {
        "n_total": n_total,
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
        "n_needs_review": n_review,
        "n_decided": n_decided,

        "actual_coverage": safe_divide(n_decided, n_total),
        "review_share": safe_divide(n_review, n_total),

        "accepted_errors": n_accepted_errors,
        "accepted_error_rate": safe_divide(
            n_accepted_errors,
            n_decided,
        ),
        "selective_accuracy": safe_divide(
            n_correct_decisions,
            n_decided,
        ),

        "false_safe_count": n_false_safe,
        "n_actual_non_potable": n_actual_non_potable,

        # Main FSR denominator includes ALL actual non-potable rows.
        "false_safe_rate_all_non_potable": safe_divide(
            n_false_safe,
            n_actual_non_potable,
        ),

        # Supplementary metric among only the decided non-potable rows.
        "conditional_false_safe_rate_decided": safe_divide(
            n_false_safe,
            decided_non_potable,
        ),

        # This is the same denominator used by Lesson 07's TARGET_FSR:
        # FP / (TP + FP), i.e. unsafe share among all declared-safe rows.
        "unsafe_share_among_declared_safe": safe_divide(
            n_false_safe,
            n_safe,
        ),

        # Recall over the full population: abstained potable rows remain
        # in the denominator and are not counted as correctly identified.
        "potable_recall_all": safe_divide(
            int(true_safe.sum()),
            n_actual_potable,
        ),

        # Supplementary recall among decided records only.
        "potable_recall_among_decided": safe_divide(
            int(true_safe.sum()),
            int((decided & actual_potable).sum()),
        ),
    }


# ============================================================
# 8. RUN SELECTIVE PREDICTION FOR ONE FOLD
# ============================================================

def run_fold(fold, raw_y):
    print("\n" + "=" * 72)
    print(f"SELECTIVE PREDICTION — OUTER FOLD {fold}")
    print("=" * 72)

    (
        predictions,
        row_indices,
        y_true,
        manifest_path,
        predictions_path,
    ) = load_fold_calibration_data(fold, raw_y)

    input_hashes_before = {
        "raw_dataset": sha256_file(DATA_PATH),
        "split_manifest": sha256_file(manifest_path),
        "calibrated_predictions": sha256_file(predictions_path),
    }

    fold_dir = EXPERIMENT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    cutoff_map = {}
    summary_rows = []
    row_prediction_frames = []

    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")

        # The majority baseline is retained only as a reference.
        # A constant score cannot rank confidence meaningfully.
        if model_name in REFERENCE_ONLY_MODELS:
            cutoff_map[model_name] = {}

            for target_coverage in TARGET_COVERAGES:
                summary_rows.append({
                    "fold": fold,
                    "model": model_name,
                    "selection_status": "REFERENCE_ONLY",
                    "target_coverage": target_coverage,
                    "confidence_cutoff": np.nan,
                    "probability_source": "not_applicable",
                    "tie_policy": "not_applicable",
                })

            print("Reference-only model: no selective cut-offs created.")
            continue

        probability_column = f"{model_name}_sigmoid_oof"

        if probability_column not in predictions.columns:
            raise KeyError(
                f"Required OOF probability column is missing:\n"
                f"{probability_column}\n\n"
                "Run the current 06_probability_calibration.py first. "
                "This script intentionally does not fall back to "
                "sigmoid_full_fit or in-sample sigmoid probabilities."
            )

        probabilities = validate_probabilities(
            predictions[probability_column].to_numpy(),
            probability_column,
        )

        confidence = np.abs(
            probabilities - DECISION_THRESHOLD
        )

        model_cutoffs = {}

        for target_coverage in TARGET_COVERAGES:
            cutoff, decided = select_confidence_cutoff(
                confidence,
                target_coverage,
            )

            metrics = calculate_selective_metrics(
                y_true,
                probabilities,
                decided,
            )

            coverage_key = str(target_coverage)
            model_cutoffs[coverage_key] = cutoff

            summary_rows.append({
                "fold": fold,
                "model": model_name,
                "selection_status": "SELECTED_ON_CALIBRATION_OOF",
                "target_coverage": target_coverage,
                "confidence_cutoff": cutoff,
                "probability_source": probability_column,
                "tie_policy": "include_all_rows_at_cutoff",
                **metrics,
            })

            # Long-format calibration predictions provide an audit trail.
            safe = decided & (probabilities >= DECISION_THRESHOLD)
            unsafe = decided & (probabilities < DECISION_THRESHOLD)

            decisions = np.full(
                len(probabilities),
                "Needs Review",
                dtype=object,
            )
            decisions[unsafe] = "Unsafe"
            decisions[safe] = "Safe"

            row_prediction_frames.append(
                pd.DataFrame({
                    "fold": fold,
                    "row_index": row_indices,
                    "y_true": y_true,
                    "model": model_name,
                    "target_coverage": target_coverage,
                    "calibrated_probability": probabilities,
                    "confidence": confidence,
                    "confidence_cutoff": cutoff,
                    "decision": decisions,
                })
            )

            print(
                f"Target coverage: {target_coverage:.0%} | "
                f"Cutoff: {cutoff:.6f} | "
                f"Actual coverage: "
                f"{metrics['actual_coverage']:.2%} | "
                f"Needs Review: {metrics['n_needs_review']}"
            )

        cutoff_map[model_name] = model_cutoffs

    # --------------------------------------------------------
    # 8.1 Save selective rules for Lesson 09
    # --------------------------------------------------------

    selective_rules = {
        "fold": fold,
        "method": "confidence_based_abstention",
        "decision_threshold": DECISION_THRESHOLD,
        "confidence_definition": (
            "absolute_difference_between_sigmoid_oof_probability "
            "and decision_threshold"
        ),
        "probability_method": "sigmoid",
        "probability_source": "out_of_fold_calibration_predictions",
        "target_coverages": TARGET_COVERAGES,
        "cutoffs": cutoff_map,
        "reference_only_models": sorted(REFERENCE_ONLY_MODELS),
        "tie_policy": (
            "All rows with confidence >= cutoff - 1e-9 are retained. "
            "Ties can make actual coverage higher than target coverage."
        ),
        "outer_test_used_for_selection": False,
        "calibration_metrics_are_final_test_results": False,
        "coverage_note": (
            "Target coverage is selected from calibration OOF confidence "
            "quantiles. Actual outer-test coverage may differ."
        ),
        "calibration_prediction_file": (
            f"calibrated_predictions_fold_{fold}.csv"
        ),
    }

    rules_path = fold_dir / "selective_rules.json"

    with open(rules_path, "w", encoding="utf-8") as file:
        json.dump(selective_rules, file, indent=4)

    # --------------------------------------------------------
    # 8.2 Save summary table
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        TABLE_DIR / f"selective_prediction_fold_{fold}.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    # --------------------------------------------------------
    # 8.3 Save row-level calibration decisions
    # --------------------------------------------------------

    if row_prediction_frames:
        row_predictions_df = pd.concat(
            row_prediction_frames,
            ignore_index=True,
        )
    else:
        row_predictions_df = pd.DataFrame()

    row_predictions_path = (
        TABLE_DIR
        / f"selective_calibration_predictions_fold_{fold}.csv"
    )

    row_predictions_df.to_csv(row_predictions_path, index=False)

    # --------------------------------------------------------
    # 8.4 Verify that input files were not changed
    # --------------------------------------------------------

    input_hashes_after = {
        "raw_dataset": sha256_file(DATA_PATH),
        "split_manifest": sha256_file(manifest_path),
        "calibrated_predictions": sha256_file(predictions_path),
    }

    if input_hashes_before != input_hashes_after:
        raise AssertionError(
            "An input file changed during selective prediction."
        )

    # --------------------------------------------------------
    # 8.5 Save an audit manifest
    # --------------------------------------------------------

    # Fingerprint the generated outputs as well as the input files.
    # The manifest is stored separately so the rules file can be checked
    # without creating a self-referential hash inside selective_rules.json.
    output_paths = {
        "selective_rules": rules_path,
        "summary_table": summary_path,
        "row_level_predictions": row_predictions_path,
    }
    output_hashes = {
        name: sha256_file(path)
        for name, path in output_paths.items()
    }

    audit_manifest = {
        "audit_schema_version": 2,
        "fold": fold,
        "outer_test_used_for_selection": False,
        "probability_source": "sigmoid_oof",
        "decision_threshold": DECISION_THRESHOLD,
        "calibration_metrics_are_final_test_results": False,
        "target_coverages": TARGET_COVERAGES,
        "input_sha256": input_hashes_after,
        "output_sha256": output_hashes,
        "outputs": {
            name: str(path.relative_to(PROJECT_DIR))
            for name, path in output_paths.items()
        },
    }

    audit_path = fold_dir / "selective_manifest.json"

    with open(audit_path, "w", encoding="utf-8") as file:
        json.dump(audit_manifest, file, indent=4)

    # --------------------------------------------------------
    # 8.6 Print results
    # --------------------------------------------------------

    display_columns = [
        "model",
        "selection_status",
        "target_coverage",
        "confidence_cutoff",
        "actual_coverage",
        "review_share",
        "accepted_error_rate",
        "false_safe_rate_all_non_potable",
        "potable_recall_all",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in summary_df.columns
    ]

    print("\n===== SELECTIVE PREDICTION RESULTS =====")
    print(
        summary_df[available_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved selective rules:", rules_path)
    print("Saved summary:", summary_path)
    print("Saved row-level decisions:", row_predictions_path)
    print("Saved audit manifest:", audit_path)

    print("\nInput SHA-256 verification: PASSED")
    print("Output SHA-256 fingerprints saved for rules and result tables.")
    print("Outer-test rows used: 0")

    return summary_df


# ============================================================
# 9. ALL-FOLD SUMMARY
# ============================================================

def save_all_fold_summary(all_results):
    """Save mean and standard deviation across the five folds."""
    evaluated = all_results[
        all_results["selection_status"]
        == "SELECTED_ON_CALIBRATION_OOF"
    ].copy()

    metric_columns = [
        "confidence_cutoff",
        "actual_coverage",
        "review_share",
        "accepted_error_rate",
        "selective_accuracy",
        "false_safe_rate_all_non_potable",
        "conditional_false_safe_rate_decided",
        "unsafe_share_among_declared_safe",
        "potable_recall_all",
        "potable_recall_among_decided",
    ]

    available_metrics = [
        column
        for column in metric_columns
        if column in evaluated.columns
    ]

    summary = (
        evaluated
        .groupby(["model", "target_coverage"])[available_metrics]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]

    summary = summary.reset_index()

    output_path = (
        TABLE_DIR / "selective_prediction_summary_all_folds.csv"
    )

    summary.to_csv(output_path, index=False)

    print("\n===== ALL-FOLD SELECTIVE PREDICTION SUMMARY =====")
    print(summary.round(4).to_string(index=False))
    print("\nSaved all-fold summary:", output_path)


# ============================================================
# 10. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select confidence-based abstention cut-offs using "
            "calibration OOF probabilities only."
        )
    )

    fold_group = parser.add_mutually_exclusive_group()

    fold_group.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Run one outer fold (1-5). Default: 1.",
    )

    fold_group.add_argument(
        "--all",
        action="store_true",
        help="Run all five outer folds.",
    )

    args = parser.parse_args()

    if args.all:
        folds = list(range(1, N_OUTER_SPLITS + 1))
    else:
        if not 1 <= args.fold <= N_OUTER_SPLITS:
            parser.error("Fold number must be between 1 and 5.")

        folds = [args.fold]

    raw_y = load_raw_labels()

    print("\n" + "=" * 72)
    print("SELECTIVE PREDICTION — CONFIDENCE-BASED ABSTENTION")
    print("=" * 72)
    print("Positive class: Potable = 1")
    print("Negative class: Non-potable = 0")
    print("Decision threshold:", DECISION_THRESHOLD)
    print("Target coverages:", TARGET_COVERAGES)
    print("Probability source: sigmoid OOF only")
    print("Majority baseline: reference-only")
    print("Outer-test data: NOT USED")

    fold_results = []

    for fold in folds:
        fold_results.append(run_fold(fold, raw_y))

    all_results = pd.concat(
        fold_results,
        ignore_index=True,
    )

    if args.all:
        save_all_fold_summary(all_results)

    print("\n" + "=" * 72)
    print("SELECTIVE PREDICTION COMPLETED")
    print("=" * 72)
    print("Folds completed:", folds)
    print("No model was retrained.")
    print("No outer-test data was used.")
    print("No threshold from Lesson 07 was modified.")
    print("Selective cut-offs were selected from sigmoid OOF probabilities.")
    print("Rules are saved for 09_model_evaluation.py.")


if __name__ == "__main__":
    main()
