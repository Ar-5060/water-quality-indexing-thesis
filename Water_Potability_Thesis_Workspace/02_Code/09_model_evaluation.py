"""
FINAL MODEL EVALUATION

Purpose
-------
Evaluate the saved models on the outer-test rows that were not used for
training, calibration, or threshold selection.

Research safeguards
-------------------
1. This script does NOT train or modify any model.
2. It does NOT select or modify any threshold.
3. Each fold uses only its own model, calibrator, split manifest and thresholds.
4. Thresholds/cut-offs were selected from sigmoid OOF calibration
   probabilities. Test probabilities are produced by each fold's saved
   sigmoid calibrator fitted on the full calibration subset.
5. The primary False-Safe Rate (FSR) is:
       FP / (FP + TN)
   where FP means actual non-potable water predicted as potable.
6. SHA-256 fingerprints verify that the input artifacts remain unchanged.
7. Selective prediction uses the saved Lesson 08 rules without re-selection.
8. Lesson 08 input hashes AND output hashes (including selective_rules.json)
   are checked when selective rules are present.
9. With --all, the five outer-test folds must cover every dataset row exactly once.

Positive class: Potable = 1
Negative class: Non-potable = 0
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)

# Saved sklearn pipelines may refer to this function when joblib loads them.
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from water_utils import fix_invalid_ph  # noqa: F401


# ============================================================
# 2. PROJECT PATHS AND FIXED SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "01_Data" / "raw" / "water_potability.csv"
EXPERIMENT_DIR = PROJECT_DIR / "03_Experiments"
TABLE_DIR = PROJECT_DIR / "04_Results" / "tables"
FIGURE_DIR = PROJECT_DIR / "04_Results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Potability"
N_OUTER_SPLITS = 5
EPS = 1e-6
N_BINS = 10

MODELS = [
    "majority_baseline",
    "logistic_regression",
    "random_forest",
    "mlp",
]

THRESHOLD_STRATEGIES = [
    "fixed_0.50",
    "empirical",
    "clopper_pearson",
    "bonferroni",
]

PROBABILITY_METHODS = [
    "uncalibrated",
    "sigmoid",
    "isotonic",
]


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
    """Return NaN rather than dividing by zero."""
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def to_logit(probabilities):
    """
    Convert raw probabilities to log-odds.

    This transformation must match 06_probability_calibration.py.
    """
    p = np.clip(
        np.asarray(probabilities, dtype=float),
        EPS,
        1.0 - EPS,
    )
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def validate_probabilities(probabilities, label):
    """Check that a probability array is finite and lies in [0, 1]."""
    p = np.asarray(probabilities, dtype=float)

    if not np.isfinite(p).all():
        raise ValueError(f"{label}: probabilities contain NaN or infinity.")

    if ((p < 0) | (p > 1)).any():
        raise ValueError(f"{label}: probabilities fall outside [0, 1].")

    return p


# ============================================================
# 4. METRIC FUNCTIONS
# ============================================================

def expected_calibration_error(y_true, probabilities, n_bins=N_BINS):
    """
    Expected Calibration Error using equal-width probability bins.

    ECE is descriptive and depends on the number and type of bins.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(
        np.digitize(probabilities, bin_edges[1:-1]),
        0,
        n_bins - 1,
    )

    ece = 0.0

    for bin_number in range(n_bins):
        mask = bin_ids == bin_number
        if mask.any():
            observed_rate = y_true[mask].mean()
            mean_probability = probabilities[mask].mean()
            ece += mask.mean() * abs(observed_rate - mean_probability)

    return float(ece)


def calculate_probability_metrics(y_true, probabilities):
    """Calculate threshold-free ranking and probability-quality metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = validate_probabilities(
        probabilities,
        "Probability metrics",
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, probabilities))
        average_precision = float(
            average_precision_score(y_true, probabilities)
        )
    else:
        roc_auc = np.nan
        average_precision = np.nan

    clipped = np.clip(probabilities, EPS, 1.0 - EPS)

    return {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(
            log_loss(y_true, clipped, labels=[0, 1])
        ),
        "ece": expected_calibration_error(y_true, probabilities),
        "mean_predicted_probability": float(probabilities.mean()),
        "actual_potable_share": float(y_true.mean()),
    }


def calculate_decision_metrics(y_true, predicted_labels):
    """
    Calculate classification and water-safety risk metrics.

    Potable = 1.
    False-safe = actual non-potable (0), predicted potable (1).

    Primary FSR = FP / (FP + TN), i.e. all actual non-potable samples
    form the denominator.
    """
    y_true = np.asarray(y_true, dtype=int)
    predicted_labels = np.asarray(predicted_labels, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted_labels,
        labels=[0, 1],
    ).ravel()

    n_actual_non_potable = int(tn + fp)
    n_actual_potable = int(tp + fn)
    n_declared_potable = int(tp + fp)
    n_samples = len(y_true)

    balanced_accuracy = (
        float(balanced_accuracy_score(y_true, predicted_labels))
        if len(np.unique(y_true)) == 2 else np.nan
    )

    return {
        "tn": int(tn),
        "fp_false_safe": int(fp),
        "fn_false_unsafe": int(fn),
        "tp": int(tp),
        "n_actual_non_potable": n_actual_non_potable,
        "n_actual_potable": n_actual_potable,
        "n_declared_potable": n_declared_potable,
        "false_safe_rate": safe_divide(fp, n_actual_non_potable),
        "unsafe_share_among_declared_potable": safe_divide(
            fp,
            n_declared_potable,
        ),
        "potable_recall": safe_divide(tp, n_actual_potable),
        "positive_precision": safe_divide(tp, n_declared_potable),
        "specificity": safe_divide(tn, n_actual_non_potable),
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "balanced_accuracy": balanced_accuracy,
        "f1_score": float(
            f1_score(y_true, predicted_labels, zero_division=0)
        ),
        "safe_prediction_share": safe_divide(
            n_declared_potable,
            n_samples,
        ),
    }


# ============================================================
# 5. LOAD AND VALIDATE THE DATASET
# ============================================================

def load_dataset():
    """Load raw data; do not fit or change any preprocessing here."""
    require_file(
        DATA_PATH,
        "Place water_potability.csv inside 01_Data/raw/.",
    )

    df = pd.read_csv(DATA_PATH).reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Potability must contain exactly labels 0 and 1.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    if X.shape[1] != 9:
        raise ValueError(
            f"Expected the project's 9 input features, found {X.shape[1]}. "
            "Check that the original water_potability.csv is being used."
        )

    if not X.columns.is_unique:
        raise ValueError("The dataset contains duplicate feature names.")

    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise TypeError("All input features must be numeric.")

    print("\n===== RAW DATASET =====")
    print("Dataset shape:", df.shape)
    print("Feature count:", X.shape[1])
    print("Target distribution:")
    print(y.value_counts().sort_index().to_string())

    return X, y


# ============================================================
# 6. FOLD ARTIFACTS AND SPLIT VALIDATION
# ============================================================

def get_manifest_indices(manifest, possible_keys, description):
    """Read and strictly validate zero-based integer row indices."""
    for key in possible_keys:
        if key in manifest and manifest[key] is not None:
            raw = np.asarray(manifest[key])

            if raw.ndim != 1 or raw.size == 0:
                raise ValueError(
                    f"{description} must be a non-empty one-dimensional list."
                )

            try:
                numeric = raw.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{description} contains non-numeric row indices."
                ) from exc

            if not np.isfinite(numeric).all():
                raise ValueError(
                    f"{description} contains NaN or infinite indices."
                )

            if not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError(
                    f"{description} must contain integer row indices."
                )

            return numeric.astype(int)

    raise KeyError(
        f"Could not find {description} in split_manifest.json. "
        f"Checked keys: {possible_keys}"
    )


def load_and_validate_manifest(fold, n_rows):
    """
    Load this fold's split manifest and prove train/calibration/test
    separation and complete partitioning of the raw dataset.
    """
    fold_dir = EXPERIMENT_DIR / f"fold_{fold}"
    manifest_path = require_file(
        fold_dir / "split_manifest.json",
        "Run 05_model_training.py for this fold first.",
    )

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    saved_fold = manifest.get("fold")
    if saved_fold is not None and int(saved_fold) != fold:
        raise AssertionError(
            f"Split manifest belongs to fold {saved_fold}, not fold {fold}."
        )

    train_idx = get_manifest_indices(
        manifest,
        ["train_indices", "train_idx"],
        "training indices",
    )
    calibration_idx = get_manifest_indices(
        manifest,
        ["calibration_indices", "calibration_idx"],
        "calibration indices",
    )
    test_idx = get_manifest_indices(
        manifest,
        ["test_indices", "test_idx"],
        "outer-test indices",
    )

    for name, indices in [
        ("training", train_idx),
        ("calibration", calibration_idx),
        ("outer test", test_idx),
    ]:
        if len(indices) == 0:
            raise AssertionError(f"Fold {fold}: {name} split is empty.")

        if len(indices) != len(np.unique(indices)):
            raise AssertionError(
                f"Fold {fold}: duplicate indices in {name} split."
            )

        if indices.min() < 0 or indices.max() >= n_rows:
            raise IndexError(
                f"Fold {fold}: {name} indices are outside the dataset."
            )

    train_set = set(train_idx.tolist())
    calibration_set = set(calibration_idx.tolist())
    test_set = set(test_idx.tolist())

    if train_set & calibration_set:
        raise AssertionError("Training and calibration rows overlap.")

    if train_set & test_set:
        raise AssertionError("Training and outer-test rows overlap.")

    if calibration_set & test_set:
        raise AssertionError("Calibration and outer-test rows overlap.")

    all_indices = train_set | calibration_set | test_set
    if all_indices != set(range(n_rows)):
        raise AssertionError(
            f"Fold {fold}: train/calibration/test do not partition "
            "the raw dataset exactly once."
        )

    # If the training script saved split sizes/class counts, verify them
    # against the raw dataset. These fields are optional for compatibility.
    sizes = manifest.get("sizes", {})
    for split_name, actual_size in [
        ("train", len(train_idx)),
        ("calibration", len(calibration_idx)),
        ("test", len(test_idx)),
    ]:
        if split_name in sizes and int(sizes[split_name]) != actual_size:
            raise AssertionError(
                f"Fold {fold}: manifest size for {split_name} does not match indices."
            )

    print(f"\n===== OUTER FOLD {fold} SPLIT CHECK =====")
    print("Training rows:", len(train_idx))
    print("Calibration rows:", len(calibration_idx))
    print("Untouched test rows:", len(test_idx))
    print("Train/calibration/test overlap check: PASSED")

    return test_idx, manifest_path


def collect_fold_input_files(fold, include_selective):
    """Collect all existing inputs used by one fold for SHA-256 checks."""
    fold_dir = EXPERIMENT_DIR / f"fold_{fold}"

    files = {
        "raw_dataset": require_file(DATA_PATH),
        "water_utils": require_file(
            CODE_DIR / "water_utils.py",
            "Keep water_utils.py inside 02_Code/.",
        ),
        "split_manifest": require_file(
            fold_dir / "split_manifest.json",
            "Run 05_model_training.py first.",
        ),
        "thresholds": require_file(
            fold_dir / "thresholds.json",
            "Run 07_threshold_selection.py first.",
        ),
    }

    for model_name in MODELS:
        files[f"model_{model_name}"] = require_file(
            fold_dir / "models" / f"{model_name}.joblib",
            "Run 05_model_training.py for this fold first.",
        )
        files[f"calibrator_{model_name}"] = require_file(
            fold_dir / "calibrators" / f"{model_name}_calibrators.joblib",
            "Run 06_probability_calibration.py for this fold first.",
        )

    selection_table = TABLE_DIR / f"threshold_selection_fold_{fold}.csv"
    if selection_table.is_file():
        files["threshold_selection_table"] = selection_table

    selective_path = fold_dir / "selective_rules.json"
    if include_selective and selective_path.is_file():
        files["selective_rules"] = selective_path

        # Lesson 08 writes both an audit manifest and the calibration
        # prediction file whose SHA-256 is recorded in that manifest.
        audit_path = require_file(
            fold_dir / "selective_manifest.json",
            "Re-run 08_selective_prediction.py for this fold to create its audit manifest.",
        )
        calibration_predictions_path = require_file(
            TABLE_DIR / f"calibrated_predictions_fold_{fold}.csv",
            "Re-run 06_probability_calibration.py for this fold.",
        )
        files["selective_manifest"] = audit_path
        files["calibrated_predictions"] = calibration_predictions_path

    return files


# ============================================================
# 7. LOAD SAVED THRESHOLDS AND OPTIONAL SELECTIVE RULES
# ============================================================

def load_threshold_configuration(fold):
    """Load thresholds selected earlier; never select new thresholds."""
    path = EXPERIMENT_DIR / f"fold_{fold}" / "thresholds.json"
    require_file(path, "Run 07_threshold_selection.py first.")

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if "thresholds" not in payload:
        raise KeyError(
            f"'thresholds' section is missing from:\n{path}"
        )

    saved_fold = payload.get("fold")
    if saved_fold is not None and int(saved_fold) != fold:
        raise AssertionError(
            f"Threshold file belongs to fold {saved_fold}, not fold {fold}."
        )

    if payload.get("outer_test_used_for_selection", False):
        raise AssertionError(
            "The threshold file indicates that outer-test data was used "
            "for threshold selection."
        )

    target_fsr = payload.get(
        "target_false_safe_rate",
        payload.get("target_fsr", payload.get("main_risk_target", np.nan)),
    )

    if np.isfinite(target_fsr) and not 0.0 <= float(target_fsr) <= 1.0:
        raise ValueError("Saved target_false_safe_rate must be between 0 and 1.")

    # Lesson 08 deliberately uses the next representable float above 1.0
    # as a reject-all threshold when no ordinary cutoff is feasible.
    reject_all_threshold = payload.get("reject_all_threshold")
    if reject_all_threshold is not None:
        reject_all_threshold = float(reject_all_threshold)
        if (
            not np.isfinite(reject_all_threshold)
            or reject_all_threshold <= 1.0
        ):
            raise ValueError(
                "Saved reject_all_threshold must be finite and strictly "
                "greater than 1.0."
            )

    if payload.get("positive_class", 1) != 1:
        raise AssertionError("Threshold file does not define Potability=1 as positive.")

    if payload.get("primary_false_safe_metric", "FP / (FP + TN)") != "FP / (FP + TN)":
        raise AssertionError(
            "Threshold file uses a different primary false-safe metric."
        )

    print("Calibration-side false-safe target:", target_fsr)
    print("Saved reject-all threshold:", reject_all_threshold)

    return payload["thresholds"], target_fsr, reject_all_threshold


def load_selection_status(fold, target_fsr):
    """Read selection-status labels for reporting only."""
    path = TABLE_DIR / f"threshold_selection_fold_{fold}.csv"

    if not path.is_file():
        return {}

    table = pd.read_csv(path)
    required = {"model", "strategy", "selection_status"}

    if not required.issubset(table.columns):
        return {}

    target_column = None
    for candidate in ["target_false_safe_rate", "target_fsr"]:
        if candidate in table.columns:
            target_column = candidate
            break

    if target_column and np.isfinite(target_fsr):
        table = table[
            np.isclose(
                pd.to_numeric(table[target_column], errors="coerce"),
                float(target_fsr),
                equal_nan=False,
            )
        ]

    status_lookup = {}

    for _, row in table.iterrows():
        key = (str(row["model"]), str(row["strategy"]))
        status_lookup[key] = {
            "selection_status": str(row["selection_status"]),
            "selection_feasible": (
                row["feasible"]
                if "feasible" in table.columns
                else np.nan
            ),
        }

    return status_lookup


def get_threshold_value(
    threshold_map,
    model_name,
    strategy,
    reject_all_threshold=None,
):
    """
    Read an unchanged threshold from thresholds.json.

    Ordinary thresholds must be in [0, 1]. The sole permitted value above
    1.0 is the exact reject-all sentinel saved by 07_threshold_selection.py.
    """
    if model_name not in threshold_map:
        raise KeyError(f"No thresholds saved for model '{model_name}'.")

    model_thresholds = threshold_map[model_name]
    if strategy not in model_thresholds:
        raise KeyError(
            f"Threshold '{strategy}' missing for model '{model_name}'."
        )

    value = model_thresholds[strategy]

    if isinstance(value, dict):
        value = value.get("threshold")

    if value is None:
        raise ValueError(
            f"Invalid saved threshold for {model_name}/{strategy}."
        )

    threshold = float(value)

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            f"Invalid threshold for {model_name}/{strategy}: {threshold}"
        )

    if threshold <= 1.0:
        return threshold

    if (
        reject_all_threshold is not None
        and threshold == float(reject_all_threshold)
    ):
        return threshold

    raise ValueError(
        f"Threshold {threshold} for {model_name}/{strategy} is above 1.0 "
        "but does not exactly match the saved reject_all_threshold. "
        "Refusing an unrecognized threshold."
    )


def load_optional_selective_rules(fold):
    """
    Load selective-prediction rules if they exist.

    This file is optional. The main outer-test evaluation does not depend
    on Lesson 08 being completed.
    """
    path = EXPERIMENT_DIR / f"fold_{fold}" / "selective_rules.json"

    if not path.is_file():
        return None, None

    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    required = {"decision_threshold", "cutoffs"}
    if not required.issubset(config):
        raise KeyError(
            "selective_rules.json must contain 'decision_threshold' "
            "and 'cutoffs'."
        )

    saved_fold = config.get("fold")
    if saved_fold is not None and int(saved_fold) != fold:
        raise AssertionError(
            f"Selective rules belong to fold {saved_fold}, not fold {fold}."
        )

    if config.get("outer_test_used_for_selection", False):
        raise AssertionError(
            "Selective rules indicate that outer-test data was used for selection."
        )

    if config.get("probability_method", "sigmoid") != "sigmoid":
        raise ValueError("Selective rules must use the predeclared sigmoid method.")

    if config.get(
        "probability_source", "out_of_fold_calibration_predictions"
    ) != "out_of_fold_calibration_predictions":
        raise ValueError("Selective cut-offs must originate from sigmoid OOF predictions.")

    if not isinstance(config["cutoffs"], dict):
        raise TypeError("'cutoffs' in selective_rules.json must be a dictionary.")

    decision_threshold = float(config["decision_threshold"])
    if not np.isfinite(decision_threshold) or not 0.0 <= decision_threshold <= 1.0:
        raise ValueError(
            "Selective decision_threshold must be finite and between 0 and 1."
        )

    for model_name, model_cutoffs in config["cutoffs"].items():
        if model_name not in MODELS:
            raise KeyError(f"Unknown model in selective cutoffs: {model_name}")
        if not isinstance(model_cutoffs, dict):
            raise TypeError(f"Cutoffs for {model_name} must be a dictionary.")
        for coverage_text, cutoff_value in model_cutoffs.items():
            coverage = float(coverage_text)
            cutoff = float(cutoff_value)
            if not np.isfinite(coverage) or not 0.0 < coverage <= 1.0:
                raise ValueError(f"Invalid target coverage: {coverage_text}")
            if not np.isfinite(cutoff) or not 0.0 <= cutoff <= 0.5:
                raise ValueError(
                    f"Invalid confidence cutoff for {model_name}/{coverage_text}: {cutoff}"
                )

    return config, path


def validate_selective_audit_manifest(fold, manifest_path):
    """Verify Lesson 08 input and output fingerprints before using its rules."""
    fold_dir = EXPERIMENT_DIR / f"fold_{fold}"
    audit_path = fold_dir / "selective_manifest.json"
    require_file(
        audit_path,
        "Run the updated 08_selective_prediction.py for this fold first.",
    )

    with open(audit_path, "r", encoding="utf-8") as file:
        audit = json.load(file)

    if int(audit.get("fold", -1)) != fold:
        raise AssertionError(
            f"Selective audit manifest belongs to fold {audit.get('fold')}, not {fold}."
        )
    if audit.get("outer_test_used_for_selection", True):
        raise AssertionError("Lesson 08 audit says outer-test data was used for selection.")
    if audit.get("probability_source") != "sigmoid_oof":
        raise AssertionError(
            "Lesson 08 audit does not identify sigmoid OOF as the probability source."
        )

    # 1. Verify the inputs used to select the selective cut-offs.
    saved_input_hashes = audit.get("input_sha256", {})
    expected_inputs = {
        "raw_dataset": DATA_PATH,
        "split_manifest": manifest_path,
        "calibrated_predictions": (
            TABLE_DIR / f"calibrated_predictions_fold_{fold}.csv"
        ),
    }

    for key, source_path in expected_inputs.items():
        require_file(source_path)
        saved_hash = saved_input_hashes.get(key)
        if not saved_hash:
            raise KeyError(
                f"Lesson 08 audit manifest is missing input_sha256['{key}']."
            )
        current_hash = sha256_file(source_path)
        if saved_hash != current_hash:
            raise AssertionError(
                f"Lesson 08 {key} fingerprint does not match the current file. "
                "Do not evaluate until the original inputs are restored or "
                "Lesson 08 is rerun consistently."
            )

    # 2. Verify the outputs generated by Lesson 08. This is the key extra
    #    check: it detects a manually edited selective_rules.json/cut-off.
    expected_outputs = {
        "selective_rules": fold_dir / "selective_rules.json",
        "summary_table": TABLE_DIR / f"selective_prediction_fold_{fold}.csv",
        "row_level_predictions": (
            TABLE_DIR / f"selective_calibration_predictions_fold_{fold}.csv"
        ),
    }
    saved_output_hashes = audit.get("output_sha256", {})
    saved_output_paths = audit.get("outputs", {})

    if int(audit.get("audit_schema_version", 0)) < 2:
        raise RuntimeError(
            "This fold has an older Lesson 08 audit manifest without output "
            "fingerprints. Re-run the updated script first:\n"
            f"python 02_Code/08_selective_prediction.py --fold {fold}"
        )

    for key, output_path in expected_outputs.items():
        require_file(output_path)

        saved_hash = saved_output_hashes.get(key)
        if not saved_hash:
            raise KeyError(
                f"Lesson 08 audit manifest is missing output_sha256['{key}']. "
                f"Re-run 08_selective_prediction.py --fold {fold}."
            )

        expected_relative_path = str(output_path.relative_to(PROJECT_DIR))
        recorded_path = saved_output_paths.get(key)
        if recorded_path != expected_relative_path:
            raise AssertionError(
                f"Lesson 08 recorded output path for '{key}' does not match "
                "the expected project path."
            )

        current_hash = sha256_file(output_path)
        if saved_hash != current_hash:
            raise AssertionError(
                f"Lesson 08 output fingerprint mismatch for '{key}'. "
                "The saved selective rules or result outputs may have been "
                "modified after Lesson 08. Restore the original files or "
                f"re-run 08_selective_prediction.py --fold {fold}."
            )

    print("Lesson 08 input SHA-256 checks: PASSED")
    print("Lesson 08 output SHA-256 checks: PASSED")
    print("Selective rules integrity check: PASSED")
    return audit_path


# ============================================================
# 8. LOAD SAVED MODEL AND CALIBRATORS
# ============================================================

def predict_model_probabilities(fold, model_name, X_test):
    """
    Return raw, sigmoid-calibrated and (if available) isotonic-calibrated
    probabilities using this fold's saved artifacts only.
    """
    fold_dir = EXPERIMENT_DIR / f"fold_{fold}"

    model_path = fold_dir / "models" / f"{model_name}.joblib"
    calibrator_path = (
        fold_dir / "calibrators" / f"{model_name}_calibrators.joblib"
    )

    model = joblib.load(model_path)
    calibrators = joblib.load(calibrator_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{model_name} does not support predict_proba.")

    if not hasattr(model, "classes_"):
        raise AttributeError(f"{model_name} does not expose classes_.")

    if "sigmoid" not in calibrators:
        raise KeyError(
            f"Saved sigmoid calibrator missing for {model_name}, fold {fold}."
        )

    # Preserve the feature order used during model fitting.
    model_features = getattr(model, "feature_names_in_", None)

    if model_features is not None:
        model_features = list(model_features)

        if set(model_features) != set(X_test.columns):
            raise ValueError(
                f"Feature mismatch for {model_name}, fold {fold}.\n"
                f"Model features: {model_features}\n"
                f"Dataset features: {list(X_test.columns)}"
            )

        X_model_test = X_test.loc[:, model_features]
    else:
        X_model_test = X_test

    classes = np.asarray(model.classes_)
    positive_positions = np.where(classes == 1)[0]

    if len(positive_positions) != 1:
        raise ValueError(
            f"Could not identify Potability=1 in {model_name}. "
            f"Classes found: {classes}"
        )

    positive_index = int(positive_positions[0])
    raw_probabilities = model.predict_proba(X_model_test)[:, positive_index]
    raw_probabilities = validate_probabilities(
        raw_probabilities,
        f"{model_name} raw",
    )

    # The sigmoid calibrator was fitted on logit(raw probability).
    sigmoid_probabilities = calibrators["sigmoid"].predict_proba(
        to_logit(raw_probabilities)
    )[:, 1]
    sigmoid_probabilities = validate_probabilities(
        sigmoid_probabilities,
        f"{model_name} sigmoid",
    )

    result = {
        "uncalibrated": raw_probabilities,
        "sigmoid": sigmoid_probabilities,
    }

    # Isotonic is a secondary calibration comparison. Threshold decisions
    # remain based on sigmoid probabilities.
    if "isotonic" in calibrators:
        isotonic_probabilities = calibrators["isotonic"].predict(
            raw_probabilities
        )
        isotonic_probabilities = np.clip(
            isotonic_probabilities,
            0.0,
            1.0,
        )
        result["isotonic"] = validate_probabilities(
            isotonic_probabilities,
            f"{model_name} isotonic",
        )

    return result


# ============================================================
# 9. OPTIONAL SELECTIVE-PREDICTION EVALUATION
# ============================================================

def evaluate_selective_rules(
    fold,
    model_name,
    y_true,
    sigmoid_probabilities,
    selective_config,
):
    """Apply saved selective cut-offs without changing them."""
    rows = []

    decision_threshold = float(
        selective_config["decision_threshold"]
    )
    model_cutoffs = selective_config["cutoffs"].get(model_name, {})

    if not model_cutoffs:
        return rows

    confidence = np.abs(sigmoid_probabilities - decision_threshold)

    for coverage, cutoff_value in model_cutoffs.items():
        cutoff = float(cutoff_value)

        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError(
                f"Invalid selective cutoff for {model_name}, "
                f"coverage {coverage}: {cutoff}"
            )

        decided = confidence >= cutoff - 1e-9

        safe = decided & (sigmoid_probabilities >= decision_threshold)
        unsafe = decided & (sigmoid_probabilities < decision_threshold)

        correct = (safe & (y_true == 1)) | (unsafe & (y_true == 0))

        # FSR denominator includes ALL actual non-potable rows.
        # Needs Review is not counted as a false-safe decision.
        all_row_metrics = calculate_decision_metrics(y_true, safe)

        if decided.any():
            decided_metrics = calculate_decision_metrics(
                y_true[decided],
                safe[decided],
            )
            selective_accuracy = float(correct.sum() / decided.sum())
        else:
            decided_metrics = {
                "tn": 0,
                "fp_false_safe": 0,
                "fn_false_unsafe": 0,
                "tp": 0,
                "false_safe_rate": np.nan,
                "potable_recall": np.nan,
                "positive_precision": np.nan,
                "f1_score": np.nan,
            }
            selective_accuracy = np.nan

        n_decided = int(decided.sum())
        n_needs_review = int((~decided).sum())
        n_correct_decided = int(correct.sum())

        rows.append({
            "fold": fold,
            "model": model_name,
            "reference_only": model_name == "majority_baseline",
            "decision_threshold": decision_threshold,
            "target_coverage": float(coverage),
            "confidence_cutoff": cutoff,
            "n_samples": int(len(y_true)),
            "n_decided": n_decided,
            "n_correct_decided": n_correct_decided,
            "n_safe": int(safe.sum()),
            "n_unsafe": int(unsafe.sum()),
            "n_needs_review": n_needs_review,
            "n_false_safe": all_row_metrics["fp_false_safe"],
            "n_actual_non_potable": all_row_metrics["n_actual_non_potable"],
            "n_actual_potable": all_row_metrics["n_actual_potable"],
            "decided_tn": int(decided_metrics.get("tn", 0)),
            "decided_fp_false_safe": int(decided_metrics.get("fp_false_safe", 0)),
            "decided_fn_false_unsafe": int(decided_metrics.get("fn_false_unsafe", 0)),
            "decided_tp": int(decided_metrics.get("tp", 0)),
            "actual_coverage": float(decided.mean()),
            "review_share": float(n_needs_review / len(y_true)),
            "selective_accuracy": selective_accuracy,
            "accepted_error_rate": (
                float(1.0 - selective_accuracy)
                if np.isfinite(selective_accuracy) else np.nan
            ),
            # Primary FSR denominator: ALL actual non-potable test rows.
            "false_safe_rate": all_row_metrics["false_safe_rate"],
            "false_safe_rate_all_non_potable": all_row_metrics["false_safe_rate"],
            # Conditional FSR excludes actual non-potable rows sent to review.
            "conditional_false_safe_rate_decided": (
                decided_metrics["false_safe_rate"]
            ),
            "unsafe_share_among_declared_potable": (
                all_row_metrics["unsafe_share_among_declared_potable"]
            ),
            "unsafe_share_among_declared_safe": (
                all_row_metrics["unsafe_share_among_declared_potable"]
            ),
            "potable_recall_all": all_row_metrics["potable_recall"],
            "recall_among_decided": decided_metrics["potable_recall"],
            "potable_recall_among_decided": decided_metrics["potable_recall"],
            "precision_among_decided": decided_metrics["positive_precision"],
            "f1_among_decided": decided_metrics["f1_score"],
        })

    return rows


# ============================================================
# 10. FIGURE CREATION
# ============================================================

def create_confusion_matrix_figure(fold, y_true, predictions):
    """Create a 4-model by 4-threshold confusion-matrix figure."""
    fig, axes = plt.subplots(
        len(MODELS),
        len(THRESHOLD_STRATEGIES),
        figsize=(16, 13),
    )

    for row_index, model_name in enumerate(MODELS):
        for col_index, strategy in enumerate(THRESHOLD_STRATEGIES):
            ax = axes[row_index, col_index]
            column = f"{model_name}_{strategy}_prediction"
            predicted = predictions[column].to_numpy(dtype=int)

            matrix = confusion_matrix(
                y_true,
                predicted,
                labels=[0, 1],
            )

            ax.imshow(matrix, interpolation="nearest")

            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        str(matrix[i, j]),
                        ha="center",
                        va="center",
                    )

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["0", "1"])
            ax.set_yticklabels(["0", "1"])

            if row_index == 0:
                ax.set_title(strategy)

            if col_index == 0:
                ax.set_ylabel(
                    model_name.replace("_", " ") + "\nActual"
                )

            if row_index == len(MODELS) - 1:
                ax.set_xlabel("Predicted")

    fig.suptitle(
        f"Outer-Test Confusion Matrices — Fold {fold}\n"
        "Rows: actual class | Columns: predicted class",
        y=1.01,
    )
    fig.tight_layout()

    output_path = (
        FIGURE_DIR / f"evaluation_confusion_matrices_fold_{fold}.png"
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved confusion matrices:", output_path)


def create_probability_figures(fold, y_true, probability_store):
    """Create sigmoid ROC curves and reliability diagrams."""
    # ROC curves use the main sigmoid-calibrated probabilities.
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name in MODELS:
        probabilities = probability_store[model_name]["sigmoid"]
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc_value = roc_auc_score(y_true, probabilities)

        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC={auc_value:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", label="No-skill reference")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"Outer-Test ROC Curves — Fold {fold} (Sigmoid probabilities)"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    roc_path = FIGURE_DIR / f"evaluation_roc_fold_{fold}.png"
    fig.savefig(roc_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Reliability diagrams compare raw, sigmoid and isotonic probabilities.
    fig, axes = plt.subplots(
        1,
        len(MODELS),
        figsize=(16, 4.5),
        sharex=True,
        sharey=True,
    )

    for ax, model_name in zip(axes, MODELS):
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

        for method_name, probabilities in probability_store[model_name].items():
            fraction_positive, mean_predicted = calibration_curve(
                y_true,
                probabilities,
                n_bins=N_BINS,
                strategy="uniform",
            )

            ax.plot(
                mean_predicted,
                fraction_positive,
                marker="o",
                label=method_name,
            )

        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Observed fraction potable")
    axes[-1].legend(loc="best", fontsize=8)

    fig.suptitle(f"Outer-Test Reliability Diagrams — Fold {fold}")
    fig.tight_layout()

    reliability_path = (
        FIGURE_DIR / f"evaluation_reliability_fold_{fold}.png"
    )
    fig.savefig(reliability_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Saved ROC curve:", roc_path)
    print("Saved reliability diagrams:", reliability_path)


# ============================================================
# 11. EVALUATE ONE OUTER FOLD
# ============================================================

def run_fold(fold, X_all, y_all):
    print("\n" + "=" * 72)
    print(f"FINAL EVALUATION — OUTER FOLD {fold}")
    print("Untouched outer-test data; no fitting or threshold selection")
    print("=" * 72)

    fold_dir = EXPERIMENT_DIR / f"fold_{fold}"

    # Read optional selective rules before collecting fingerprinted inputs.
    selective_config, selective_path = load_optional_selective_rules(fold)

    input_files = collect_fold_input_files(
        fold,
        include_selective=selective_config is not None,
    )
    if selective_path is not None:
        input_files["selective_rules"] = selective_path

    hashes_before = {
        name: sha256_file(path)
        for name, path in input_files.items()
    }

    # Validate the fold's original split.
    test_indices, manifest_path = load_and_validate_manifest(
        fold,
        n_rows=len(X_all),
    )

    X_test = X_all.iloc[test_indices].copy()
    y_test = y_all.iloc[test_indices].to_numpy(dtype=int)
    if set(np.unique(y_test)) != {0, 1}:
        raise ValueError(
            f"Fold {fold}: outer-test split must contain both Potability classes."
        )

    if selective_config is not None:
        validate_selective_audit_manifest(fold, manifest_path)

    threshold_map, target_fsr, reject_all_threshold = (
        load_threshold_configuration(fold)
    )
    selection_status = load_selection_status(fold, target_fsr)

    print("\nOuter-test rows:", len(y_test))
    print("Actual non-potable (0):", int((y_test == 0).sum()))
    print("Actual potable (1):", int((y_test == 1).sum()))

    prediction_output = pd.DataFrame({
        "fold": fold,
        "row_index": test_indices,
        "y_true": y_test,
    })

    probability_rows = []
    decision_rows = []
    selective_rows = []
    probability_store = {}

    for model_name in MODELS:
        print(f"\n--- Evaluating: {model_name} ---")

        probabilities_by_method = predict_model_probabilities(
            fold,
            model_name,
            X_test,
        )
        probability_store[model_name] = probabilities_by_method

        # Save every available probability method per test row.
        for method_name, probabilities in probabilities_by_method.items():
            prediction_output[
                f"{model_name}_{method_name}_probability"
            ] = probabilities

            probability_rows.append({
                "fold": fold,
                "model": model_name,
                "probability_method": method_name,
                **calculate_probability_metrics(y_test, probabilities),
            })

        # Thresholds are applied ONLY to sigmoid probabilities.
        sigmoid_probabilities = probabilities_by_method["sigmoid"]

        for strategy in THRESHOLD_STRATEGIES:
            threshold = get_threshold_value(
                threshold_map,
                model_name,
                strategy,
                reject_all_threshold=reject_all_threshold,
            )

            predicted_labels = (
                sigmoid_probabilities >= threshold
            ).astype(int)

            prediction_output[
                f"{model_name}_{strategy}_prediction"
            ] = predicted_labels
            prediction_output[
                f"{model_name}_{strategy}_threshold"
            ] = threshold

            selection_record = selection_status.get(
                (model_name, strategy),
                {
                    "selection_status": "NOT_RECORDED",
                    "selection_feasible": np.nan,
                },
            )
            status = selection_record["selection_status"]
            selection_feasible = selection_record["selection_feasible"]

            is_reference_only = model_name == "majority_baseline"

            if is_reference_only:
                if strategy == "fixed_0.50":
                    status = "PREDEFINED_REFERENCE"
                else:
                    status = "REFERENCE_ONLY_CONSTANT_BASELINE"
                selection_feasible = np.nan

            decision_rows.append({
                "fold": fold,
                "model": model_name,
                "strategy": strategy,
                "reference_only": is_reference_only,
                "selection_status": status,
                "selection_feasible": selection_feasible,
                "threshold": threshold,
                "is_reject_all_threshold": bool(threshold > 1.0),
                "calibration_target_false_safe_rate": target_fsr,
                **calculate_decision_metrics(
                    y_test,
                    predicted_labels,
                ),
            })

        # Selective evaluation is optional and never changes the main rules.
        if selective_config is not None:
            selective_rows.extend(
                evaluate_selective_rules(
                    fold,
                    model_name,
                    y_test,
                    sigmoid_probabilities,
                    selective_config,
                )
            )

    # Verify that none of the input artifacts changed during evaluation.
    hashes_after = {
        name: sha256_file(path)
        for name, path in input_files.items()
    }

    if hashes_before != hashes_after:
        changed = [
            name
            for name in hashes_before
            if hashes_before[name] != hashes_after[name]
        ]
        raise AssertionError(
            f"Input artifacts changed during evaluation: {changed}"
        )

    # Save fold-level tables.
    probability_df = pd.DataFrame(probability_rows).round(6)

    # Preserve the exact saved threshold, especially the reject-all sentinel
    # (1.0000000000000002). Rounding it to 1.0 would change its meaning.
    decision_df = pd.DataFrame(decision_rows)
    decision_numeric_columns = decision_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    decision_round_columns = [
        column for column in decision_numeric_columns
        if column != "threshold"
    ]
    decision_df[decision_round_columns] = decision_df[
        decision_round_columns
    ].round(6)

    selective_df = pd.DataFrame(selective_rows).round(6)

    output_paths = {
        "predictions": (
            TABLE_DIR / f"test_predictions_fold_{fold}.csv"
        ),
        "probability_metrics": (
            TABLE_DIR / f"evaluation_probability_fold_{fold}.csv"
        ),
        "threshold_metrics": (
            TABLE_DIR / f"evaluation_thresholds_fold_{fold}.csv"
        ),
    }

    prediction_output.to_csv(output_paths["predictions"], index=False)
    probability_df.to_csv(
        output_paths["probability_metrics"],
        index=False,
    )
    decision_df.to_csv(
        output_paths["threshold_metrics"],
        index=False,
    )

    if selective_config is not None:
        output_paths["selective_metrics"] = (
            TABLE_DIR / f"evaluation_selective_fold_{fold}.csv"
        )
        selective_df.to_csv(
            output_paths["selective_metrics"],
            index=False,
        )

    # Save an audit manifest for this fold.
    evaluation_manifest_path = fold_dir / "evaluation_manifest.json"

    manifest_payload = {
        "fold": fold,
        "n_test_rows": int(len(y_test)),
        "models_retrained": False,
        "thresholds_reselected": False,
        "threshold_selection_probability_source": "sigmoid_oof_calibration_predictions",
        "outer_test_probability_source": "saved_full_calibration_fit_sigmoid_calibrator",
        "selective_cutoff_selection_probability_source": (
            "sigmoid_oof_calibration_predictions"
            if selective_config is not None else None
        ),
        "reject_all_threshold": reject_all_threshold,
        "primary_false_safe_rate": "FP / (FP + TN)",
        "selective_evaluation_performed": selective_config is not None,
        "input_sha256": hashes_before,
        "inputs_unchanged_after_evaluation": True,
        "outputs": {
            name: str(path.relative_to(PROJECT_DIR))
            for name, path in output_paths.items()
        },
        "figures": {
            "confusion_matrices": (
                f"04_Results/figures/"
                f"evaluation_confusion_matrices_fold_{fold}.png"
            ),
            "roc": (
                f"04_Results/figures/evaluation_roc_fold_{fold}.png"
            ),
            "reliability": (
                f"04_Results/figures/"
                f"evaluation_reliability_fold_{fold}.png"
            ),
        },
    }

    with open(
        evaluation_manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest_payload, file, indent=4)

    # Create figures only after metrics and predictions have been saved.
    create_confusion_matrix_figure(
        fold,
        y_test,
        prediction_output,
    )
    create_probability_figures(
        fold,
        y_test,
        probability_store,
    )

    # Print concise, thesis-relevant results.
    print("\n===== OUTER-TEST PROBABILITY METRICS =====")
    print(
        probability_df[
            [
                "model",
                "probability_method",
                "roc_auc",
                "average_precision",
                "brier_score",
                "log_loss",
                "ece",
            ]
        ].to_string(index=False)
    )

    print("\n===== SAVED THRESHOLDS APPLIED TO OUTER TEST =====")
    print(
        decision_df[
            [
                "model",
                "strategy",
                "threshold",
                "fp_false_safe",
                "n_actual_non_potable",
                "false_safe_rate",
                "potable_recall",
                "positive_precision",
                "f1_score",
            ]
        ].to_string(index=False)
    )

    if selective_config is None:
        print(
            "\nSelective evaluation skipped: selective_rules.json "
            "was not found. Main evaluation is unaffected."
        )
    else:
        print(
            "\nSelective evaluation completed using saved rules only."
        )

    print("\nInput SHA-256 checks: PASSED")
    print("Evaluation manifest:", evaluation_manifest_path)

    return (
        prediction_output,
        probability_df,
        decision_df,
        selective_df,
    )


# ============================================================
# 12. FIVE-FOLD SUMMARY
# ============================================================

def create_pooled_decision_summary(all_predictions):
    """Calculate pooled decisions while retaining fold-specific thresholds."""
    rows = []

    y_true = all_predictions["y_true"].to_numpy(dtype=int)

    for model_name in MODELS:
        sigmoid_column = f"{model_name}_sigmoid_probability"
        probabilities = all_predictions[sigmoid_column].to_numpy(dtype=float)
        probability_summary = calculate_probability_metrics(
            y_true,
            probabilities,
        )

        for strategy in THRESHOLD_STRATEGIES:
            prediction_column = f"{model_name}_{strategy}_prediction"
            predicted = all_predictions[
                prediction_column
            ].to_numpy(dtype=int)

            threshold_column = f"{model_name}_{strategy}_threshold"
            thresholds = all_predictions[
                threshold_column
            ].to_numpy(dtype=float)

            rows.append({
                "model": model_name,
                "strategy": strategy,
                "n_folds": int(all_predictions["fold"].nunique()),
                "mean_applied_threshold": float(thresholds.mean()),
                **calculate_decision_metrics(y_true, predicted),
                "sigmoid_roc_auc": probability_summary["roc_auc"],
                "sigmoid_average_precision": (
                    probability_summary["average_precision"]
                ),
                "sigmoid_brier_score": probability_summary["brier_score"],
                "sigmoid_log_loss": probability_summary["log_loss"],
                "sigmoid_ece": probability_summary["ece"],
            })

    return pd.DataFrame(rows)


def summarize_all_folds(
    all_predictions,
    all_probability_metrics,
    all_decision_metrics,
    all_selective_metrics,
):
    """Save fold details, fold mean/SD and pooled outer-test metrics."""
    # Save combined row-level predictions and fold-level metric details.
    all_predictions_path = (
        TABLE_DIR / "test_predictions_all_folds.csv"
    )
    all_probability_path = (
        TABLE_DIR / "evaluation_probability_all_folds.csv"
    )
    all_decision_path = (
        TABLE_DIR / "evaluation_thresholds_all_folds.csv"
    )

    all_predictions.to_csv(all_predictions_path, index=False)
    all_probability_metrics.to_csv(all_probability_path, index=False)
    all_decision_metrics.to_csv(all_decision_path, index=False)

    if not all_selective_metrics.empty:
        all_selective_path = (
            TABLE_DIR / "evaluation_selective_all_folds.csv"
        )
        all_selective_metrics.to_csv(all_selective_path, index=False)

        selective_numeric = [
            "confidence_cutoff",
            "actual_coverage",
            "review_share",
            "selective_accuracy",
            "accepted_error_rate",
            "false_safe_rate_all_non_potable",
            "conditional_false_safe_rate_decided",
            "unsafe_share_among_declared_safe",
            "potable_recall_all",
            "potable_recall_among_decided",
            "precision_among_decided",
            "f1_among_decided",
        ]
        selective_numeric = [
            column for column in selective_numeric
            if column in all_selective_metrics.columns
        ]
        selective_summary = (
            all_selective_metrics
            .groupby(["model", "target_coverage"])[selective_numeric]
            .agg(["mean", "std"])
            .round(5)
        )
        selective_summary.columns = [
            f"{column}_{stat}"
            for column, stat in selective_summary.columns
        ]
        selective_summary = selective_summary.reset_index()
        selective_summary_path = (
            TABLE_DIR / "evaluation_selective_summary_mean_std.csv"
        )
        selective_summary.to_csv(selective_summary_path, index=False)

        # Pooled selective metrics are reconstructed from integer counts,
        # rather than averaging fold-level percentages.
        pooled_selective_rows = []
        for (model_name, target_coverage), group in all_selective_metrics.groupby(
            ["model", "target_coverage"], sort=False
        ):
            n_samples = int(group["n_samples"].sum())
            n_decided = int(group["n_decided"].sum())
            n_review = int(group["n_needs_review"].sum())
            n_correct = int(group["n_correct_decided"].sum())
            n_false_safe = int(group["n_false_safe"].sum())
            n_actual_non_potable = int(group["n_actual_non_potable"].sum())
            n_actual_potable = int(group["n_actual_potable"].sum())
            decided_tn = int(group["decided_tn"].sum())
            decided_fp = int(group["decided_fp_false_safe"].sum())
            decided_fn = int(group["decided_fn_false_unsafe"].sum())
            decided_tp = int(group["decided_tp"].sum())
            n_safe = int(group["n_safe"].sum())

            precision = safe_divide(decided_tp, decided_tp + decided_fp)
            recall_decided = safe_divide(decided_tp, decided_tp + decided_fn)
            f1_value = (
                float(2 * precision * recall_decided / (precision + recall_decided))
                if np.isfinite(precision) and np.isfinite(recall_decided)
                and (precision + recall_decided) > 0 else 0.0
            )

            pooled_selective_rows.append({
                "model": model_name,
                "target_coverage": float(target_coverage),
                "n_folds": int(group["fold"].nunique()),
                "n_samples": n_samples,
                "n_decided": n_decided,
                "n_needs_review": n_review,
                "actual_coverage": safe_divide(n_decided, n_samples),
                "review_share": safe_divide(n_review, n_samples),
                "selective_accuracy": safe_divide(n_correct, n_decided),
                "accepted_error_rate": safe_divide(n_decided - n_correct, n_decided),
                "n_false_safe": n_false_safe,
                "n_actual_non_potable": n_actual_non_potable,
                "false_safe_rate_all_non_potable": safe_divide(
                    n_false_safe, n_actual_non_potable
                ),
                "conditional_false_safe_rate_decided": safe_divide(
                    decided_fp, decided_fp + decided_tn
                ),
                "unsafe_share_among_declared_safe": safe_divide(
                    decided_fp, n_safe
                ),
                "potable_recall_all": safe_divide(decided_tp, n_actual_potable),
                "potable_recall_among_decided": recall_decided,
                "precision_among_decided": precision,
                "f1_among_decided": f1_value,
            })

        pooled_selective = pd.DataFrame(pooled_selective_rows)
        pooled_selective_path = (
            TABLE_DIR / "evaluation_selective_metrics_pooled.csv"
        )
        pooled_selective.to_csv(pooled_selective_path, index=False)

    # Mean and standard deviation across the five fold-level results.
    decision_numeric = [
        "threshold",
        "false_safe_rate",
        "potable_recall",
        "positive_precision",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "safe_prediction_share",
    ]
    decision_numeric = [
        col for col in decision_numeric
        if col in all_decision_metrics.columns
    ]

    decision_summary = (
        all_decision_metrics
        .groupby(["model", "strategy"])[decision_numeric]
        .agg(["mean", "std"])
        .round(5)
    )

    reject_all_counts = (
        all_decision_metrics
        .groupby(["model", "strategy"])["is_reject_all_threshold"]
        .sum()
        .rename("reject_all_folds")
    )
    decision_summary.columns = [
        f"{column}_{stat}"
        for column, stat in decision_summary.columns
    ]
    decision_summary = decision_summary.join(reject_all_counts).reset_index()

    decision_summary_path = (
        TABLE_DIR / "evaluation_threshold_summary_mean_std.csv"
    )
    decision_summary.to_csv(decision_summary_path, index=False)

    probability_numeric = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "ece",
        "mean_predicted_probability",
        "actual_potable_share",
    ]
    probability_numeric = [
        col for col in probability_numeric
        if col in all_probability_metrics.columns
    ]

    probability_summary = (
        all_probability_metrics
        .groupby(["model", "probability_method"])[probability_numeric]
        .agg(["mean", "std"])
        .round(5)
    )
    probability_summary.columns = [
        f"{column}_{stat}"
        for column, stat in probability_summary.columns
    ]
    probability_summary = probability_summary.reset_index()

    probability_summary_path = (
        TABLE_DIR / "evaluation_probability_summary_mean_std.csv"
    )
    probability_summary.to_csv(
        probability_summary_path,
        index=False,
    )

    # Pooled metrics combine all outer-test predictions. Each row appears
    # exactly once, and each row's own fold-specific threshold is retained.
    pooled_decision = create_pooled_decision_summary(all_predictions)
    pooled_decision_path = (
        TABLE_DIR / "evaluation_threshold_metrics_pooled.csv"
    )
    pooled_decision.to_csv(pooled_decision_path, index=False)

    pooled_probability_rows = []
    y_true = all_predictions["y_true"].to_numpy(dtype=int)

    for model_name in MODELS:
        for method_name in PROBABILITY_METHODS:
            column = f"{model_name}_{method_name}_probability"

            if column not in all_predictions.columns:
                continue

            probabilities = all_predictions[column].to_numpy(dtype=float)

            pooled_probability_rows.append({
                "model": model_name,
                "probability_method": method_name,
                **calculate_probability_metrics(y_true, probabilities),
            })

    pooled_probability = pd.DataFrame(pooled_probability_rows)
    pooled_probability_path = (
        TABLE_DIR / "evaluation_probability_metrics_pooled.csv"
    )
    pooled_probability.to_csv(
        pooled_probability_path,
        index=False,
    )

    print("\n===== FIVE-FOLD DECISION SUMMARY (MEAN ± SD) =====")
    print(decision_summary.to_string(index=False))

    print("\n===== POOLED OUTER-TEST DECISION METRICS =====")
    print(pooled_decision.to_string(index=False))

    print("\nSaved combined predictions:", all_predictions_path)
    print("Saved probability details:", all_probability_path)
    print("Saved threshold details:", all_decision_path)
    print("Saved decision mean/SD:", decision_summary_path)
    print("Saved probability mean/SD:", probability_summary_path)
    print("Saved pooled decisions:", pooled_decision_path)
    print("Saved pooled probabilities:", pooled_probability_path)

    if not all_selective_metrics.empty:
        selective_folds = sorted(
            all_selective_metrics["fold"].astype(int).unique().tolist()
        )
        print("Saved combined selective results:", all_selective_path)
        print(
            "Saved selective mean/SD:",
            TABLE_DIR / "evaluation_selective_summary_mean_std.csv",
        )
        print(
            "Saved pooled selective metrics:",
            TABLE_DIR / "evaluation_selective_metrics_pooled.csv",
        )
        if len(selective_folds) != N_OUTER_SPLITS:
            print(
                "WARNING: Selective rules were available for only "
                f"{len(selective_folds)}/{N_OUTER_SPLITS} folds "
                f"({selective_folds}). This selective file is partial; "
                "do not report it as a complete five-fold result."
            )


# ============================================================
# 13. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Final evaluation on untouched outer-test folds. "
            "No model fitting or threshold selection is performed."
        )
    )

    fold_group = parser.add_mutually_exclusive_group()

    fold_group.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Evaluate one outer fold (1-5). Default: 1.",
    )

    fold_group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all five outer folds.",
    )

    args = parser.parse_args()

    if args.all:
        folds = list(range(1, N_OUTER_SPLITS + 1))
    else:
        if not 1 <= args.fold <= N_OUTER_SPLITS:
            parser.error("Fold number must be between 1 and 5.")
        folds = [args.fold]

    X_all, y_all = load_dataset()

    prediction_frames = []
    probability_frames = []
    decision_frames = []
    selective_frames = []

    for fold in folds:
        (
            predictions,
            probability_metrics_df,
            decision_metrics_df,
            selective_metrics_df,
        ) = run_fold(fold, X_all, y_all)

        prediction_frames.append(predictions)
        probability_frames.append(probability_metrics_df)
        decision_frames.append(decision_metrics_df)

        if not selective_metrics_df.empty:
            selective_frames.append(selective_metrics_df)

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    all_probability_metrics = pd.concat(
        probability_frames,
        ignore_index=True,
    )
    all_decision_metrics = pd.concat(
        decision_frames,
        ignore_index=True,
    )

    if selective_frames:
        all_selective_metrics = pd.concat(
            selective_frames,
            ignore_index=True,
        )
    else:
        all_selective_metrics = pd.DataFrame()

    if args.all:
        # Every raw row must be evaluated in exactly one outer test fold.
        if all_predictions["row_index"].duplicated().any():
            raise AssertionError(
                "A raw dataset row appeared in more than one outer-test fold."
            )

        observed_ids = set(
            all_predictions["row_index"].astype(int).tolist()
        )
        expected_ids = set(range(len(X_all)))

        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)
            extra = sorted(observed_ids - expected_ids)
            raise AssertionError(
                "The five outer-test folds do not cover the raw dataset "
                "exactly once.\n"
                f"Missing row IDs (first 20): {missing[:20]}\n"
                f"Unexpected row IDs: {extra[:20]}"
            )

        summarize_all_folds(
            all_predictions,
            all_probability_metrics,
            all_decision_metrics,
            all_selective_metrics,
        )

        print(
            "\nAll", len(X_all),
            "raw dataset rows were evaluated exactly once."
        )

    print("\n" + "=" * 72)
    print("FINAL OUTER-TEST EVALUATION COMPLETED")
    print("=" * 72)
    print("Folds evaluated:", folds)
    print("Models evaluated:", MODELS)
    print("Threshold strategies:", THRESHOLD_STRATEGIES)
    print("Outer-test rows evaluated:", len(all_predictions))
    print("No model was retrained.")
    print("No threshold or selective cut-off was re-selected.")
    print("Threshold decisions used each fold's saved full-fit sigmoid calibrator.")
    print("Thresholds and selective cut-offs were selected from calibration OOF probabilities.")
    print("Saved reject-all thresholds above 1.0 are accepted only when")
    print("they exactly match the sentinel in thresholds.json.")
    print("Majority baseline is reference-only, not a risk-controlled model.")


if __name__ == "__main__":
    main()
