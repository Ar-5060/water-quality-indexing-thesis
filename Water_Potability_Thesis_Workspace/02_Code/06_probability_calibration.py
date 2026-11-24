"""
PROBABILITY CALIBRATION
1. Uses uncalibrated probabilities from the calibration set created by
   05_model_training.py for each outer fold.
2. Main method: sigmoid (Platt-style) calibration on logit(probability).
3. Isotonic regression is included as a comparison method only.
4. Calibrator comparison uses stratified 5-fold CV INSIDE the calibration
   set. Metrics are calculated from out-of-fold (OOF) calibrated predictions.
5. Final sigmoid and isotonic calibrators are fitted on the FULL calibration
   set and saved for later application to the untouched outer test set.
6. OOF sigmoid probabilities are saved for threshold selection in Lesson 08.
   Do NOT use the full-fit calibration-set probabilities for threshold
   selection because those rows were used to fit the final calibrator.
7. Metrics: Brier score, log loss, and ECE (10 equal-width bins).

IMPORTANT
---------
Outer test folds are never used in this script.
The saved full-fit calibrators are intended for transforming outer-test
probabilities in the final evaluation script.

Usage
-----
python 02_Code/06_probability_calibration.py
python 02_Code/06_probability_calibration.py --fold 3
python 02_Code/06_probability_calibration.py --all
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = PROJECT_DIR / "03_Experiments"
TABLE_DIR = PROJECT_DIR / "04_Results" / "tables"
FIGURE_DIR = PROJECT_DIR / "04_Results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_CALIBRATION_CV = 5
N_BINS = 10
EPS = 1e-6

# Sigmoid is fixed in advance as the primary method.
# Isotonic is a secondary comparison only.
MAIN_METHOD = "sigmoid"

MODELS = [
    "majority_baseline",
    "logistic_regression",
    "random_forest",
    "mlp",
]

METHODS = ["uncalibrated", "sigmoid", "isotonic"]


# ============================================================
# 3. VALIDATION AND CALIBRATION HELPERS
# ============================================================

def validate_probabilities(probabilities, model_name):
    """Check that model probabilities are valid numeric values."""
    probabilities = np.asarray(probabilities, dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(
            f"{model_name} contains NaN or infinite probabilities."
        )

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            f"{model_name} contains probabilities outside [0, 1]."
        )

    return probabilities


def to_logit(probabilities):
    """Convert probabilities to log-odds, clipping 0 and 1 safely."""
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        EPS,
        1 - EPS,
    )

    return np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)


def fit_calibrator(method, probabilities, y_true):
    """Fit a calibration method using calibration-training rows only."""
    if method == "sigmoid":
        calibrator = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
        calibrator.fit(to_logit(probabilities), y_true)
        return calibrator

    if method == "isotonic":
        calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        calibrator.fit(probabilities, y_true)
        return calibrator

    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method, calibrator, probabilities):
    """Apply a previously fitted calibrator to new probabilities."""
    probabilities = validate_probabilities(
        probabilities,
        "Input to calibrator",
    )

    if method == "uncalibrated":
        return probabilities

    if method == "sigmoid":
        return calibrator.predict_proba(
            to_logit(probabilities)
        )[:, 1]

    if method == "isotonic":
        return calibrator.predict(probabilities)

    raise ValueError(f"Unknown calibration method: {method}")


def expected_calibration_error(y_true, probabilities, n_bins=N_BINS):
    """
    Equal-width-bin ECE:
    weighted average of the absolute difference between observed
    positive frequency and mean predicted probability in each bin.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(
        probabilities,
        bin_edges[1:-1],
        right=False,
    )
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0

    for bin_number in range(n_bins):
        mask = bin_ids == bin_number

        if mask.any():
            observed_rate = y_true[mask].mean()
            mean_probability = probabilities[mask].mean()
            bin_weight = mask.mean()

            ece += bin_weight * abs(
                observed_rate - mean_probability
            )

    return float(ece)


def compute_metrics(y_true, probabilities):
    """Calculate probability-quality metrics."""
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities_clipped = np.clip(
        probabilities,
        EPS,
        1 - EPS,
    )

    return {
        "brier": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities_clipped,
                labels=[0, 1],
            )
        ),
        "ece": expected_calibration_error(
            y_true,
            probabilities,
            N_BINS,
        ),
    }


# ============================================================
# 4. LOAD AND VALIDATE ONE FOLD
# ============================================================

def load_fold_data(fold):
    """Load calibration predictions and verify outer-test separation."""
    prediction_path = (
        TABLE_DIR
        / f"calibration_predictions_fold_{fold}.csv"
    )

    manifest_path = (
        EXPERIMENT_DIR
        / f"fold_{fold}"
        / "split_manifest.json"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Calibration predictions not found:\n{prediction_path}\n"
            f"Run 05_model_training.py for fold {fold} first."
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found:\n{manifest_path}"
        )

    predictions = pd.read_csv(prediction_path)

    required_columns = ["row_index", "y_true"]
    required_columns += [
        f"{model_name}_proba" for model_name in MODELS
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in calibration prediction file: "
            f"{missing_columns}"
        )

    if predictions[required_columns].isnull().any().any():
        raise ValueError(
            "Calibration prediction file contains missing values "
            "in required columns."
        )

    if predictions["row_index"].duplicated().any():
        raise ValueError(
            "Duplicate row_index values found in calibration predictions."
        )

    y_true = predictions["y_true"].to_numpy(dtype=int)

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Target must contain only binary labels 0 and 1.")

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Calibration set must contain both target classes."
        )

    if np.bincount(y_true, minlength=2).min() < N_CALIBRATION_CV:
        raise ValueError(
            "Each class needs at least "
            f"{N_CALIBRATION_CV} samples for stratified calibration CV."
        )

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if "test_indices" not in manifest:
        raise KeyError(
            "The split manifest does not contain 'test_indices'."
        )

    calibration_indices = set(
        predictions["row_index"].tolist()
    )
    outer_test_indices = set(manifest["test_indices"])

    if not calibration_indices.isdisjoint(outer_test_indices):
        raise AssertionError(
            "Calibration rows overlap the outer test fold."
        )

    return predictions, y_true


# ============================================================
# 5. CALIBRATE ONE OUTER FOLD
# ============================================================

def run_fold(fold):
    print("\n" + "=" * 60)
    print(f"PROBABILITY CALIBRATION: OUTER FOLD {fold}")
    print("=" * 60)

    predictions, y_true = load_fold_data(fold)

    print("Calibration samples:", len(y_true))
    print("Class distribution:")
    print(pd.Series(y_true).value_counts().sort_index())

    calibrator_dir = (
        EXPERIMENT_DIR
        / f"fold_{fold}"
        / "calibrators"
    )
    calibrator_dir.mkdir(parents=True, exist_ok=True)

    calibration_cv = StratifiedKFold(
        n_splits=N_CALIBRATION_CV,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []
    oof_predictions = predictions[["row_index", "y_true"]].copy()

    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")

        raw_probabilities = validate_probabilities(
            predictions[f"{model_name}_proba"].to_numpy(),
            model_name,
        )

        # Store the untouched base-model probabilities.
        oof_predictions[f"{model_name}_raw"] = raw_probabilities

        # Generate out-of-fold predictions for each calibration method.
        oof_calibrated = {
            "sigmoid": np.zeros(len(y_true), dtype=float),
            "isotonic": np.zeros(len(y_true), dtype=float),
        }

        for fit_indices, validation_indices in calibration_cv.split(
            raw_probabilities.reshape(-1, 1),
            y_true,
        ):
            for method in ["sigmoid", "isotonic"]:
                calibrator = fit_calibrator(
                    method,
                    raw_probabilities[fit_indices],
                    y_true[fit_indices],
                )

                oof_calibrated[method][validation_indices] = (
                    apply_calibrator(
                        method,
                        calibrator,
                        raw_probabilities[validation_indices],
                    )
                )

        # Store OOF probabilities. These are the appropriate calibration
        # probabilities for the next lesson's threshold-selection step.
        oof_predictions[
            f"{model_name}_sigmoid_oof"
        ] = oof_calibrated["sigmoid"]

        oof_predictions[
            f"{model_name}_isotonic_oof"
        ] = oof_calibrated["isotonic"]

        # Compare all methods using OOF predictions.
        probabilities_by_method = {
            "uncalibrated": raw_probabilities,
            "sigmoid": oof_calibrated["sigmoid"],
            "isotonic": oof_calibrated["isotonic"],
        }

        for method in METHODS:
            metric_values = compute_metrics(
                y_true,
                probabilities_by_method[method],
            )

            results.append({
                "fold": fold,
                "model": model_name,
                "method": method,
                "main_method": method == MAIN_METHOD,
                "brier": metric_values["brier"],
                "log_loss": metric_values["log_loss"],
                "ece": metric_values["ece"],
            })

        # Fit final calibrators on the FULL calibration set.
        # These saved objects will later transform outer-test probabilities.
        final_calibrators = {}

        for method in ["sigmoid", "isotonic"]:
            final_calibrators[method] = fit_calibrator(
                method,
                raw_probabilities,
                y_true,
            )

        # Save full-fit calibration-set probabilities separately for
        # diagnostics only. Do not use these columns for threshold tuning.
        oof_predictions[
            f"{model_name}_sigmoid_full_fit"
        ] = apply_calibrator(
            "sigmoid",
            final_calibrators["sigmoid"],
            raw_probabilities,
        )

        oof_predictions[
            f"{model_name}_isotonic_full_fit"
        ] = apply_calibrator(
            "isotonic",
            final_calibrators["isotonic"],
            raw_probabilities,
        )

        calibrator_path = (
            calibrator_dir
            / f"{model_name}_calibrators.joblib"
        )

        joblib.dump(final_calibrators, calibrator_path)

        print("Saved calibrators:", calibrator_path)

    # --------------------------------------------------------
    # 5.1 Save metrics and calibrated predictions
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    comparison_path = (
        TABLE_DIR
        / f"calibration_comparison_fold_{fold}.csv"
    )

    predictions_path = (
        TABLE_DIR
        / f"calibrated_predictions_fold_{fold}.csv"
    )

    results_df.to_csv(comparison_path, index=False)
    oof_predictions.to_csv(predictions_path, index=False)

    # --------------------------------------------------------
    # 5.2 Reliability diagrams using OOF predictions
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(MODELS),
        figsize=(17, 4.5),
        sharex=True,
        sharey=True,
    )

    styles = {
        "uncalibrated": ("#777777", "o"),
        "sigmoid": ("#1f77b4", "s"),
        "isotonic": ("#d62728", "^"),
    }

    for axis, model_name in zip(axes, MODELS):
        axis.plot(
            [0, 1],
            [0, 1],
            "k--",
            linewidth=1,
            label="Perfect calibration",
        )

        probabilities_by_method = {
            "uncalibrated": oof_predictions[
                f"{model_name}_raw"
            ].to_numpy(),
            "sigmoid": oof_predictions[
                f"{model_name}_sigmoid_oof"
            ].to_numpy(),
            "isotonic": oof_predictions[
                f"{model_name}_isotonic_oof"
            ].to_numpy(),
        }

        for method in METHODS:
            fraction_positive, mean_predicted = calibration_curve(
                y_true,
                probabilities_by_method[method],
                n_bins=N_BINS,
                strategy="uniform",
            )

            color, marker = styles[method]

            axis.plot(
                mean_predicted,
                fraction_positive,
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=4,
                label=method,
            )

        axis.set_title(model_name.replace("_", " ").title())
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Mean predicted probability")
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Observed fraction potable")
    axes[-1].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Reliability Diagrams — Outer Fold {fold} "
        "(out-of-fold calibration-set predictions)"
    )
    fig.tight_layout()

    figure_path = (
        FIGURE_DIR
        / f"reliability_diagram_fold_{fold}.png"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n===== FOLD CALIBRATION RESULTS =====")
    print(
        results_df[
            ["model", "method", "brier", "log_loss", "ece"]
        ].round(5).to_string(index=False)
    )

    print("\nSaved comparison:", comparison_path)
    print("Saved calibrated predictions:", predictions_path)
    print("Saved reliability diagram:", figure_path)

    return results_df


# ============================================================
# 6. MAIN FUNCTION
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Water Potability Probability Calibration"
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
            raise ValueError("Fold number must be between 1 and 5.")

        folds = [args.fold]

    all_results = pd.concat(
        [run_fold(fold) for fold in folds],
        ignore_index=True,
    )

    if args.all:
        summary = (
            all_results
            .groupby(["model", "method"])[
                ["brier", "log_loss", "ece"]
            ]
            .agg(["mean", "std"])
            .round(5)
        )

        summary.columns = [
            f"{metric}_{statistic}"
            for metric, statistic in summary.columns
        ]
        summary = summary.reset_index()

        summary_path = (
            TABLE_DIR / "calibration_summary_all_folds.csv"
        )
        summary.to_csv(summary_path, index=False)

        print("\n===== SUMMARY ACROSS ALL OUTER FOLDS =====")
        print(summary.to_string(index=False))
        print("Saved summary:", summary_path)

    print("\n" + "=" * 60)
    print("PROBABILITY CALIBRATION COMPLETED")
    print("=" * 60)
    print("Outer test folds were not used.")
    print("Sigmoid is the fixed primary calibration method.")
    print("OOF sigmoid probabilities are ready for threshold selection.")
    print("Final calibrators are saved for later outer-test evaluation.")


if __name__ == "__main__":
    main()
