
"""
LESSON 06: MODEL TRAINING

Research design:
1. Outer: Stratified 5-fold cross-validation.
2. Outer training: 80% model-training + 20% calibration.
3. Inner: Stratified 3-fold cross-validation.
4. Hyperparameter tuning: GridSearchCV.
5. Models:
   - Majority Baseline
   - Logistic Regression
   - Random Forest
   - MLP Neural Network

IMPORTANT:
Outer test folds are kept untouched.
Only calibration predictions are generated here.

Usage:
    python 02_Code/05_model_training.py
    python 02_Code/05_model_training.py --fold 3
    python 02_Code/05_model_training.py --all
"""

# ==================================================
# 1. IMPORT LIBRARIES
# ==================================================

import argparse
import json
import time
import sys

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split
)

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler
)


# Import shared helper
CODE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(CODE_DIR))

from water_utils import fix_invalid_ph


# ==================================================
# 2. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "01_Data"
    / "raw"
    / "water_potability.csv"
)

EXPERIMENT_DIR = PROJECT_DIR / "03_Experiments"

TABLE_DIR = PROJECT_DIR / "04_Results" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. CONFIGURATION
# ==================================================

TARGET = "Potability"

RANDOM_STATE = 42

N_OUTER_SPLITS = 5

N_INNER_SPLITS = 3

CALIBRATION_SIZE = 0.20

SCORING = "roc_auc"

N_JOBS = -1


# ==================================================
# 4. DATA VALIDATION
# ==================================================

def load_dataset():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    X = df.drop(columns=[TARGET])

    y = df[TARGET]

    if y.isna().any():

        raise ValueError("Target contains missing values.")

    if set(y.unique()) != {0, 1}:

        raise ValueError(
            "Potability must contain exactly labels 0 and 1."
        )

    if not all(
        pd.api.types.is_numeric_dtype(X[col])
        for col in X.columns
    ):

        raise TypeError(
            "All predictor features must be numeric."
        )

    print("\n===== DATASET VALIDATION =====")

    print("Dataset shape:", df.shape)

    print("Feature count:", X.shape[1])

    print("Target distribution:")

    print(y.value_counts().sort_index())

    print("\nMissing values:")

    print(X.isnull().sum())

    return X, y.astype(int)


# ==================================================
# 5. PREPROCESSING PIPELINES
# ==================================================

def scaled_preprocessor():

    """
    Logistic Regression and MLP.

    Order:
    1. Invalid pH correction
    2. Median imputation
    3. Standard scaling
    """

    return [

        (
            "fix_ph",
            FunctionTransformer(
                fix_invalid_ph,
                validate=False
            )
        ),

        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),

        (
            "scaler",
            StandardScaler()
        )

    ]


def tree_preprocessor():

    """
    Random Forest and Majority Baseline.

    Scaling is not required.
    """

    return [

        (
            "fix_ph",
            FunctionTransformer(
                fix_invalid_ph,
                validate=False
            )
        ),

        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        )

    ]


# ==================================================
# 6. DEFINE MODELS AND PARAMETER GRIDS
# ==================================================

def build_models():

    models = {

        # ------------------------------------------
        # MODEL 1: MAJORITY BASELINE
        # ------------------------------------------

        "majority_baseline": (

            Pipeline(
                tree_preprocessor() + [

                    (
                        "model",
                        DummyClassifier(
                            strategy="prior"
                        )
                    )

                ]
            ),

            {}

        ),


        # ------------------------------------------
        # MODEL 2: LOGISTIC REGRESSION
        # ------------------------------------------

        "logistic_regression": (

            Pipeline(
                scaled_preprocessor() + [

                    (
                        "model",
                        LogisticRegression(
                            max_iter=5000,
                            random_state=RANDOM_STATE
                        )
                    )

                ]
            ),

            {

                "model__C": [
                    0.01,
                    0.1,
                    1,
                    10
                ],

                "model__class_weight": [
                    None,
                    "balanced"
                ]

            }

        ),


        # ------------------------------------------
        # MODEL 3: RANDOM FOREST
        # ------------------------------------------

        "random_forest": (

            Pipeline(
                tree_preprocessor() + [

                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=RANDOM_STATE,
                            n_jobs=1
                        )
                    )

                ]
            ),

            {

                "model__max_depth": [
                    None,
                    10,
                    20
                ],

                "model__min_samples_leaf": [
                    1,
                    5
                ],

                "model__max_features": [
                    "sqrt",
                    0.5
                ],

                "model__class_weight": [
                    None,
                    "balanced"
                ]

            }

        ),


        # ------------------------------------------
        # MODEL 4: MLP NEURAL NETWORK
        # ------------------------------------------

        "mlp": (

            Pipeline(
                scaled_preprocessor() + [

                    (
                        "model",
                        MLPClassifier(
                            max_iter=1000,
                            early_stopping=True,
                            random_state=RANDOM_STATE
                        )
                    )

                ]
            ),

            {

                "model__hidden_layer_sizes": [
                    (32,),
                    (64,),
                    (64, 32)
                ],

                "model__alpha": [
                    0.0001,
                    0.001,
                    0.01
                ],

                "model__learning_rate_init": [
                    0.001,
                    0.01
                ]

            }

        )

    }

    return models


# ==================================================
# 7. TRAIN ONE OUTER FOLD
# ==================================================

def run_fold(fold, X, y, outer_splits):

    print("\n" + "=" * 55)

    print(f"MODEL TRAINING: OUTER FOLD {fold}")

    print("=" * 55)


    # ----------------------------------------------
    # 7.1 Outer split
    # ----------------------------------------------

    outer_train_idx, test_idx = outer_splits[fold - 1]


    # ----------------------------------------------
    # 7.2 Separate training and calibration data
    # ----------------------------------------------

    train_idx, calib_idx = train_test_split(

        outer_train_idx,

        test_size=CALIBRATION_SIZE,

        stratify=y.iloc[outer_train_idx],

        random_state=RANDOM_STATE

    )

    train_idx = np.sort(train_idx)

    calib_idx = np.sort(calib_idx)

    test_idx = np.sort(test_idx)


    # Verify separation

    assert len(set(train_idx) & set(calib_idx)) == 0

    assert len(set(train_idx) & set(test_idx)) == 0

    assert len(set(calib_idx) & set(test_idx)) == 0

    assert len(train_idx) + len(calib_idx) == len(
        outer_train_idx
    )


    # ----------------------------------------------
    # 7.3 Extract datasets
    # ----------------------------------------------

    X_train = X.iloc[train_idx]

    y_train = y.iloc[train_idx]

    X_calib = X.iloc[calib_idx]

    y_calib = y.iloc[calib_idx]


    print("\n===== DATA SPLIT =====")

    print("Model training:", len(X_train))

    print("Calibration:", len(X_calib))

    print("Outer test (untouched):", len(test_idx))


    print("\nTraining class distribution:")

    print(y_train.value_counts().sort_index())

    print("\nCalibration class distribution:")

    print(y_calib.value_counts().sort_index())


    # ----------------------------------------------
    # 7.4 Create output directories
    # ----------------------------------------------

    fold_dir = EXPERIMENT_DIR / f"fold_{fold}"

    model_dir = fold_dir / "models"

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ----------------------------------------------
    # 7.5 Save split manifest
    # ----------------------------------------------

    def get_counts(indices):

        return {

            str(label): int(count)

            for label, count in
            y.iloc[indices].value_counts().sort_index().items()

        }


    manifest = {

        "fold": fold,

        "random_state": RANDOM_STATE,

        "outer_folds": N_OUTER_SPLITS,

        "inner_folds": N_INNER_SPLITS,

        "calibration_fraction": CALIBRATION_SIZE,

        "sizes": {

            "train": len(train_idx),

            "calibration": len(calib_idx),

            "test": len(test_idx)

        },

        "class_counts": {

            "train": get_counts(train_idx),

            "calibration": get_counts(calib_idx)

        },

        "train_indices": train_idx.tolist(),

        "calibration_indices": calib_idx.tolist(),

        "test_indices": test_idx.tolist()

    }


    with open(
        fold_dir / "split_manifest.json",
        "w"
    ) as f:

        json.dump(manifest, f, indent=4)


    # ----------------------------------------------
    # 7.6 Inner cross-validation
    # ----------------------------------------------

    inner_cv = StratifiedKFold(

        n_splits=N_INNER_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE

    )


    # ----------------------------------------------
    # 7.7 Prepare result containers
    # ----------------------------------------------

    summary = []

    calibration_predictions = pd.DataFrame({

        "row_index": calib_idx,

        "y_true": y_calib.values

    })


    # ----------------------------------------------
    # 7.8 Train all models
    # ----------------------------------------------

    for name, (pipeline, param_grid) in build_models().items():

        print("\n------------------------------------------")

        print("Training:", name.upper())

        print("------------------------------------------")

        start_time = time.time()


        # All models use the same inner CV procedure.
        # Baseline has an empty parameter grid.

        search = GridSearchCV(

            estimator=pipeline,

            param_grid=param_grid,

            scoring=SCORING,

            cv=inner_cv,

            n_jobs=N_JOBS,

            refit=True,

            error_score="raise",

            return_train_score=False

        )


        search.fit(X_train, y_train)


        best_model = search.best_estimator_

        best_params = search.best_params_

        inner_score = search.best_score_

        n_candidates = len(
            search.cv_results_["params"]
        )

        fit_seconds = time.time() - start_time


        # ------------------------------------------
        # 7.9 Save fitted model
        # ------------------------------------------

        model_path = (
            model_dir / f"{name}.joblib"
        )

        joblib.dump(
            best_model,
            model_path
        )


        # ------------------------------------------
        # 7.10 Predict calibration probabilities
        # ------------------------------------------

        classes = best_model.classes_

        positive_class_index = np.where(
            classes == 1
        )[0]

        if len(positive_class_index) != 1:

            raise ValueError(
                f"Positive class 1 not found in {name}"
            )

        probabilities = best_model.predict_proba(
            X_calib
        )[:, positive_class_index[0]]


        if not np.isfinite(probabilities).all():

            raise ValueError(
                f"Invalid probabilities generated by {name}"
            )

        if ((probabilities < 0) | (probabilities > 1)).any():

            raise ValueError(
                f"Probabilities outside [0, 1] for {name}"
            )


        calibration_predictions[
            f"{name}_proba"
        ] = probabilities


        # ------------------------------------------
        # 7.11 Record training summary
        # ------------------------------------------

        summary.append({

            "fold": fold,

            "model": name,

            "inner_cv_metric": SCORING,

            "inner_cv_best_score": round(
                inner_score, 5
            ),

            "n_candidates": n_candidates,

            "best_params": json.dumps(
                best_params,
                default=str
            ),

            "train_size": len(train_idx),

            "calibration_size": len(calib_idx),

            "fit_seconds": round(
                fit_seconds, 2
            )

        })


        print(
            f"Inner CV ROC-AUC: {inner_score:.4f}"
        )

        print(
            "Best parameters:",
            best_params
        )

        print(
            f"Training time: {fit_seconds:.2f} seconds"
        )

        print(
            "Saved:",
            model_path
        )


    # ==================================================
    # 8. SAVE FOLD RESULTS
    # ==================================================

    summary_df = pd.DataFrame(summary)


    summary_path = (
        TABLE_DIR /
        f"model_training_summary_fold_{fold}.csv"
    )

    calibration_path = (
        TABLE_DIR /
        f"calibration_predictions_fold_{fold}.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False
    )


    calibration_predictions.to_csv(
        calibration_path,
        index=False
    )


    print("\n===== FOLD COMPLETED =====")

    print("\nModel summary:")

    print(
        summary_df[
            [
                "model",
                "inner_cv_best_score",
                "n_candidates",
                "fit_seconds"
            ]
        ].to_string(index=False)
    )

    print("\nSaved summary:", summary_path)

    print("Saved calibration predictions:", calibration_path)

    return summary_df


# ==================================================
# 9. MAIN FUNCTION
# ==================================================

def main():

    parser = argparse.ArgumentParser(

        description="Water Potability Model Training"

    )


    group = parser.add_mutually_exclusive_group()

    group.add_argument(

        "--fold",

        type=int,

        default=1,

        help="Run one outer fold (1-5)"

    )

    group.add_argument(

        "--all",

        action="store_true",

        help="Run all five outer folds"

    )


    args = parser.parse_args()


    # Load original raw dataset

    X, y = load_dataset()


    print("\n===== OUTER CROSS-VALIDATION SETUP =====")


    outer_cv = StratifiedKFold(

        n_splits=N_OUTER_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE

    )


    outer_splits = list(
        outer_cv.split(X, y)
    )


    if args.all:

        folds = range(
            1,
            N_OUTER_SPLITS + 1
        )

    else:

        if not 1 <= args.fold <= N_OUTER_SPLITS:

            raise ValueError(
                "Fold number must be between 1 and 5."
            )

        folds = [args.fold]


    # ----------------------------------------------
    # Run requested folds
    # ----------------------------------------------

    all_summaries = []

    for fold in folds:

        result = run_fold(
            fold,
            X,
            y,
            outer_splits
        )

        all_summaries.append(result)


    print("\n==========================================")

    print("MODEL TRAINING COMPLETED")

    print("==========================================")


    print("\nModels saved inside:")

    print(EXPERIMENT_DIR)


    print("\nResults saved inside:")

    print(TABLE_DIR)


    print("\nIMPORTANT:")

    print("Outer test folds were not used for training.")

    print("No final test evaluation was performed.")

    print("Calibration predictions are ready for the next lesson.")


# ==================================================
# 10. EXECUTION
# ==================================================

if __name__ == "__main__":

    main()
