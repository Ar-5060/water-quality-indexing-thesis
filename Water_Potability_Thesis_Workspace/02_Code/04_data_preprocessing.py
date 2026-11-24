
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


# ==================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "01_Data"
    / "raw"
    / "water_potability.csv"
)

TARGET = "Potability"
RANDOM_STATE = 42
N_SPLITS = 5


# ==================================================
# 2. LOAD DATASET
# ==================================================

df = pd.read_csv(DATA_PATH)

print("\n===== DATASET LOADED =====")
print("Dataset shape:", df.shape)

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

features = X.columns.tolist()


# ==================================================
# 3. BASIC DATA VALIDATION
# ==================================================

print("\n===== DATA VALIDATION =====")

# Check missing target values
if y.isnull().any():
    raise ValueError("Target column contains missing values.")

# Check target labels
if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Invalid target labels found.")

# Check duplicate rows (report only)
print("Duplicate rows:", df.duplicated().sum())

# Check zero pH values (preserve original values)
print("Zero pH values:", (X["ph"] == 0).sum())

# Check negative values (report only)
print("\nNegative values:")
print((X < 0).sum())

# Missing values before preprocessing
print("\nMissing values:")
print(X.isnull().sum())


# ==================================================
# 4. DEFINE PREPROCESSING PIPELINES
# ==================================================

# Pipeline A: Logistic Regression and MLP
scaled_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Pipeline B: Random Forest
tree_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# ==================================================
# 5. STRATIFIED OUTER 5-FOLD SPLITTING
# ==================================================

outer_cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

print("\n===== OUTER CROSS-VALIDATION =====")
print("Number of folds:", N_SPLITS)

# Use first fold only to test preprocessing
train_idx, test_idx = next(outer_cv.split(X, y))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==================================================
# 6. PREPROCESSING FOR LOGISTIC REGRESSION AND MLP
# ==================================================

X_train_scaled = scaled_preprocessor.fit_transform(X_train)

X_test_scaled = scaled_preprocessor.transform(X_test)

print("\n===== SCALED PREPROCESSING =====")

print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)

print(
    "Training mean (first 3 features):",
    np.mean(X_train_scaled, axis=0)[:3].round(3)
)

print(
    "Training standard deviation (first 3 features):",
    np.std(X_train_scaled, axis=0)[:3].round(3)
)


# ==================================================
# 7. PREPROCESSING FOR RANDOM FOREST
# ==================================================

X_train_tree = tree_preprocessor.fit_transform(X_train)

X_test_tree = tree_preprocessor.transform(X_test)

print("\n===== TREE PREPROCESSING =====")

print("Training shape:", X_train_tree.shape)
print("Testing shape:", X_test_tree.shape)


# ==================================================
# 8. FINAL VALIDATION
# ==================================================

assert np.isfinite(X_train_scaled).all()
assert np.isfinite(X_test_scaled).all()

assert np.isfinite(X_train_tree).all()
assert np.isfinite(X_test_tree).all()

print("\n===== PREPROCESSING VALIDATION =====")
print("Missing values handled successfully.")
print("Scaling completed successfully.")
print("No NaN or infinite values remain.")

print("\n===== PREPROCESSING SETUP COMPLETED =====")

print("Note: Only the first outer fold was tested.")
print("No final model or evaluation results were generated.")
