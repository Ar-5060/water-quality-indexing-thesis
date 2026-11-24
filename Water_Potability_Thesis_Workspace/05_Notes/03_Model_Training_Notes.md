# Lesson 03 — Model Training and Nested Cross-Validation

## Purpose

Train and tune the selected models while keeping each outer test fold untouched until final evaluation.

## Validation design

1. **Outer layer:** Stratified 5-fold cross-validation.
2. Each outer fold: approximately 80% development data and 20% untouched test data.
3. Development data is divided into:
   - 80% model-training subset;
   - 20% calibration subset.
4. **Inner layer:** Stratified 3-fold CV for hyperparameter selection.
5. Hyperparameter tuning uses `GridSearchCV` with ROC-AUC scoring.

For the recorded Fold 1:
- Model-training rows: **2,096**
- Calibration rows: **524**
- Untouched outer-test rows: **656**

## Models

| Model | Role |
|---|---|
| Majority Baseline | Reference-only baseline |
| Logistic Regression | Linear classification baseline |
| Random Forest | Tree-ensemble machine-learning model |
| MLP | Feed-forward neural network for tabular data |

The Majority Baseline is kept for comparison. It is not treated as a useful risk-controlled model.

## Fold 1 inner-CV result

Recorded output from `05_model_training.py`:

| Model | Best inner-CV ROC-AUC |
|---|---:|
| Majority Baseline | 0.5000 |
| Logistic Regression | 0.4889 |
| Random Forest | 0.6576 |
| MLP | 0.6690 |

These are Fold 1 hyperparameter-selection scores. They are **not** final outer-test scores and do not prove MLP is the best model overall.

## Saved outputs

Inside each `03_Experiments/fold_k/` folder:
- `models/` — saved model pipelines (`.joblib`)
- `split_manifest.json` — row indices and split information

In `04_Results/tables/`:
- `model_training_summary_fold_k.csv`
- `calibration_predictions_fold_k.csv`

The exact set of outputs depends on the final script version. Keep all fold-specific files.

## Commands

Run one fold:

```powershell
python 02_Code/05_model_training.py --fold 1
```

Run all five folds:

```powershell
python 02_Code/05_model_training.py --all
```

## Important safeguards

- The outer-test fold is not used for training or tuning.
- Each fold has its own model files and split manifest.
- Preprocessing is included in the pipelines.
- Do not compare one fold's inner-CV score with another model's final outer-test score.

## Defence explanation

**Why use outer and inner CV?**  
Inner CV selects model settings. Outer CV estimates performance on data that was not used for model selection.

**Why stratified splitting?**  
It keeps the approximate potable/non-potable class proportion similar across partitions.

**Why include a baseline?**  
It provides a simple reference so that complex models must demonstrate value beyond a naive prediction strategy.

## Next

Run probability calibration for every fold. Do not open or use the outer-test data for calibration.
