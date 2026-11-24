# Lesson 02 — Data Preprocessing Notes

## Purpose

Prepare the features for model training while preventing information from the validation or test data from leaking into the training process.

## Main decisions

1. Keep the original raw dataset unchanged.
2. Convert physically invalid pH values (`pH <= 0` or `pH > 14`) to missing values.
3. Use median imputation, with the median learned from the relevant training partition only.
4. Scale features for Logistic Regression and MLP.
5. Do not scale features for Random Forest or the Majority Baseline.
6. Keep preprocessing inside each model pipeline so each CV training split learns its own imputation/scaling parameters.

## Shared helper: `water_utils.py`

The helper function `fix_invalid_ph(X)`:
- makes a copy of the input;
- sets pH values less than or equal to 0, or greater than 14, to `NaN`;
- does not learn any statistic from the data;
- returns the corrected feature table.

It is imported by `05_model_training.py`. Running `water_utils.py` directly may produce no output; it is a helper module, not a standalone analysis script.

## Preprocessing pipelines used for model training

**Scaled pipeline — Logistic Regression and MLP**
1. Invalid pH correction
2. Median imputation
3. Standard scaling

**Unscaled pipeline — Random Forest and Majority Baseline**
1. Invalid pH correction
2. Median imputation

All learned transformations are fitted only within the relevant training data during cross-validation.

## Separate preprocessing setup check

`04_data_preprocessing.py` was run as a setup/validation stage. Its recorded output showed:
- 5-fold outer-CV setup;
- training shape `(2620, 9)` and test shape `(656, 9)` for the first outer fold;
- scaled and tree-style preprocessing checks;
- no remaining NaN or infinite values in the tested transformed data;
- only the first outer fold was tested in that run.

**Important distinction:** An earlier preprocessing prototype included IQR capping and saved separate clean/scaled CSVs. Do not state that IQR capping was part of the final nested model-training pipeline unless the final `05_model_training.py` actually applies it. The documented final model pipelines above use pH correction, median imputation and model-appropriate scaling.

## Why split-aware preprocessing matters

If the median or scaler is calculated using test data, information from the test set enters the training process. This is data leakage and can make evaluation results overly optimistic.

## Defence explanation

**Why median imputation?**  
It fills missing numeric measurements using the median learned from training data. The median is less sensitive to extreme values than the mean.

**Why scaling for Logistic Regression and MLP?**  
These models can be sensitive to feature scales. StandardScaler transforms features to approximately zero mean and unit standard deviation.

**Why not scale Random Forest?**  
Tree-based models split features by thresholds and generally do not require standard scaling.

## Run

```powershell
python 02_Code/04_data_preprocessing.py
```

The main training pipeline is run in the next lesson:

```powershell
python 02_Code/05_model_training.py --all
```

## Preserve

- `02_Code/water_utils.py`
- `02_Code/04_data_preprocessing.py`
- `01_Data/raw/water_potability.csv`
- any generated preprocessing reports and configuration files
