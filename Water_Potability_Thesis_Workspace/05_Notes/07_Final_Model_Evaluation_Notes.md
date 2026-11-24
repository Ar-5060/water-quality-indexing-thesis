# Lesson 07 — Final Model Evaluation on Untouched Outer-Test Folds

## Purpose

Measure the performance of the frozen models and decision rules on outer-test records that were not used for fitting, calibration or threshold selection.

## What Lesson 09 does

For each fold, the script loads:
- saved model pipelines;
- saved calibration objects;
- `split_manifest.json`;
- `thresholds.json`;
- `selective_rules.json` and its audit manifest.

It then applies the saved model, sigmoid calibrator, thresholds and selective cut-offs to that fold's untouched outer-test data.

**It does not retrain models, fit calibrators, reselect thresholds or reselect selective cut-offs.**

## Integrity checks

The recorded Fold 1 run reported:
- training/calibration/test overlap check: PASSED;
- Lesson 08 input SHA-256 checks: PASSED;
- Lesson 08 output SHA-256 checks: PASSED;
- selective rules integrity check: PASSED;
- outer-test rows used for threshold selection: 0.

The SHA-256 checks detect changes to the saved inputs/outputs when the corresponding manifest remains unchanged. A hash manifest is an integrity check, not a digital signature.

## Fold 1 example from the recorded terminal output

Outer-test rows: 656  
Actual non-potable: 400  
Actual potable: 256

| Model | Probability method | ROC-AUC | Average Precision | Brier |
|---|---|---:|---:|---:|
| Random Forest | Uncalibrated | 0.6763 | 0.6156 | 0.2135 |
| Random Forest | Sigmoid | 0.6763 | 0.6156 | 0.2124 |
| Random Forest | Isotonic | 0.6559 | 0.5699 | 0.2154 |
| MLP | Uncalibrated | 0.6635 | 0.6036 | 0.2140 |
| MLP | Sigmoid | 0.6635 | 0.6036 | 0.2139 |
| MLP | Isotonic | 0.6605 | 0.5559 | 0.2202 |

These are **Fold 1 only**, not the pooled five-fold result.

Fold 1 threshold example:
- Random Forest fixed 0.50: FSR 14.25%, potable recall 33.98%.
- Random Forest empirical threshold 0.585: FSR 5.00%, potable recall 22.66%.
- Random Forest Bonferroni threshold 0.730: FSR 0%, potable recall 7.03%.

Again, these are Fold 1 test results only. Do not use them as the overall five-fold conclusion.

## All-fold outputs

The all-fold evaluation creates combined predictions and fold/pooled summaries, including:
- `test_predictions_all_folds.csv`
- `evaluation_probability_all_folds.csv`
- `evaluation_thresholds_all_folds.csv`
- `evaluation_selective_all_folds.csv`
- `evaluation_probability_metrics_pooled.csv`
- `evaluation_probability_summary_mean_std.csv`
- `evaluation_threshold_metrics_pooled.csv`
- `evaluation_threshold_summary_mean_std.csv`
- `evaluation_selective_metrics_pooled.csv`
- `evaluation_selective_summary_mean_std.csv`

It also saves per-fold confusion matrices, ROC curves and reliability diagrams.

## Commands

One fold:

```powershell
python 02_Code/09_model_evaluation.py --fold 1
```

All five folds:

```powershell
python 02_Code/09_model_evaluation.py --all
```

## Interpretation rules

- Report pooled metrics and fold mean ± standard deviation separately.
- Pooled results combine each outer-test sample once across its assigned fold.
- Fold mean ± SD describes variation across the five folds.
- Keep the primary FSR denominator explicit: `FP / (FP + TN)`.
- In selective evaluation, the primary FSR denominator includes all actual non-potable samples, including Needs Review cases.
- Do not claim a calibration-side 5% target guarantees 5% test FSR.

## Defence explanation

The outer test is the final evaluation layer. It estimates how the already-selected pipeline performs on data that was not used to fit the model or choose its decision rules.
