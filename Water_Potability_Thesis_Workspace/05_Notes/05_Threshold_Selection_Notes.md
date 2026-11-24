# Lesson 05 — Risk-Aware Threshold Selection

## Purpose

Choose a probability threshold using calibration data to study the trade-off between false-safe errors and identifying genuinely potable samples.

## Definitions

Positive class: `Potability = 1`.

- FP (false-safe): actual non-potable (`0`) predicted potable (`1`).
- TN: actual non-potable (`0`) predicted non-potable (`0`).
- Primary FSR: `FP / (FP + TN)`.
- Potable recall: `TP / (TP + FN)`.

The primary FSR denominator is **all actual non-potable calibration samples**. It is not the same as `FP / (TP + FP)`.

## Inputs and safeguards

Input: `04_Results/tables/calibrated_predictions_fold_k.csv`

Use only `*_sigmoid_oof` probabilities for threshold selection. The outer-test fold is not loaded or used. Row IDs are checked against `split_manifest.json`.

## Fixed settings

| Setting | Value |
|---|---:|
| Main FSR target | 5% |
| Sensitivity targets | 10%, 20%, 30% |
| One-sided confidence level | 95% |
| Ordinary threshold grid | 0.000 to 1.000 |
| Grid step | 0.005 |
| Ordinary candidates | 201 |
| Reject-all candidate | One cutoff just above 1.000 |
| Total candidates | 202 |
| Bonferroni alpha per candidate | 0.05 / 202 |

## Strategies compared

1. `fixed_0.50`: reference threshold.
2. `empirical`: observed calibration FSR is at or below the target.
3. `clopper_pearson`: one-sided exact binomial upper confidence bound is at or below the target.
4. `bonferroni`: Clopper–Pearson bound with alpha adjusted for the 202 candidate thresholds.

The selected eligible threshold aims to retain the greatest potable recall under the stated criterion. The Majority Baseline is reference-only because its constant probability cannot rank samples meaningfully.

## Fold 1 example — main 5% target

| Model | Strategy | Threshold | False-safe count | Calibration FSR | Potable recall |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Empirical | 0.415 | 16 | 5.00% | 6.86% |
| Logistic Regression | Clopper–Pearson | 0.430 | 4 | 1.25% | 1.96% |
| Logistic Regression | Bonferroni | 0.435 | 2 | 0.63% | 1.47% |
| Random Forest | Empirical | 0.585 | 15 | 4.69% | 26.47% |
| Random Forest | Clopper–Pearson | 0.615 | 9 | 2.81% | 20.10% |
| Random Forest | Bonferroni | 0.730 | 3 | 0.94% | 6.37% |
| MLP | Empirical | 0.575 | 14 | 4.38% | 28.92% |
| MLP | Clopper–Pearson | 0.595 | 9 | 2.81% | 26.47% |
| MLP | Bonferroni | 0.795 | 3 | 0.94% | 5.39% |

Majority Baseline rows are marked reference-only. Its zero FSR comes with zero potable recall and must not be presented as a useful model.

## All-fold calibration-side summary

An existing all-fold note records mean results across five calibration folds. For example, Random Forest mean FSR was 4.56% (empirical), 2.75% (Clopper–Pearson) and 0.87% (Bonferroni), while mean potable recall declined from 23.05% to 17.09% and 7.03%, respectively.

These are **calibration-side threshold-selection results**, not final outer-test results.

## Outputs

Per fold:
- `04_Results/tables/threshold_grid_fold_k.csv`
- `04_Results/tables/threshold_selection_fold_k.csv`
- `03_Experiments/fold_k/thresholds.json`

All folds:
- `04_Results/tables/threshold_selection_summary.csv`

## Commands

```powershell
python 02_Code/07_threshold_selection.py --fold 1
python 02_Code/07_threshold_selection.py --all
```

## Limitations

- A 5% calibration-side FSR target does not guarantee a 5% FSR on future or outer-test samples.
- Bonferroni correction here covers the 202 candidates within each model's threshold family; it does not automatically correct across all four model families.
- A low FSR must be reported alongside potable recall and the safe-prediction share.
- The reject-all threshold is a fallback, not proof that a useful classifier met the target.

## Defence explanation

We select a threshold because probability scores need a decision rule. The empirical rule checks observed risk. Clopper–Pearson adds uncertainty around the estimated risk. Bonferroni makes the candidate search more conservative. All three are compared with the fixed 0.50 reference.
