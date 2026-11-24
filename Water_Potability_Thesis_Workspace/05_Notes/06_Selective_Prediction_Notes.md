# Lesson 06 — Selective Prediction (Abstention / Needs Review)

## Purpose

A standard classifier must assign every sample to a class. Selective prediction allows the system to abstain when its score is close to the decision boundary and send the sample to **Needs Review**.

This is a separate method from threshold selection.

## Rule used

- Decision threshold: `0.50`
- Confidence score: `abs(sigmoid_oof_probability - 0.50)`
- Higher confidence means farther from the 0.50 boundary.
- Retain the highest-confidence rows for target coverages of 25%, 50%, 75% and 100%.
- If retained probability is at least 0.50: predict potable.
- If below 0.50: predict non-potable.
- If not retained: Needs Review.

Tied confidence values are all retained. Therefore, achieved calibration coverage can be slightly above the requested target.

The Majority Baseline is reference-only because its constant score cannot rank confidence meaningfully.

## Calibration-side selection

Only `*_sigmoid_oof` probabilities are used to select confidence cut-offs. Outer-test data is not used. The cut-offs are saved and later applied unchanged in Lesson 09.

## Fold 1 cut-off examples

| Model | 25% cutoff | 50% cutoff | 75% cutoff | 100% cutoff |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.121660 | 0.109690 | 0.099482 | 0.000000 |
| Random Forest | 0.229467 | 0.151815 | 0.093593 | 0.000000 |
| MLP | 0.243668 | 0.160588 | 0.093418 | 0.000000 |

At 100% coverage, cutoff 0 means every sample is decided and none is sent for review.

## Actual pooled outer-test selective results

The uploaded five-fold pooled selective table contains 3,276 outer-test predictions in total.

| Model | Target coverage | Actual coverage | Review share | Primary FSR | Potable recall (all actual potable) |
|---|---:|---:|---:|---:|---:|
| Random Forest | 25% | 23.26% | 76.74% | 0.65% | 7.12% |
| Random Forest | 50% | 50.12% | 49.88% | 1.65% | 13.62% |
| Random Forest | 75% | 74.02% | 25.98% | 4.45% | 20.58% |
| Random Forest | 100% | 100% | 0% | 11.41% | 33.41% |
| MLP | 25% | 24.54% | 75.46% | 1.15% | 9.62% |
| MLP | 50% | 49.69% | 50.31% | 3.00% | 15.49% |
| MLP | 75% | 73.81% | 26.19% | 4.90% | 21.21% |
| MLP | 100% | 100% | 0% | 11.91% | 33.72% |

**Important Logistic Regression observation:** at 25%, 50% and 75% target coverage, the pooled outer-test output contains zero potable declarations, so potable recall is zero and precision among declared-potable samples is undefined. Its zero FSR is therefore not a useful safety-performance win. At 100% coverage its pooled FSR is 0.25%, but potable recall is only about 0.08%. This must be discussed rather than hidden.

## Interpretation

- Lower coverage means more cases are sent to review.
- FSR can decrease as coverage decreases, but potable recall can also fall.
- A Needs Review sample is neither confirmed potable nor confirmed non-potable.
- A real system would need a defined human/laboratory review process.
- The target coverage is selected on calibration data; actual test coverage can differ.

## Outputs

- `03_Experiments/fold_k/selective_rules.json`
- `03_Experiments/fold_k/selective_manifest.json`
- `04_Results/tables/selective_prediction_fold_k.csv`
- `04_Results/tables/selective_calibration_predictions_fold_k.csv`
- `04_Results/tables/selective_prediction_summary_all_folds.csv`

## Commands

```powershell
python 02_Code/08_selective_prediction.py --fold 1
python 02_Code/08_selective_prediction.py --all
```

## Defence explanation

Selective prediction is not the same as selecting a binary threshold. Threshold selection changes the potable/non-potable cutoff. Selective prediction adds a third outcome—Needs Review—when the model's score is insufficiently far from the decision boundary.
