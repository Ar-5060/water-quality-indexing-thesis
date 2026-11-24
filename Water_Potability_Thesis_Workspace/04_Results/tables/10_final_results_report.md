# Final Results Analysis — Water Potability Thesis

## Evaluation scope

- Raw dataset rows: **3,276**
- Outer-test folds: **5**
- Evaluation basis: untouched outer-test predictions from Lesson 09
- Positive class: Potable (1); negative class: Non-potable (0)
- Primary false-safe rate: `FP / (FP + TN)` (verified from saved counts)
- Target false-safe rate used in Lesson 07: **5%**
- No model, threshold, calibrator, or selective cut-off was modified.
- Nominal 95% intervals: stratified sample bootstrap (ROC-AUC, Brier; 2000 resamples, seed 42) and exact Clopper–Pearson (FSR, recall).
- These intervals are descriptive summaries conditional on the pooled out-of-fold predictions. They do not capture the full variation from retraining across folds and are not formal paired model-comparison tests.

### Outer-test rows by fold

| fold | outer_test_rows |
| --- | --- |
| 1 | 656 |
| 2 | 655 |
| 3 | 655 |
| 4 | 655 |
| 5 | 655 |

## 1. Probability model comparison (sigmoid)

| model | roc_auc_pooled_ci | roc_auc_fold_mean_sd | average_precision | average_precision_fold_mean_sd | brier_pooled_ci | brier_fold_mean_sd | log_loss | log_loss_fold_mean_sd | ece | ece_fold_mean_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | 0.500 [0.481, 0.520] | 0.500 ± 0.000 | 0.3904 | 0.390 ± 0.001 | 0.238 [0.238, 0.238] | 0.238 ± 0.000 | 0.6688 | 0.669 ± 0.000 | 0.0001 | 0.001 ± 0.000 |
| logistic_regression | 0.485 [0.465, 0.505] | 0.487 ± 0.013 | 0.3868 | 0.394 ± 0.014 | 0.239 [0.238, 0.240] | 0.239 ± 0.001 | 0.671 | 0.671 ± 0.002 | 0.0155 | 0.015 ± 0.019 |
| random_forest | 0.685 [0.666, 0.703] | 0.687 ± 0.016 | 0.6041 | 0.612 ± 0.024 | 0.211 [0.207, 0.216] | 0.211 ± 0.004 | 0.6112 | 0.611 ± 0.009 | 0.0244 | 0.042 ± 0.009 |
| mlp | 0.684 [0.666, 0.703] | 0.685 ± 0.022 | 0.5988 | 0.604 ± 0.030 | 0.212 [0.207, 0.217] | 0.212 ± 0.005 | 0.6116 | 0.612 ± 0.012 | 0.0169 | 0.041 ± 0.012 |

- Random forest ROC-AUC 0.685 [0.666, 0.703] vs MLP 0.684 [0.666, 0.703]: their descriptive 95% intervals overlap.
- Interval overlap is only a visual descriptive comparison; it is not a formal paired test of model differences.
- Models whose ROC-AUC interval includes 0.5 (no clear evidence of ranking ability): Logistic regression.

## 2. Threshold-selection results

| model | strategy | mean_applied_threshold | threshold_fold_mean | threshold_fold_std | n_declared_potable | fsr_pooled_ci | fsr_fold_mean_sd | folds_meeting_target_fsr | potable_recall | positive_precision | f1_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority_baseline | fixed_0.50 | 0.5 | 0.5 | 0.0 | 0 | 0.0% [0.0, 0.2] | 0.0 ± 0.0 | 5 | 0.0 | — | 0.0 |
| majority_baseline | empirical | 0.5 | 0.5 | 0.0 | 0 | 0.0% [0.0, 0.2] | 0.0 ± 0.0 | 5 | 0.0 | — | 0.0 |
| majority_baseline | clopper_pearson | 0.5 | 0.5 | 0.0 | 0 | 0.0% [0.0, 0.2] | 0.0 ± 0.0 | 5 | 0.0 | — | 0.0 |
| majority_baseline | bonferroni | 0.5 | 0.5 | 0.0 | 0 | 0.0% [0.0, 0.2] | 0.0 ± 0.0 | 5 | 0.0 | — | 0.0 |
| logistic_regression | fixed_0.50 | 0.5 | 0.5 | 0.0 | 6 | 0.3% [0.1, 0.6] | 0.3 ± 0.6 | 5 | 0.0008 | 0.1667 | 0.0016 |
| logistic_regression | empirical | 0.427 | 0.427 | 0.0152 | 127 | 3.7% [2.9, 4.6] | 3.7 ± 2.9 | 3 | 0.0415 | 0.4173 | 0.0754 |
| logistic_regression | clopper_pearson | 0.436 | 0.436 | 0.0139 | 48 | 1.5% [1.0, 2.1] | 1.5 ± 2.2 | 4 | 0.0149 | 0.3958 | 0.0287 |
| logistic_regression | bonferroni | 0.45 | 0.45 | 0.0232 | 11 | 0.4% [0.1, 0.7] | 0.4 ± 0.5 | 5 | 0.0031 | 0.3636 | 0.0062 |
| random_forest | fixed_0.50 | 0.5 | 0.5 | 0.0 | 655 | 11.4% [10.0, 12.9] | 11.4 ± 4.3 | 1 | 0.3341 | 0.6519 | 0.4418 |
| random_forest | empirical | 0.58 | 0.58 | 0.0106 | 354 | 4.6% [3.7, 5.6] | 4.6 ± 1.6 | 3 | 0.2058 | 0.7429 | 0.3223 |
| random_forest | clopper_pearson | 0.624 | 0.624 | 0.0204 | 254 | 2.6% [1.9, 3.4] | 2.6 ± 1.0 | 5 | 0.1581 | 0.7953 | 0.2637 |
| random_forest | bonferroni | 0.743 | 0.743 | 0.0645 | 89 | 0.5% [0.2, 0.9] | 0.5 ± 0.6 | 5 | 0.0618 | 0.8876 | 0.1156 |
| mlp | fixed_0.50 | 0.5 | 0.5 | 0.0 | 669 | 11.9% [10.5, 13.4] | 11.9 ± 3.2 | 0 | 0.3372 | 0.6442 | 0.4427 |
| mlp | empirical | 0.589 | 0.589 | 0.0119 | 372 | 5.0% [4.1, 6.1] | 5.0 ± 1.8 | 3 | 0.2128 | 0.7312 | 0.3297 |
| mlp | clopper_pearson | 0.651 | 0.651 | 0.0471 | 273 | 3.4% [2.6, 4.2] | 3.4 ± 1.8 | 4 | 0.1612 | 0.7546 | 0.2656 |
| mlp | bonferroni | 0.785 | 0.785 | 0.0355 | 104 | 0.7% [0.4, 1.2] | 0.7 ± 0.8 | 5 | 0.0704 | 0.8654 | 0.1302 |

`folds_meeting_target_fsr` counts how many of the five outer-test folds kept FSR at or below the target. A pooled FSR below the target does not guarantee every fold met it.

## 3. Selective-prediction results

| model | target_coverage | actual_coverage | review_share | n_declared_safe | fsr_pooled_ci | folds_meeting_target_fsr | potable_recall_all | accepted_error_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.25 | 0.2811 | 0.7189 | 0 | 0.0% [0.0, 0.2] | 5 | 0.0 | 0.4137 |
| logistic_regression | 0.5 | 0.4902 | 0.5098 | 0 | 0.0% [0.0, 0.2] | 5 | 0.0 | 0.3985 |
| logistic_regression | 0.75 | 0.7057 | 0.2943 | 0 | 0.0% [0.0, 0.2] | 5 | 0.0 | 0.3927 |
| logistic_regression | 1.0 | 1.0 | 0.0 | 6 | 0.3% [0.1, 0.6] | 5 | 0.0008 | 0.3913 |
| random_forest | 0.25 | 0.2326 | 0.7674 | 104 | 0.7% [0.3, 1.1] | 5 | 0.0712 | 0.1995 |
| random_forest | 0.5 | 0.5012 | 0.4988 | 207 | 1.7% [1.1, 2.3] | 5 | 0.1362 | 0.2479 |
| random_forest | 0.75 | 0.7402 | 0.2598 | 352 | 4.5% [3.6, 5.5] | 3 | 0.2058 | 0.2812 |
| random_forest | 1.0 | 1.0 | 0.0 | 655 | 11.4% [10.0, 12.9] | 1 | 0.3341 | 0.3294 |
| mlp | 0.25 | 0.2454 | 0.7546 | 146 | 1.2% [0.7, 1.7] | 5 | 0.0962 | 0.1953 |
| mlp | 0.5 | 0.4969 | 0.5031 | 258 | 3.0% [2.3, 3.8] | 5 | 0.1549 | 0.2531 |
| mlp | 0.75 | 0.7381 | 0.2619 | 369 | 4.9% [4.0, 5.9] | 3 | 0.2121 | 0.2883 |
| mlp | 1.0 | 1.0 | 0.0 | 669 | 11.9% [10.5, 13.4] | 0 | 0.3372 | 0.3312 |

Target coverage is chosen on calibration data; actual coverage is measured on the untouched outer test, so they differ slightly. Recall counts rows sent to Needs Review as missed, and review workload must be reported alongside error. At 100% coverage nothing is abstained.

## 4. Operating points meeting the 5% target

Learned-model operating points with pooled FSR ≤ 5%, ranking ability better than chance and at least one potable declaration. Listed in fixed model order, not ranked. `fsr_upper_ci_within_target` separately shows whether the descriptive 95% interval's upper limit is also at or below 5%; this is stricter than the pooled point-estimate target.

| model | approach | setting | decision_coverage | fsr | fsr_ci_upper | fsr_upper_ci_within_target | recall_all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | threshold | Empirical | 1.0 | 0.0455 | 0.0556 | False | 0.2058 |
| random_forest | threshold | Clopper–Pearson | 1.0 | 0.026 | 0.034 | True | 0.1581 |
| random_forest | threshold | Bonferroni | 1.0 | 0.005 | 0.0092 | True | 0.0618 |
| random_forest | selective | Target coverage 25% | 0.2326 | 0.0065 | 0.0111 | True | 0.0712 |
| random_forest | selective | Target coverage 50% | 0.5012 | 0.0165 | 0.0231 | True | 0.1362 |
| random_forest | selective | Target coverage 75% | 0.7402 | 0.0445 | 0.0545 | False | 0.2058 |
| mlp | threshold | Clopper–Pearson | 1.0 | 0.0335 | 0.0424 | True | 0.1612 |
| mlp | threshold | Bonferroni | 1.0 | 0.007 | 0.0117 | True | 0.0704 |
| mlp | selective | Target coverage 25% | 0.2454 | 0.0115 | 0.0172 | True | 0.0962 |
| mlp | selective | Target coverage 50% | 0.4969 | 0.03 | 0.0385 | True | 0.1549 |
| mlp | selective | Target coverage 75% | 0.7381 | 0.049 | 0.0595 | False | 0.2121 |

## 5. Metric-wise leaders

Best observed value for each metric separately. This is not a composite ranking or a declaration of one universally best model. `separation` summarises descriptive interval overlap; it is not a formal paired test.

| comparison | metric | direction | leader_model | strategy | target_coverage | value | runner_up | separation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pooled sigmoid ROC-AUC | roc_auc | Higher is better | random_forest |  | — | 0.6849 | MLP (0.6838) | 95% intervals overlap (descriptive only) |
| Pooled sigmoid Average Precision | average_precision | Higher is better | random_forest |  | — | 0.6041 | MLP (0.5988) | No interval available |
| Pooled sigmoid Brier score | brier_score | Lower is better | random_forest |  | — | 0.2114 | MLP (0.2117) | 95% intervals overlap (descriptive only) |
| Pooled sigmoid log loss | log_loss | Lower is better | random_forest |  | — | 0.6112 | MLP (0.6116) | No interval available |
| Pooled sigmoid ECE | ece | Lower is better | mlp |  | — | 0.0169 | Random forest (0.0244) | No interval available |
| Threshold FSR — Fixed 0.50 | false_safe_rate | Lower is better | random_forest | fixed_0.50 | — | 0.1141 | MLP (0.1191) | 95% intervals overlap (descriptive only) |
| Threshold potable recall — Fixed 0.50 | potable_recall | Higher is better | mlp | fixed_0.50 | — | 0.3372 | Random forest (0.3341) | 95% intervals overlap (descriptive only) |
| Threshold FSR — Empirical | false_safe_rate | Lower is better | random_forest | empirical | — | 0.0455 | MLP (0.0501) | 95% intervals overlap (descriptive only) |
| Threshold potable recall — Empirical | potable_recall | Higher is better | mlp | empirical | — | 0.2128 | Random forest (0.2058) | 95% intervals overlap (descriptive only) |
| Threshold FSR — Clopper–Pearson | false_safe_rate | Lower is better | random_forest | clopper_pearson | — | 0.026 | MLP (0.0335) | 95% intervals overlap (descriptive only) |
| Threshold potable recall — Clopper–Pearson | potable_recall | Higher is better | mlp | clopper_pearson | — | 0.1612 | Random forest (0.1581) | 95% intervals overlap (descriptive only) |
| Threshold FSR — Bonferroni | false_safe_rate | Lower is better | random_forest | bonferroni | — | 0.005 | MLP (0.0070) | 95% intervals overlap (descriptive only) |
| Threshold potable recall — Bonferroni | potable_recall | Higher is better | mlp | bonferroni | — | 0.0704 | Random forest (0.0618) | 95% intervals overlap (descriptive only) |
| Selective FSR — 25% target coverage | false_safe_rate_all_non_potable | Lower is better | random_forest |  | 0.25 | 0.0065 | MLP (0.0115) | 95% intervals overlap (descriptive only) |
| Selective accepted error — 25% target coverage | accepted_error_rate | Lower is better | mlp |  | 0.25 | 0.1953 | Random forest (0.1995) | No interval available |
| Selective potable recall — 25% target coverage | potable_recall_all | Higher is better | mlp |  | 0.25 | 0.0962 | Random forest (0.0712) | 95% intervals overlap (descriptive only) |
| Selective FSR — 50% target coverage | false_safe_rate_all_non_potable | Lower is better | random_forest |  | 0.5 | 0.0165 | MLP (0.0300) | 95% intervals overlap (descriptive only) |
| Selective accepted error — 50% target coverage | accepted_error_rate | Lower is better | random_forest |  | 0.5 | 0.2479 | MLP (0.2531) | No interval available |
| Selective potable recall — 50% target coverage | potable_recall_all | Higher is better | mlp |  | 0.5 | 0.1549 | Random forest (0.1362) | 95% intervals overlap (descriptive only) |
| Selective FSR — 75% target coverage | false_safe_rate_all_non_potable | Lower is better | random_forest |  | 0.75 | 0.0445 | MLP (0.0490) | 95% intervals overlap (descriptive only) |
| Selective accepted error — 75% target coverage | accepted_error_rate | Lower is better | random_forest |  | 0.75 | 0.2812 | MLP (0.2883) | No interval available |
| Selective potable recall — 75% target coverage | potable_recall_all | Higher is better | mlp |  | 0.75 | 0.2121 | Random forest (0.2058) | 95% intervals overlap (descriptive only) |
| Selective FSR — 100% target coverage | false_safe_rate_all_non_potable | Lower is better | random_forest |  | 1.0 | 0.1141 | MLP (0.1191) | 95% intervals overlap (descriptive only) |
| Selective accepted error — 100% target coverage | accepted_error_rate | Lower is better | random_forest |  | 1.0 | 0.3294 | MLP (0.3312) | No interval available |
| Selective potable recall — 100% target coverage | potable_recall_all | Higher is better | mlp |  | 1.0 | 0.3372 | Random forest (0.3341) | 95% intervals overlap (descriptive only) |

## 6. Methodological interpretation and limitations

1. Pooled metrics and fold mean ± SD answer different questions: pooled metrics combine all outer-test predictions; mean ± SD describes variation across the five fold-level estimates.
2. The five outer-test folds together cover the raw dataset exactly once.
3. Thresholds and selective cut-offs were not selected using outer-test labels.
4. ECE is descriptive and depends on the binning scheme used in Lesson 09.
5. The bootstrap and Clopper–Pearson intervals are nominal descriptive intervals for pooled out-of-fold predictions. They do not model the dependence induced by overlapping outer-fold training sets or include full model-refitting uncertainty; interval overlap is not a formal paired test.
6. A low false-safe rate alone does not establish a useful classifier. A model that cannot rank better than chance, or declares (almost) nothing potable, reaches a low FSR trivially. Such models are excluded from the FSR leader rows and flagged in every table.
7. `unsafe_share_among_declared_potable` = FP / (TP + FP) is a different denominator and is reported only as a supplementary quantity.
8. These are model-based estimates on one public dataset and split design. They do not certify real-world water safety or guarantee future risk.
9. Any final model recommendation should follow the thesis objective and pre-specified safety/utility priorities, not a post-hoc composite score.
