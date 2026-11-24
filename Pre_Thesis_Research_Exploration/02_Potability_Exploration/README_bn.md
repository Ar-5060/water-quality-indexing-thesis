# Water Potability Exploration

## এই research track কেন শুরু হয়েছিল?
Dhaka WQI direction-এর target provenance অনিশ্চিত থাকায় Kaggle Water Potability dataset নিয়ে binary classification exploration শুরু হয়। এখানে label 0-কে Non-potable এবং label 1-কে Potable হিসেবে ধরা হয়েছে।

## Dataset
`data/water_potability.csv`

- Rows: 3,276
- Input features: 9
- Non-potable (0): 1,998
- Potable (1): 1,278
- Missing predictor থাকা row: 1,265
- Missing cells by column: ph=491, Sulfate=781, Trihalomethanes=162

## Code map

| Script | উদ্দেশ্য |
|---|---|
| `target_provenance_baseline.py` | Dataset audit এবং Majority, Logistic Regression ও Random Forest baseline |
| `potability_calibration_experiment.py` | Fixed threshold, validation threshold, sigmoid calibration ও repeated CV |
| `potability_repeated_threshold_validation.py` | 8 seeded 60/20/20 split-এ validation-selected 5% false-safe threshold পরীক্ষা |
| `potability_risk_bound_experiment.py` | Clopper–Pearson upper bound ভিত্তিক exploratory threshold selection |
| `potability_corrected_risk_experiment.py` | 202 threshold candidate grid-এ Bonferroni-adjusted upper-bound experiment |
| `potability_risk_coverage_analysis.py` | Corrected-risk result থেকে positive prediction rate, precision ও coverage summary |
| `potability_selective_prediction.py` | Uncertain prediction-কে `Needs Review` পাঠানোর selective prediction পরীক্ষা |

## Results
`results/` folder-এ baseline, threshold, repeated CV, risk-bound, corrected-risk, risk-coverage এবং selective prediction-এর detailed ও summary CSV রাখা হয়েছে।

## গুরুত্বপূর্ণ সতর্কতা
- এগুলো আগের exploratory experiment-এর ফল; formal final thesis evaluation-এর বিকল্প নয়।
- Repeated split-গুলোর test observation overlap করতে পারে; তাই repeat-গুলো independent external validation নয়।
- Validation-এ threshold বাছাইয়ের পর test-এ ফল আলাদা হতে পারে।
- False-safe কমাতে গিয়ে potable recall, decision coverage বা review workload খারাপ হতে পারে।
- Dataset label-এর laboratory provenance যাচাই করা নেই; model water safety guarantee করে না।
