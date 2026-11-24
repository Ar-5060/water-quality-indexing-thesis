# Dataset পরিচিতি: Kaggle Water Potability

## ১. Dataset-এর ভূমিকা
এই dataset-এ পানির physicochemical measurement থেকে supplied Potability label (0/1) predict করার exploratory classification কাজ করা হয়েছে।

## ২. ফাইল ও আকার
- ফাইল: `data/water_potability.csv`
- Rows: 3,276
- Columns: 10
- Input features: 9
- Label 0 (Non-potable): 1,998
- Label 1 (Potable): 1,278
- Predictor-এ missing value থাকা row: 1,265
- মোট missing cell: 1,434

## ৩. Feature ও target
| Column | ভূমিকা |
|---|---|
| `ph` | pH measurement |
| `Hardness` | পানির hardness |
| `Solids` | মোট dissolved solids সম্পর্কিত measurement |
| `Chloramines` | Chloramines measurement |
| `Sulfate` | Sulfate measurement |
| `Conductivity` | Electrical conductivity |
| `Organic_carbon` | Organic carbon measurement |
| `Trihalomethanes` | Trihalomethanes measurement |
| `Turbidity` | পানির turbidity |
| `Potability` | target: 0 = Non-potable, 1 = Potable |

## ৪. Missing value
| Column | Missing cell |
|---|---:|
| `ph` | 491 |
| `Sulfate` | 781 |
| `Trihalomethanes` | 162 |

Missing predictor থাকা row স্বয়ংক্রিয়ভাবে বাদ না দিয়ে experiment-এ training-only median imputation ব্যবহার করা হয়েছে। Imputer training data-তে fit হবে; validation/test data থেকে median শেখা যাবে না।

## ৫. Dataset-এর সীমাবদ্ধতা
এই CSV-তে sampling location, collection date, laboratory protocol বা label adjudication details নেই। তাই supplied `Potability` label-এর provenance স্বাধীনভাবে যাচাই করা যায় না। Model prediction-কে real-world drinking-water safety certificate বলা যাবে না।

## ৬. ব্যবহার
এই dataset Potability classification track-এর জন্য। Dhaka WQI dataset-এর সঙ্গে row-level merge বা target substitution করা যাবে না।
