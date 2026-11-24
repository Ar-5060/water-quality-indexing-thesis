# Dhaka WQI Exploration — বাংলা পরিচিতি

## এই research track কেন শুরু হয়েছিল?
প্রাথমিকভাবে Dhaka-এর water-quality measurement থেকে numeric Water Quality Index (WQI) predict করা সম্ভব কি না তা পরীক্ষা করা হয়েছিল। একই সঙ্গে `Pollution_Label` কীভাবে WQI-এর সঙ্গে সম্পর্কিত, phosphate feature কতটা প্রভাব ফেলে এবং dataset-এর target integrity কেমন—এসব প্রশ্ন দেখা হয়েছিল।

## ব্যবহৃত dataset
`data/Water quality dhaka.csv`

- Rows: 1,495
- Columns: 16
- Numeric WQI target: `WQI`
- Category label: `Pollution_Label`
- Predictor-এর মধ্যে temperature, pH, DO, COD, nitrate, turbidity, phosphate ও rainfall রয়েছে।

## Code map

| Script | উদ্দেশ্য |
|---|---|
| `wqi_baseline_audit.py` | Random, temporal ও unseen-station split-এ WQI regression baseline তুলনা |
| `wqi_phosphate_validation.py` | সব feature, phosphate-only, phosphate ছাড়া এবং dummy baseline তুলনা |
| `wqi_target_integrity_check.py` | WQI threshold-label match, COD anomaly, phosphate-WQI সম্পর্ক ও target summary পরীক্ষা |

## Results map
- `wqi_baseline_results.csv`
- `wqi_phosphate_ablation_results.csv`
- `wqi_permutation_importance.csv`
- `wqi_target_integrity_report.csv`

## কেন এই direction থামানো হয়েছিল?
আগের audit-এ WQI-এর formula, parameter weight ও standard-এর নির্ভরযোগ্য provenance নিশ্চিত করা যায়নি। `Pollution_Label` observed WQI threshold দিয়ে পুনর্গঠন করা যায়, তাই এটি independent biological বা potability outcome নয়। COD-এর অস্বাভাবিক মান এবং phosphate feature-এর dependence-ও target integrity নিয়ে প্রশ্ন তৈরি করে।

## সিদ্ধান্ত
এই WQI track-কে **Paused / Exploratory** হিসেবে সংরক্ষণ করা হবে। এটি primary potability thesis-এর সঙ্গে এক dataset হিসেবে মেশানো যাবে না। Original code ও results মুছে ফেলা যাবে না।
