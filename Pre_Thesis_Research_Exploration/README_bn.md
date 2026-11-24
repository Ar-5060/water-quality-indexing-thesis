# প্রি-থিসিস রিসার্চ এক্সপ্লোরেশন

এই ফোল্ডারে থিসিসের formal methodology চূড়ান্ত হওয়ার আগের গবেষণা-অনুসন্ধান, Python experiment, ব্যবহৃত dataset এবং ঐ সময়ের সংরক্ষিত CSV output রাখা হয়েছে।

## কেন এই ফোল্ডার রাখা হয়েছে
- আগের research work ও experiment history সংরক্ষণ করা।
- কোন code কেন লেখা হয়েছিল তা বোঝা।
- প্রতিটি script-এর input dataset ও output result খুঁজে পাওয়া।
- WQI direction থেকে Potability classification direction-এ যাওয়ার গবেষণা-ইতিহাস ধরে রাখা।
- exploratory result-কে formal thesis-এর confirmatory result হিসেবে ভুলভাবে উপস্থাপন না করা।

## Folder map

| Folder | কী আছে |
|---|---|
| `01_Dhaka_WQI_Exploration/` | Dhaka dataset-এর WQI regression, phosphate dependence ও target integrity পরীক্ষা |
| `02_Potability_Exploration/` | Kaggle potability baseline, threshold, calibration, risk ও selective prediction experiment |
| `03_Research_Notes/` | `journal_recommendations.xlsx` এবং সংশ্লিষ্ট research note |

## Dataset-এর সংক্ষিপ্ত পরিচয়

### Dhaka Water Quality
- ফাইল: `01_Dhaka_WQI_Exploration/data/Water quality dhaka.csv`
- আকার: 1,495 rows × 16 columns
- তারিখের পরিসর: 2017-01-25 থেকে 2021-07-17
- Location: 58টি; River/Stream: 13টি
- Missing cell: 0
- WQI-এর 50/75 threshold দিয়ে তৈরি category-এর সঙ্গে supplied `Pollution_Label` মিলেছে: 1495/1495
- COD = 70: 1301টি; Negative COD: 6টি

**সতর্কতা:** category match হওয়া মানেই dataset publisher-এর WQI formula, weight বা standard যাচাই হয়ে গেছে—এমন নয়। WQI formula ও provenance এখনো আলাদাভাবে যাচাই করতে হবে।

### Kaggle Water Potability
- ফাইল: `02_Potability_Exploration/data/water_potability.csv`
- আকার: 3,276 rows × 10 columns
- Input feature: 9
- Label 0 (Non-potable): 1,998
- Label 1 (Potable): 1,278
- Predictor-এ missing value থাকা row: 1,265
- মোট missing cell: 1,434

**সতর্কতা:** CSV-তে sampling location, collection date বা laboratory labeling protocol নেই। তাই prediction-কে drinking-water safety certification বলা যাবে না।

## Python code ও result
- WQI: 3টি Python script এবং 4টি CSV result।
- Potability: 7টি Python script এবং 17টি CSV result।
- মোট: 10টি Python script, 21টি CSV result।


## গুরুত্বপূর্ণ সীমাবদ্ধতা ও যাচাইয়ের নোট
1. এই folder-এর কাজগুলোকে **Pre-Thesis Exploratory Work** হিসেবে চিহ্নিত করতে হবে।
2. CSV result archive-এ থাকলেও প্রতিটি result-এর source script, run environment ও execution date আলাদাভাবে নথিবদ্ধ করা হয়নি।
3. `potability_unified_evaluation_detailed.csv` ও `potability_unified_evaluation_summary.csv` আছে, কিন্তু supplied ZIP-এ এগুলো তৈরির নির্দিষ্ট Python script পাওয়া যায়নি।
4. `wqi_baseline_audit.py` একটি `wqi_audit_summary.json` তৈরি করার কথা বলে; সেই JSON supplied ZIP-এ নেই।
5. কয়েকটি পুরোনো script `/mnt/data/` অথবা নির্দিষ্ট `Water quality.csv` path ব্যবহার করে। নতুন folder structure-এ সরাসরি চালানোর আগে path ঠিক করতে হবে। Original script এখানে অপরিবর্তিত রাখা হয়েছে।
6. এই folder-এর code-কে পরবর্তী formal thesis pipeline-এর সঙ্গে মিশিয়ে ফেলবে না। Formal pipeline আলাদা থাকবে।

