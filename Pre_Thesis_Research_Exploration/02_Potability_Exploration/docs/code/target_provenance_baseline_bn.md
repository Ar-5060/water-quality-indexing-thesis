# Target provenance ও potability baseline — বাংলা কোড ব্যাখ্যা

## ১. কোডটি কেন তৈরি করা হয়েছিল?
দুটি dataset-এর কিছু provenance/integrity check এবং Kaggle Potability dataset-এ Majority, Logistic Regression ও Random Forest baseline চালানো।

## ২. কোন Dataset ব্যবহার করে?
`water_potability.csv` এবং Dhaka WQI CSV—দুটিই script-এ পড়া হয়।

## ৩. কোডটি ধাপে ধাপে কী করে
1. Potability dataset থেকে 80/20 stratified train-test split তৈরি করে।
2. Majority baseline, median imputation + scaling + Logistic Regression, এবং median imputation + Random Forest fit করে।
3. Accuracy, balanced accuracy, potable F1/recall, false-safe count/rate, ROC-AUC ও PR-AUC গণনা করে।
4. Dhaka dataset-এ WQI threshold-label match, COD=70, negative COD পরীক্ষা করে।
5. Potability dataset-এ missing predictor row, missing target, class count এবং metadata অনুপস্থিতি report করে।

## ৪. কী Output তৈরি করে?
`potability_provenance_baseline.csv` এবং `dataset_target_provenance_audit.csv`।

## ৫. কীভাবে ফলাফল বুঝবে?
এই script-এর output CSV-তে row/column, metric এবং split/rule অনুযায়ী ফলাফল থাকে। Summary file থাকলে সেটি বিভিন্ন split-এর aggregate summary; detailed file-এ split বা record-level ফল থাকে। CSV-এর মানকে এই documentation একা final thesis conclusion হিসেবে ঘোষণা করছে না—ফলাফল ব্যাখ্যার আগে code version, dataset version ও evaluation design মিলিয়ে দেখতে হবে।

## ৬. সীমাবদ্ধতা ও সতর্কতা
- এটি একবারের stratified 80/20 baseline; repeated/nested evaluation নয়।
- Script-এর input path `/mnt/data/Water quality.csv` এবং `/mnt/data/water_potability.csv` hard-coded। তোমার WQI dataset filename আলাদা; Windows-এ path ঠিক না করলে চলবে না।
- Label provenance verified নয়; baseline result water safety certification নয়।

## ৭. Thesis defence-এ সহজ ব্যাখ্যা
এই code-টি research exploration-এর একটি নির্দিষ্ট প্রশ্ন পরীক্ষা করার জন্য লেখা হয়েছিল। আমি প্রথমে dataset ও target বুঝেছি, তারপর নির্দিষ্ট split এবং model/method ব্যবহার করেছি, এরপর validation/test বা descriptive audit থেকে metric বের করেছি। ফলাফলকে তার experiment design-এর সীমার মধ্যে ব্যাখ্যা করতে হবে; এটি নিজে থেকে drinking-water safety guarantee বা external validation প্রমাণ করে না।
