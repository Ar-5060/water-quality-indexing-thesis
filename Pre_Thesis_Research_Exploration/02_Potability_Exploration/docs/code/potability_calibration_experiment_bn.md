# Potability threshold ও calibration experiment — বাংলা কোড ব্যাখ্যা

## ১. কোডটি কেন তৈরি করা হয়েছিল?
Raw এবং sigmoid-calibrated probability-এর ভিত্তিতে বিভিন্ন threshold-এ classification performance এবং validation-selected threshold-এর test performance তুলনা করা।

## ২. কোন Dataset ব্যবহার করে?
`water_potability.csv`; 9 predictors এবং `Potability` target।

## ৩. কোডটি ধাপে ধাপে কী করে
1. Stratified 60/20/20 train-validation-test split তৈরি করে।
2. Median imputation ব্যবহার করে; Logistic Regression ও MLP-তে StandardScaler pipeline-এর মধ্যে রাখে।
3. Logistic Regression, Random Forest ও MLP model fit করে; raw এবং sigmoid-calibrated variant পরীক্ষা করে।
4. Validation set-এ 5% false-safe rate-এর মধ্যে potable recall সর্বোচ্চ করে এমন threshold বেছে নেয়।
5. Validation ও test-এ 0.50, 0.65, 0.80 এবং selected threshold-এর metric গণনা করে।
6. Training+validation অংশে 3-fold × 2-repeat CV summary তৈরি করে।

## ৪. কী Output তৈরি করে?
`potability_threshold_results.csv`, `potability_threshold_selection.csv`, `potability_repeated_cv.csv`।

## ৫. কীভাবে ফলাফল বুঝবে?
এই script-এর output CSV-তে row/column, metric এবং split/rule অনুযায়ী ফলাফল থাকে। Summary file থাকলে সেটি বিভিন্ন split-এর aggregate summary; detailed file-এ split বা record-level ফল থাকে। CSV-এর মানকে এই documentation একা final thesis conclusion হিসেবে ঘোষণা করছে না—ফলাফল ব্যাখ্যার আগে code version, dataset version ও evaluation design মিলিয়ে দেখতে হবে।

## ৬. সীমাবদ্ধতা ও সতর্কতা
- Threshold validation set-এ বাছাই হয়; test set threshold selection-এ ব্যবহার করা হয়নি।
- একটি fixed seed-এর holdout result independent external validation নয়।
- Calibration probability quality উন্নত করবে—এমন নিশ্চয়তা নেই; Brier score-সহ একাধিক metric দেখতে হবে।

## ৭. Thesis defence-এ সহজ ব্যাখ্যা
এই code-টি research exploration-এর একটি নির্দিষ্ট প্রশ্ন পরীক্ষা করার জন্য লেখা হয়েছিল। আমি প্রথমে dataset ও target বুঝেছি, তারপর নির্দিষ্ট split এবং model/method ব্যবহার করেছি, এরপর validation/test বা descriptive audit থেকে metric বের করেছি। ফলাফলকে তার experiment design-এর সীমার মধ্যে ব্যাখ্যা করতে হবে; এটি নিজে থেকে drinking-water safety guarantee বা external validation প্রমাণ করে না।
