# Clopper–Pearson risk-bound experiment — বাংলা কোড ব্যাখ্যা

## ১. কোডটি কেন তৈরি করা হয়েছিল?
False-safe probability-এর একপাশের 95% Clopper–Pearson upper confidence bound ব্যবহার করে validation threshold বেছে নেওয়ার exploratory পরীক্ষা।

## ২. কোন Dataset ব্যবহার করে?
Script-এর পাশে থাকা `water_potability.csv`।

## ৩. কোডটি ধাপে ধাপে কী করে
1. আটটি seed-এ 60/20/20 stratified split তৈরি করে।
2. Random Forest ও MLP fit করে; median imputation, MLP-তে scaling ব্যবহার করে।
3. Validation data-তে fixed, empirical 5% এবং upper-bound 5% threshold নির্বাচন করে।
4. প্রতিটি threshold-এ validation ও held-out test false-safe count/rate, potable recall, upper bound, ROC-AUC, PR-AUC ও Brier score গণনা করে।

## ৪. কী Output তৈরি করে?
`potability_risk_bound_detailed.csv` এবং `potability_risk_bound_summary.csv`।

## ৫. কীভাবে ফলাফল বুঝবে?
এই script-এর output CSV-তে row/column, metric এবং split/rule অনুযায়ী ফলাফল থাকে। Summary file থাকলে সেটি বিভিন্ন split-এর aggregate summary; detailed file-এ split বা record-level ফল থাকে। CSV-এর মানকে এই documentation একা final thesis conclusion হিসেবে ঘোষণা করছে না—ফলাফল ব্যাখ্যার আগে code version, dataset version ও evaluation design মিলিয়ে দেখতে হবে।

## ৬. সীমাবদ্ধতা ও সতর্কতা
- একই validation data-তে অনেক threshold পরীক্ষা করে সেখান থেকে বাছাই করলে uncorrected bound-এর nominal coverage নষ্ট হতে পারে; script docstring-ও এই সতর্কতা দিয়েছে।
- এটি exploratory threshold-search; 95% drinking-water safety guarantee নয়।
- Repeated test sets overlap করতে পারে।

## ৭. Thesis defence-এ সহজ ব্যাখ্যা
এই code-টি research exploration-এর একটি নির্দিষ্ট প্রশ্ন পরীক্ষা করার জন্য লেখা হয়েছিল। আমি প্রথমে dataset ও target বুঝেছি, তারপর নির্দিষ্ট split এবং model/method ব্যবহার করেছি, এরপর validation/test বা descriptive audit থেকে metric বের করেছি। ফলাফলকে তার experiment design-এর সীমার মধ্যে ব্যাখ্যা করতে হবে; এটি নিজে থেকে drinking-water safety guarantee বা external validation প্রমাণ করে না।
