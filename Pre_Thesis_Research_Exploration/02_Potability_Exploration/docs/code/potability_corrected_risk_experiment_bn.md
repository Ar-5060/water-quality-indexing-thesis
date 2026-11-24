# Bonferroni-corrected risk experiment — বাংলা কোড ব্যাখ্যা

## ১. কোডটি কেন তৈরি করা হয়েছিল?
Predeclared 202-candidate threshold grid-এ one-sided Clopper–Pearson upper bound এবং Bonferroni correction ব্যবহার করে risk/recall trade-off পরীক্ষা করা।

## ২. কোন Dataset ব্যবহার করে?
Script-এর পাশে থাকা `water_potability.csv`।

## ৩. কোডটি ধাপে ধাপে কী করে
1. আটটি seed-এ stratified 60/20/20 split তৈরি করে।
2. Random Forest ও MLP model fit করে।
3. Fixed 0.50, empirical 5%, uncorrected upper-bound 5% এবং Bonferroni 5% rule-এর threshold নির্ধারণ করে।
4. প্রতিটি rule-এর validation ও test false-safe count/rate, potable recall, ROC-AUC, PR-AUC ও Brier score হিসাব করে।
5. প্রতি model/rule-এর mean threshold, false-safe, recall ও 5% exceedance count summary করে।

## ৪. কী Output তৈরি করে?
`potability_corrected_risk_detailed.csv` এবং `potability_corrected_risk_summary.csv`।

## ৫. কীভাবে ফলাফল বুঝবে?
এই script-এর output CSV-তে row/column, metric এবং split/rule অনুযায়ী ফলাফল থাকে। Summary file থাকলে সেটি বিভিন্ন split-এর aggregate summary; detailed file-এ split বা record-level ফল থাকে। CSV-এর মানকে এই documentation একা final thesis conclusion হিসেবে ঘোষণা করছে না—ফলাফল ব্যাখ্যার আগে code version, dataset version ও evaluation design মিলিয়ে দেখতে হবে।

## ৬. সীমাবদ্ধতা ও সতর্কতা
- Bonferroni correction threshold family-এর multiple comparison সামলানোর জন্য; এটি dataset label quality বা distribution shift সমাধান করে না।
- প্রয়োজনীয় independence/exchangeability assumptions পূরণ না হলে formal coverage claim করা যাবে না।
- এই historical repeated-split analysis external validation নয়।

## ৭. Thesis defence-এ সহজ ব্যাখ্যা
এই code-টি research exploration-এর একটি নির্দিষ্ট প্রশ্ন পরীক্ষা করার জন্য লেখা হয়েছিল। আমি প্রথমে dataset ও target বুঝেছি, তারপর নির্দিষ্ট split এবং model/method ব্যবহার করেছি, এরপর validation/test বা descriptive audit থেকে metric বের করেছি। ফলাফলকে তার experiment design-এর সীমার মধ্যে ব্যাখ্যা করতে হবে; এটি নিজে থেকে drinking-water safety guarantee বা external validation প্রমাণ করে না।
