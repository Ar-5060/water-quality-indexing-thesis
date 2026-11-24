# Selective prediction ও Needs Review experiment — বাংলা কোড ব্যাখ্যা

## ১. কোডটি কেন তৈরি করা হয়েছিল?
Model-এর probability 0.50-এর কাছাকাছি হলে জোর করে 0/1 না দিয়ে `Needs Review` পাঠালে coverage, accepted error ও false-safe কীভাবে বদলায় তা পরীক্ষা করা।

## ২. কোন Dataset ব্যবহার করে?
Script-এর পাশে থাকা `water_potability.csv`।

## ৩. কোডটি ধাপে ধাপে কী করে
1. আটটি seed-এ stratified train/validation/test split তৈরি করে।
2. Random Forest ও MLP fit করে।
3. Validation probability-এর 0.50 থেকে দূরত্বকে confidence proxy হিসেবে নেয়।
4. 25%, 50%, 75% ও 100% intended coverage-এর জন্য validation cutoff বেছে নেয়।
5. Test set-এ accepted sample, Needs Review count, coverage, accepted error, false-safe rate, potable recall, precision, ROC-AUC ও Brier score হিসাব করে।

## ৪. কী Output তৈরি করে?
`potability_selective_prediction_detailed.csv` এবং `potability_selective_prediction_summary.csv`।

## ৫. কীভাবে ফলাফল বুঝবে?
এই script-এর output CSV-তে row/column, metric এবং split/rule অনুযায়ী ফলাফল থাকে। Summary file থাকলে সেটি বিভিন্ন split-এর aggregate summary; detailed file-এ split বা record-level ফল থাকে। CSV-এর মানকে এই documentation একা final thesis conclusion হিসেবে ঘোষণা করছে না—ফলাফল ব্যাখ্যার আগে code version, dataset version ও evaluation design মিলিয়ে দেখতে হবে।

## ৬. সীমাবদ্ধতা ও সতর্কতা
- 0.50 থেকে দূরত্ব calibrated confidence-এর সমান নয়; এটি কেবল score-based proxy।
- Validation থেকে নির্ধারিত cutoff test-এ একই coverage নিশ্চিত করে না।
- Needs Review বাস্তবে কার্যকর করতে laboratory বা human review workflow দরকার।
- Review-এ পাঠানো sample-কে potable বা non-potable ধরে নেওয়া যাবে না।

## ৭. Thesis defence-এ সহজ ব্যাখ্যা
এই code-টি research exploration-এর একটি নির্দিষ্ট প্রশ্ন পরীক্ষা করার জন্য লেখা হয়েছিল। আমি প্রথমে dataset ও target বুঝেছি, তারপর নির্দিষ্ট split এবং model/method ব্যবহার করেছি, এরপর validation/test বা descriptive audit থেকে metric বের করেছি। ফলাফলকে তার experiment design-এর সীমার মধ্যে ব্যাখ্যা করতে হবে; এটি নিজে থেকে drinking-water safety guarantee বা external validation প্রমাণ করে না।
