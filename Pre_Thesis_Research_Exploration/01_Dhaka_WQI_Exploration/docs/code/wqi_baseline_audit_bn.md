# WQI baseline audit — বাংলা কোড ব্যাখ্যা

## ১. কোডটি কেন তৈরি করা হয়েছিল?
Dhaka dataset-এর WQI score predict করার জন্য কয়েকটি regression baseline তুলনা এবং random, time-based ও unseen-station split-এ performance কেমন হয় তা দেখা।

## ২. কোন Dataset ব্যবহার করে?
`Water quality dhaka.csv`; input হিসেবে temperature, pH, DO, COD, nitrate, turbidity, phosphate ও rainfall; target `WQI`। `Pollution_Label`, date এবং location identifier predictor হিসেবে ব্যবহার করা হয়নি।

## ৩. কোডটি ধাপে ধাপে কী করে
1. CSV থেকে feature, WQI, date, year ও coordinate-based station group তৈরি করে।
2. Random 80/20 split তৈরি করে।
3. 2017–2019 training এবং 2020–2021 test temporal split তৈরি করে।
4. GroupShuffleSplit দিয়ে unseen-station split তৈরি করে।
5. Dummy mean, Ridge, Random Forest, HistGradientBoosting ও MLP regression fit করে।
6. প্রতিটি split/model-এর MAE, RMSE ও R² গণনা করে।
7. CSV result এবং অতিরিক্ত JSON audit summary লেখে।

## ৪. কী Output তৈরি করে?
`wqi_baseline_results.csv`; script অনুযায়ী `wqi_audit_summary.json`-ও তৈরি হওয়ার কথা, কিন্তু supplied ZIP-এ JSON ফাইলটি নেই।

## ৫. কীভাবে ফলাফল বুঝবে?
এই script-এর output CSV-তে row/column, metric এবং split/rule অনুযায়ী ফলাফল থাকে। Summary file থাকলে সেটি বিভিন্ন split-এর aggregate summary; detailed file-এ split বা record-level ফল থাকে। CSV-এর মানকে এই documentation একা final thesis conclusion হিসেবে ঘোষণা করছে না—ফলাফল ব্যাখ্যার আগে code version, dataset version ও evaluation design মিলিয়ে দেখতে হবে।

## ৬. সীমাবদ্ধতা ও সতর্কতা
- এটি exploratory baseline; final WQI model নির্বাচন নয়।
- WQI formula ও weights verified না হওয়ায় target-এর অর্থ ও provenance অনিশ্চিত।
- Script-এ `/mnt/data/Water quality.csv` hard-coded; তোমার দেওয়া filename `Water quality dhaka.csv`। Windows/local project-এ চালানোর আগে path ও filename ঠিক করতে হবে।
- Temporal split-এ year cutoff script-এ স্থির করা আছে; dataset update হলে split logic পুনরায় যাচাই করতে হবে।

## ৭. Thesis defence-এ সহজ ব্যাখ্যা
এই code-টি research exploration-এর একটি নির্দিষ্ট প্রশ্ন পরীক্ষা করার জন্য লেখা হয়েছিল। আমি প্রথমে dataset ও target বুঝেছি, তারপর নির্দিষ্ট split এবং model/method ব্যবহার করেছি, এরপর validation/test বা descriptive audit থেকে metric বের করেছি। ফলাফলকে তার experiment design-এর সীমার মধ্যে ব্যাখ্যা করতে হবে; এটি নিজে থেকে drinking-water safety guarantee বা external validation প্রমাণ করে না।
