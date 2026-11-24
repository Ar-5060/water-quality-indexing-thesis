# WQI target integrity check — বাংলা কোড ব্যাখ্যা

## ১. কোডটি কেন তৈরি করা হয়েছিল?
WQI target ও Pollution_Label-এর সম্পর্ক, COD anomaly এবং phosphate/WQI summary বর্ণনামূলকভাবে পরীক্ষা করা।

## ২. কোন Dataset ব্যবহার করে?
Command-line argument-এ দেওয়া CSV; default filename `Water quality.csv`।

## ৩. কোডটি ধাপে ধাপে কী করে
1. CSV পড়ে WQI, phosphate, COD ও date column নেয়।
2. WQI < 50, 50–75 এবং ≥75 threshold থেকে label পুনর্গঠন করে।
3. Label match/mismatch, negative COD, COD=70, phosphate correlation ও unique-value count গণনা করে।
4. বছরভিত্তিক COD=70 count যোগ করে report CSV লেখে।

## ৪. কী Output তৈরি করে?
Default output `wqi_target_integrity_report.csv`; `--csv` ও `--out` argument দিয়ে input/output বদলানো যায়।

## ৫. কীভাবে ফলাফল বুঝবে?
এই script-এর output CSV-তে row/column, metric এবং split/rule অনুযায়ী ফলাফল থাকে। Summary file থাকলে সেটি বিভিন্ন split-এর aggregate summary; detailed file-এ split বা record-level ফল থাকে। CSV-এর মানকে এই documentation একা final thesis conclusion হিসেবে ঘোষণা করছে না—ফলাফল ব্যাখ্যার আগে code version, dataset version ও evaluation design মিলিয়ে দেখতে হবে।

## ৬. সীমাবদ্ধতা ও সতর্কতা
- Observed threshold-এ label match মানে publisher-এর প্রকৃত WQI formula বা label-generation process প্রমাণ নয়।
- Script default input `Water quality.csv`; তোমার dataset-এর filename `Water quality dhaka.csv`, তাই argument দিয়ে সঠিক নাম দিতে হবে।

## ৭. Thesis defence-এ সহজ ব্যাখ্যা
এই code-টি research exploration-এর একটি নির্দিষ্ট প্রশ্ন পরীক্ষা করার জন্য লেখা হয়েছিল। আমি প্রথমে dataset ও target বুঝেছি, তারপর নির্দিষ্ট split এবং model/method ব্যবহার করেছি, এরপর validation/test বা descriptive audit থেকে metric বের করেছি। ফলাফলকে তার experiment design-এর সীমার মধ্যে ব্যাখ্যা করতে হবে; এটি নিজে থেকে drinking-water safety guarantee বা external validation প্রমাণ করে না।
