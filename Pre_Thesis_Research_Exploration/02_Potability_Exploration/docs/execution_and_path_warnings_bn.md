# পুরোনো Script চালানোর আগে Path ও Environment সতর্কতা

এই archive-এর script-গুলো historical original হিসেবে রাখা হয়েছে; path পরিবর্তন করে original source overwrite করা হয়নি।

## WQI scripts
কয়েকটি WQI script `/mnt/data/Water quality.csv` অথবা default `Water quality.csv` খোঁজে। তোমার uploaded dataset-এর নাম `Water quality dhaka.csv`। Windows-এ সরাসরি run করলে path/filename error হতে পারে।

## Potability scripts
কয়েকটি script `Path(__file__).parent / 'water_potability.csv'` ব্যবহার করে। নতুন folder structure-এ dataset `data/` subfolder-এ আছে, `code/` folder-এর পাশে নয়। ফলে script চালানোর আগে path adjust করা প্রয়োজন।

## কীভাবে নিরাপদে পরিবর্তন করবে
1. Original script অপরিবর্তিত রাখবে।
2. প্রয়োজন হলে `*_portable.py` নামে আলাদা copy তৈরি করবে।
3. সেই copy-তে relative path ব্যবহার করবে।
4. `requirements.txt`, Python version, random seed ও run command নথিবদ্ধ করবে।
5. নতুন run-এর output আলাদা results/version folder-এ রাখবে; historical CSV overwrite করবে না।
