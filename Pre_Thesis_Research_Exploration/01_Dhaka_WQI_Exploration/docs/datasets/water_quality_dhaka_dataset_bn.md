# Dataset পরিচিতি: Dhaka Water Quality

## ১. Dataset-এর ভূমিকা
এই dataset প্রাথমিক WQI regression এবং pollution-category exploration-এ ব্যবহার করা হয়েছিল। এটি Kaggle Potability dataset থেকে আলাদা research track।

## ২. ফাইল ও আকার
- ফাইল: `data/Water quality dhaka.csv`
- Rows: 1,495
- Columns: 16
- Date range: 2017-01-25 থেকে 2021-07-17
- Unique location: 58
- Unique river/stream: 13
- Missing cell: 0

## ৩. Column-এর অর্থ
| Column | অর্থ / ভূমিকা |
|---|---|
| `Date` | পর্যবেক্ষণের তারিখ |
| `River/Stream` | নদী বা stream-এর নাম |
| `Location` | sampling location-এর নাম |
| `Latitude`, `Longitude` | ভৌগোলিক স্থানাঙ্ক |
| `Temp (°C)` | পানির তাপমাত্রা |
| `pH` | acidity/alkalinity নির্দেশক |
| `DO (mg/L)` | Dissolved Oxygen |
| `COD (mg/L)` | Chemical Oxygen Demand |
| `Nitrate (mg/L)` | Nitrate concentration |
| `Turbidity (NTU)` | পানির ঘোলাভাব |
| `Phosphate (mg/L)` | Phosphate concentration |
| `Rainfall (mm)` | বৃষ্টিপাত |
| `Season` | ঋতু |
| `WQI` | supplied numeric Water Quality Index |
| `Pollution_Label` | supplied category: Good / Poor / Very Poor |

## ৪. Audit-এ পাওয়া বিষয়
- Missing cell: 0
- `WQI < 50`, `50 ≤ WQI < 75`, `WQI ≥ 75` নিয়মে পুনর্গঠিত label supplied label-এর সঙ্গে 1495/1495 row-এ মেলে।
- COD = 70: 1301 row।
- COD negative: 6 row।

## ৫. সতর্কতা
Label threshold match হওয়া dataset publisher-এর প্রকৃত label-generation method প্রমাণ করে না। WQI formula, weights, parameter units এবং standards-এর authoritative documentation না পাওয়া পর্যন্ত target provenance unresolved থাকবে। `Pollution_Label`-কে independent drinking-water potability label বলা যাবে না।

## ৬. ব্যবহার
এই dataset কেবল Dhaka WQI exploratory track-এ ব্যবহার করতে হবে। Kaggle Potability dataset-এর row বা target-এর সঙ্গে merge করা যাবে না।
