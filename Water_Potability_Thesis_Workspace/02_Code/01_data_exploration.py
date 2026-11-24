
from pathlib import Path
import pandas as pd

# 1. Find project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# 2. Dataset path
DATA_PATH = PROJECT_DIR / "01_Data" / "raw" / "water_potability.csv"

# 3. Load dataset
df = pd.read_csv(DATA_PATH)

# 4. Dataset shape
print("\n===== DATASET SHAPE =====")
print(df.shape)

# 5. Column names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# 6. First five rows
print("\n===== FIRST 5 ROWS =====")
print(df.head())

# 7. Data types
print("\n===== DATA TYPES =====")
print(df.dtypes)

# 8. Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 9. Missing value percentage
print("\n===== MISSING VALUE PERCENTAGE =====")
print((df.isnull().mean() * 100).round(2))

# 10. Duplicate rows
print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

# 11. Class distribution
print("\n===== CLASS DISTRIBUTION =====")
print(df["Potability"].value_counts())

# 12. Class distribution percentage
print("\n===== CLASS DISTRIBUTION PERCENTAGE =====")
print(df["Potability"].value_counts(normalize=True).round(3))

# 13. Statistical summary
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe().round(2))
