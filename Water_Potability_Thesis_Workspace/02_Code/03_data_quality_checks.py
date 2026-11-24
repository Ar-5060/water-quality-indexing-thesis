
from pathlib import Path
import pandas as pd

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "01_Data" / "raw" / "water_potability.csv"

OUTPUT_DIR = PROJECT_DIR / "04_Results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

features = df.columns.drop("Potability")

print("\n===== DATA QUALITY INSPECTION =====")

# 3. Check zero values
print("\n--- ZERO VALUES ---")
print((df[features] == 0).sum())

# 4. Check negative values
print("\n--- NEGATIVE VALUES ---")
print((df[features] < 0).sum())

# 5. Check invalid target labels
print("\n--- TARGET LABEL VALIDATION ---")
print("Unique labels:", sorted(df["Potability"].dropna().unique()))
print("Invalid labels:", (~df["Potability"].isin([0, 1])).sum())

# 6. Check duplicate rows
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# 7. IQR-based outlier detection
print("\n--- OUTLIER DETECTION (IQR METHOD) ---")

quality_report = []

for feature in features:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (df[feature] < lower_bound) |
        (df[feature] > upper_bound)
    )

    quality_report.append({
        "Feature": feature,
        "Missing": df[feature].isnull().sum(),
        "Zero": (df[feature] == 0).sum(),
        "Negative": (df[feature] < 0).sum(),
        "Outliers": outliers.sum(),
        "Lower Bound": round(lower_bound, 2),
        "Upper Bound": round(upper_bound, 2)
    })

report = pd.DataFrame(quality_report)

print(report.to_string(index=False))

# 8. Save quality report
report.to_csv(
    OUTPUT_DIR / "data_quality_report.csv",
    index=False
)

print("\n===== INSPECTION COMPLETED =====")
print("Report saved to:", OUTPUT_DIR)
