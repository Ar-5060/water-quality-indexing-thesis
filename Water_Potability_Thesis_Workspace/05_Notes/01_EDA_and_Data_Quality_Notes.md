# Lesson 01 — EDA and Data Quality Notes

## Purpose

Understand the raw dataset before preprocessing or model training. EDA means Exploratory Data Analysis.

## Dataset

File: `01_Data/raw/water_potability.csv`

- Shape: **3,276 rows × 10 columns**
- 9 input features + 1 target (`Potability`)
- Numeric features: `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`
- Target: `Potability` (`0` = non-potable, `1` = potable)

## Class distribution

| Label | Meaning | Samples | Approx. share |
|---:|---|---:|---:|
| 0 | Non-potable | 1,998 | 61% |
| 1 | Potable | 1,278 | 39% |

The classes are not perfectly balanced. Therefore, accuracy alone is not enough to evaluate a model.

## Missing values

| Feature | Missing cells | Approx. missing % |
|---|---:|---:|
| `ph` | 491 | 14.99% |
| `Sulfate` | 781 | 23.84% |
| `Trihalomethanes` | 162 | 4.95% |
| Other features | 0 | 0% |
| **Total** | **1,434** | — |

Dropping every row with a missing value would discard a substantial amount of data, so missing-value handling is required.

## Data-quality checks

Recorded output from `03_data_quality_checks.py`:

- Exact duplicate rows: **0**
- Negative values: **0** in all nine features
- Zero pH values: **1**
- Target labels found: `[0, 1]`
- Invalid target labels: **0**

The single pH value of zero was flagged for correction in the later preprocessing pipeline. A zero pH is treated as physically implausible for this dataset's measurement context; the code sets it to missing rather than deleting the entire row.

## IQR outlier inspection

The IQR method flagged possible outliers, not automatically invalid observations.

| Feature | IQR outlier count |
|---|---:|
| ph | 46 |
| Hardness | 83 |
| Solids | 47 |
| Chloramines | 61 |
| Sulfate | 41 |
| Conductivity | 11 |
| Organic_carbon | 25 |
| Trihalomethanes | 33 |
| Turbidity | 19 |

An IQR outlier is a statistical flag. It is not proof that the measurement is wrong. Do not remove such values without scientific justification.

## Figures created

Saved in `04_Results/figures/`:

1. `01_class_distribution.png`
2. `02_missing_values.png`
3. `03_feature_distributions.png`
4. `04_correlation_heatmap.png`
5. `05_feature_boxplots.png`

The class-wise histograms show substantial overlap between potable and non-potable feature distributions. The correlation heatmap shows weak linear correlations with the target. Weak Pearson correlation does not rule out nonlinear relationships.

## What we learned

1. The dataset has enough rows for the planned five-fold internal evaluation.
2. Missing values are concentrated in three features.
3. There is a moderate class imbalance.
4. No exact duplicate or negative feature values were found.
5. Outliers need careful treatment; automatic deletion is not justified.
6. The target labels' laboratory provenance is not documented in the CSV.

## Defence explanation

**What is EDA?**  
EDA is the first inspection of a dataset. It helps us understand its size, variables, missing values, class balance and unusual observations before building a model.

**Why not delete all missing rows?**  
Because many rows contain missing measurements. Deleting them would reduce the dataset and may change its class distribution.

**Does an outlier mean an error?**  
No. It is an unusually distant value under a statistical rule. It must be investigated before removal.

## Run

```powershell
python 02_Code/01_data_exploration.py
python 02_Code/02_data_visualization.py
python 02_Code/03_data_quality_checks.py
```

## Outputs to preserve

Keep all five figures and `04_Results/tables/data_quality_report.csv`.
