
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "01_Data" / "raw" / "water_potability.csv"

FIG_DIR = PROJECT_DIR / "04_Results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

features = df.columns.drop("Potability")

sns.set_theme(style="whitegrid")


# 3. Class distribution
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Potability",
    order=[0, 1],
    color="steelblue"
)

plt.title("Water Potability Class Distribution")
plt.xlabel("Potability (0 = Non-potable, 1 = Potable)")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=300)
plt.close()


# 4. Missing value percentage
missing_pct = (df.isnull().mean() * 100)
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

plt.figure(figsize=(7, 4))

sns.barplot(
    x=missing_pct.index,
    y=missing_pct.values,
    color="coral"
)

plt.title("Missing Values by Feature")
plt.xlabel("Features")
plt.ylabel("Missing Values (%)")

plt.tight_layout()
plt.savefig(FIG_DIR / "02_missing_values.png", dpi=300)
plt.close()


# 5. Feature distributions by class
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for ax, feature in zip(axes.flatten(), features):

    sns.histplot(
        data=df,
        x=feature,
        hue="Potability",
        kde=True,
        stat="density",
        common_norm=False,
        ax=ax
    )

    ax.set_title(feature)

plt.suptitle("Water Quality Feature Distributions by Potability",
             fontsize=16)

plt.tight_layout()
plt.savefig(FIG_DIR / "03_feature_distributions.png", dpi=300)
plt.close()


# 6. Correlation heatmap
correlation = df.corr(numeric_only=True)

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)

plt.title("Feature and Target Correlation Matrix")

plt.tight_layout()
plt.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=300)
plt.close()


# 7. Boxplots by potability class
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for ax, feature in zip(axes.flatten(), features):

    sns.boxplot(
        data=df,
        x="Potability",
        y=feature,
        ax=ax,
        color="skyblue"
    )

    ax.set_title(feature)
    ax.set_xlabel("Potability (0 = Non-potable, 1 = Potable)")

plt.suptitle("Water Quality Feature Boxplots by Potability",
             fontsize=16)

plt.tight_layout()
plt.savefig(FIG_DIR / "05_feature_boxplots.png", dpi=300)
plt.close()


# 8. Completion message
print("\n===== VISUALIZATION COMPLETED =====")
print("Total visualizations: 5")
print("Saved location:", FIG_DIR)
