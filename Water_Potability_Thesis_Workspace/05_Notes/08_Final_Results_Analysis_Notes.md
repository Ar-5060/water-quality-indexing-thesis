# Lesson 08 — Final Results Analysis and Thesis-Ready Outputs

## Purpose

Combine the verified five-fold Lesson 09 outputs into readable tables, graphs and a Markdown report. This is a reporting/analysis stage, not another model-training stage.

## Input requirements

The final script `10_final_results_analysis.py` reads the five-fold Lesson 09 tables from `04_Results/tables/`, including:
- pooled and fold-level probability metrics;
- pooled and fold-level threshold metrics;
- pooled and fold-level selective metrics;
- `test_predictions_all_folds.csv`.

It also checks the original `01_Data/raw/water_potability.csv`.

The script requires all five outer folds and checks that all 3,276 raw rows appear exactly once in the combined outer-test predictions. A partial fold run must not be reported as a completed five-fold analysis.

## What the script verifies

- Five expected folds are present.
- Row IDs and labels match the original raw dataset.
- Probability, threshold and selective result grids are complete.
- Primary FSR is verified from confusion counts as `FP / (FP + TN)`.
- Pooled metrics are distinguished from fold mean ± SD.
- No model is retrained.
- No threshold or selective cut-off is changed.
- No single composite model score is created.

## Statistical summaries

The script creates nominal descriptive intervals:
- stratified bootstrap intervals for selected probability metrics (2,000 resamples; seed 42);
- exact Clopper–Pearson intervals for relevant risk/recall quantities.

These intervals do not capture all model-refit uncertainty and are not a substitute for formal paired model-comparison tests. Do not describe them as proof of statistical superiority.

## Outputs

In `04_Results/tables/`:
- `10_probability_quality.csv`
- `10_threshold_comparison.csv`
- `10_selective_comparison.csv`
- `10_operating_points_meeting_target.csv`
- `10_metric_leaders.csv`
- `10_thesis_ready_model_comparison.csv`
- `10_final_results_report.md`
- `10_final_results_manifest.json`

In `04_Results/figures/`, the script saves eight thesis figures in PNG (300 dpi) and PDF formats:
- probability quality;
- average precision;
- threshold false-safe vs recall;
- selective false-safe vs recall;
- selective risk vs coverage;
- actual vs target coverage;
- false-safe confidence intervals;
- pooled ROC and reliability plots.

## Model selection interpretation

The script reports metric-wise leaders, not one overall winner. A model may rank best in ROC-AUC but not in calibration or false-safe risk. Safety-related metric leaders are not rewarded for trivially predicting no potable samples; the analysis checks discrimination and whether any potable declaration is made.

Interpret the following together:
- ROC-AUC and Average Precision: ranking/discrimination;
- Brier Score, Log Loss and ECE: probability quality;
- FSR and potable recall: risk/utility trade-off;
- coverage, review share and accepted error: selective-prediction workload and performance.

## Run

From the project root:

```powershell
python 02_Code/10_final_results_analysis.py
```

If the script reports that its own `10_` outputs already exist:

```powershell
python 02_Code/10_final_results_analysis.py --overwrite
```

`--overwrite` is intended to replace only the Lesson 10 outputs listed by the script. It does not change Lessons 01–09.

## Current status

The user confirmed that the Lesson 10 work completed. Preserve the actual local `10_final_results_report.md`, all `10_` tables/figures and `10_final_results_manifest.json`.

**Note:** The exact numerical contents of the local Lesson 10 report were not available in the current note-generation context. Copy final thesis numbers directly from that generated report and tables; do not invent or reconstruct them from Fold 1 logs.

## Defence explanation

Lesson 10 consolidates the five-fold test results and makes them easier to interpret. It does not improve or retrain the model. It helps explain the trade-offs and document the analysis reproducibly.
