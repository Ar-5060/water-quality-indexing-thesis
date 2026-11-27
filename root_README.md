<div align="center">

<h1>Evaluating Risk-Aware and Selective Machine Learning<br>for Water Potability Classification</h1>

B.Sc. Thesis in Software Engineering<br>
Department of Software Engineering, Daffodil International University

<img src="Water_Potability_Thesis_Workspace/06_Documentation/assets/water_wave.svg" alt="Animated water waves" width="100%">

<br>

<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
<img src="https://img.shields.io/badge/Thesis-Submitted%20Nov%202026-brightgreen" alt="Thesis submitted November 2026">
<img src="https://img.shields.io/badge/Evaluation-Nested%205--Fold%20CV-green" alt="Nested 5-fold CV">
<img src="https://img.shields.io/badge/Documentation-Complete-blueviolet" alt="Documentation complete">

</div>

<br>

## 1. Research Overview

This thesis investigates **risk-aware and selective machine learning approaches for water potability classification**, with particular attention to reducing false-safe predictions, that is, non-potable water classified as potable.

Rather than looking for the model with the highest accuracy, the study examines how probability calibration, risk-controlled decision thresholds and selective prediction change the error profile and decision coverage of potability classifiers, under a leakage-aware nested cross-validation design.

## 2. Research Information

| | |
|:--|:--|
| **Researcher** | Anisur Rahman |
| **Programme** | B.Sc. in Software Engineering |
| **Institution** | Department of Software Engineering, Daffodil International University |
| **Supervisor** | Fahad Hossain, Senior Lecturer, Department of Software Engineering, Daffodil International University |
| **Research Area** | Machine Learning and Water Potability |
| **Thesis Title** | *Evaluating Risk-Aware and Selective Machine Learning for Water Potability Classification* |
| **Status** | Completed; final thesis submitted November 2026 |

## 3. Key Findings

Pooled over five untouched outer-test folds (n = 3,276):

- **Random forest** and **MLP** rank samples moderately well and almost identically (ROC-AUC **0.685** and **0.684**); logistic regression does not rank better than chance (0.485).
- At the conventional **0.50 threshold**, about **11–12%** of non-potable samples are declared potable (RF 11.41%, MLP 11.91%).
- Against a **5% False-Safe Rate (FSR)** target, the Empirical, Clopper–Pearson and Bonferroni threshold strategies lower FSR to **4.55%, 2.60% and 0.50%** for random forest (5.01%, 3.35% and 0.70% for MLP), at a large cost in potable recall (down to 6.2–7.0% under Bonferroni).
- **Selective prediction** lowers the accepted error from about 33% to about 20%, but refers up to 76.7% of samples to *Needs Review*.
- **Sigmoid calibration** gives the best Brier score and log loss for all learned models without improving ranking ability.
- No single model is best on every measure; choosing an operating point is a policy decision about the acceptable FSR margin, required potable recall and available review capacity.

## 4. Dataset and Models

**Dataset.** ["Water Quality" dataset by A. Kadiwal](https://www.kaggle.com/datasets/adityakadiwal/water-potability) (Kaggle, 2021): 3,276 observations, 9 physicochemical input features and a binary target `Potability` (0 = non-potable, 1 = potable). The publisher describes the data as synthetically generated.

**Models compared.**

| Model | Role |
|:--|:--|
| Logistic Regression | Linear classifier |
| Random Forest | Tree ensemble |
| Multi-Layer Perceptron (MLP) | Neural network |
| Majority Class Baseline | Reference only |

## 5. Research Progress

| Research Component | Status |
|:--|:--|
| Literature review | Completed |
| Dataset exploration and quality assessment | Completed |
| Data preprocessing pipeline | Completed |
| Model training and comparison pipeline | Completed |
| Probability calibration | Completed |
| Risk-controlled threshold selection | Completed |
| Selective prediction | Completed |
| Five-fold experiments | Completed |
| Model evaluation and results analysis | Completed |
| Thesis writing (Chapters 1–5, references, Appendices A–D) | Completed |
| Final thesis submission | Submitted, November 2026 |

## 6. Repository Structure

```text
water-quality-indexing-thesis/
├── Literature Review/
├── Pre Thesis Research Exploration/
├── Proposal/
├── Thesis hand written notes/
├── Water_Potability_Thesis_Workspace/
│   ├── 01_Data/
│   ├── 02_Code/
│   ├── 03_Experiments/
│   ├── 04_Results/
│   ├── 05_Notes/
│   ├── 06_Documentation/      # includes the final thesis document (thesis/)
│   ├── README.md
│   ├── requirements.txt
│   └── .gitignore
└── README.md
```

The full methodology, experimental design, detailed results, thesis structure and reproduction steps are documented in [`Water_Potability_Thesis_Workspace/README.md`](Water_Potability_Thesis_Workspace/README.md).

## 7. Thesis Status

The research is complete. The final thesis, written in the DIU thesis format, has five chapters (Introduction, Literature Review, Methodology, Results and Discussion, Conclusion and Future Work), a reference list and four appendices (literature review matrix, fold-level tuning scores and decision rules, fold-level confusion matrices and fold-level selective prediction results). It was submitted in November 2026.

The code, saved experiment artifacts and result tables in this repository correspond to the submitted version of the thesis.

## 8. Research Limitations

- The study uses a single public dataset that its publisher describes as synthetically generated; its sampling and labelling procedures are not documented.
- No external validation on an independently labelled dataset was performed.
- The 5% False-Safe Rate is a calibration-side selection criterion, not a guarantee of future safety.
- Model predictions must not be interpreted as drinking-water safety certification.

---

<p align="center">
<sub>Anisur Rahman · B.Sc. in Software Engineering · Daffodil International University</sub>
</p>
