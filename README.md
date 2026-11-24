<div align="center">

<h1>Evaluating Risk-Aware and Selective Machine Learning<br>for Water Potability Classification</h1>

B.Sc. Thesis Research in Software Engineering<br>
Daffodil International University

<img src="Water_Potability_Thesis_Workspace/06_Documentation/assets/water_wave.svg" alt="Animated water waves" width="100%">

<br>

<img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
<img src="https://img.shields.io/badge/Research-In%20Progress-orange" alt="Research in progress">
<img src="https://img.shields.io/badge/Evaluation-5--Fold%20CV-green" alt="5-fold CV">
<img src="https://img.shields.io/badge/Documentation-Ongoing-blueviolet" alt="Documentation ongoing">

</div>

<br>

## 1. Research Overview

This research investigates **risk-aware and selective machine learning approaches for water potability classification**, with particular attention to reducing false-safe predictions, that is, non-potable water classified as potable.

Rather than relying solely on conventional classification accuracy, the study examines how probability calibration, risk-controlled decision thresholds and selective prediction influence classification errors and decision coverage.

## 2. Research Information

| | |
|:--|:--|
| **Researcher** | Anisur Rahman |
| **Programme** | B.Sc. in Software Engineering |
| **Institution** | Daffodil International University |
| **Supervisor** | Fahad Hossain |
| **Research Area** | Machine Learning and Water Potability |
| **Working Title** | *Evaluating Risk-Aware and Selective Machine Learning for Water Potability Classification* |
| **Status** | Research implementation and thesis documentation in progress |

## 3. Research Focus

- Water potability classification using machine learning
- Probability calibration and reliability assessment
- Comparison of fixed and risk-controlled thresholds
- False-Safe Rate (FSR) analysis
- Selective prediction with a *Needs Review* option
- Evaluation of decision coverage, potable recall and review workload

## 4. Dataset and Models

**Dataset.** [Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability) (Kaggle): 3,276 observations, 9 physicochemical input features and a binary target `Potability` (0 = non-potable, 1 = potable).

**Models under investigation.**

| Model | Role |
|:--|:--|
| Logistic Regression | Linear classifier |
| Random Forest | Tree ensemble |
| Multi-Layer Perceptron (MLP) | Neural network |
| Majority Class Baseline | Reference only |

## 5. Current Research Progress

| Research Component | Status |
|:--|:--|
| Literature review and research preparation | Organized |
| Dataset exploration and quality assessment | Completed |
| Data preprocessing pipeline | Implemented |
| Model training and comparison pipeline | Implemented |
| Probability calibration | Implemented |
| Risk-controlled threshold selection | Implemented |
| Selective prediction | Implemented |
| Five-fold experiment organization | Prepared |
| Model evaluation and results analysis | Implemented |
| Research interpretation and thesis writing | Ongoing |

The experimental outputs are being organized and reviewed. Final interpretation and thesis documentation remain ongoing.

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
│   ├── 06_Documentation/
│   ├── README.md
│   ├── requirements.txt
│   └── .gitignore
└── README.md
```

The full methodology, experimental design and reproduction steps are documented in [`Water_Potability_Thesis_Workspace/README.md`](Water_Potability_Thesis_Workspace/README.md).

## 7. Research and Documentation Status

The computational workflow and experimental workspace have been organized into a structured research environment. Thesis documentation is currently in progress, including methodology documentation, experimental result interpretation, research discussion, limitations and preparation of the final thesis chapters.

The repository will continue to be updated as the research and documentation develop.

## 8. Research Limitations

- The study uses a publicly available labelled dataset; independent laboratory label provenance has not been established.
- External validation on an independently labelled dataset remains outstanding.
- The 5% False-Safe Rate is an experimental target, not a guarantee of future safety.
- Model predictions must not be interpreted as drinking-water safety certification.

---

<p align="center">
<sub>Anisur Rahman · B.Sc. in Software Engineering · Daffodil International University</sub>
</p>
