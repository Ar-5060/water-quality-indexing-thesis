# Research Interpretation and Thesis-Writing Roadmap

## 1. Research questions to organise the discussion

Use the approved proposal/protocol as the governing source and confirm wording with the supervisor. The working questions recorded in the project handover are:

- **RQ1:** How do fixed, empirical and Bonferroni-corrected thresholds change false-safe rate and potable recall under the same denominator?
- **RQ2:** What changes in decision coverage, accepted error, false-safe rate and review workload occur when abstention is allowed?
- **RQ3:** How stable are these risk–utility trade-offs across outer folds and across Logistic Regression, Random Forest and MLP?
- **RQ4 (secondary):** How do missing-value handling and probability calibration affect the results?

## 2. Suggested Chapter 4 structure — Results and Discussion

### 4.1 Dataset profile and data quality
Use Lesson 01:
- dataset size and features;
- class distribution;
- missing-value profile;
- duplicate/negative/zero checks;
- figures for distributions, correlation and boxplots.

### 4.2 Experimental protocol
Explain:
- stratified outer 5-fold evaluation;
- inner 3-fold model tuning;
- separate calibration subset;
- four models and preprocessing pipelines;
- which data are allowed in each stage.

### 4.3 Probability-model performance
Use Lesson 09/10:
- ROC-AUC;
- Average Precision;
- Brier Score;
- Log Loss;
- ECE;
- pooled results and fold mean ± SD.

### 4.4 Risk-aware threshold comparison
Compare:
- fixed 0.50;
- empirical;
- Clopper–Pearson;
- Bonferroni.

For every method discuss FSR and potable recall together. Explain the cost of a conservative threshold.

### 4.5 Selective prediction
Discuss:
- 25%, 50%, 75%, 100% target coverage;
- achieved coverage;
- Needs Review share;
- accepted error;
- all-population FSR;
- all-population potable recall.

Do not describe Needs Review as non-potable or potable.

### 4.6 Cross-fold consistency
Use fold-level mean ± SD and pooled metrics. Explain differences between pooled results and the average of five fold-level metrics.

### 4.7 Discussion
Answer each research question using actual tables. Explain why one model can be better in discrimination but worse in risk or calibration. Avoid declaring one universal winner unless the thesis has a pre-specified decision rule.

### 4.8 Limitations
Discuss label provenance, no external validation, public dataset limitations, sample representativeness, distribution shift, finite calibration sample size and the need for laboratory confirmation.

## 3. Chapter 5 — Conclusion, Limitations and Future Work

Suggested sections:
1. Summary of the research.
2. Main findings (insert verified Lesson 10 values).
3. Contributions: reproducible applied comparison under explicit risk definitions and abstention.
4. Practical implications: review workload must be feasible; predictions are decision support only.
5. Limitations.
6. Future work: independently labelled data, location/time-aware validation, external validation, and collaboration with water-quality laboratories.

## 4. Claims to avoid

- “The model guarantees safe drinking water.”
- “The model is the first risk-aware water-potability model.”
- “The 5% calibration target guarantees 5% future test FSR.”
- “Five folds on one public dataset equal external validation.”
- “The Dhaka WQI-derived pollution label is an independent laboratory potability label.”

## 5. Source and citation discipline

- Verify every paper against its original publication before final citation formatting.
- The project literature workbook records 17 entries (3 base and 14 supporting entries); do not assume that every source has been fully verified merely because it is in the workbook.
- Use the university's thesis template and current DIU formatting requirements.
- Do not create references, DOIs, sample sizes or publication details from memory.

## 6. Defence preparation topics

Be ready to explain in your own words:
- why the research moved from Dhaka WQI to Kaggle Potability;
- why pH zero was corrected;
- why imputation/scaling must be training-only;
- why outer and inner CV are separate;
- what probability calibration does;
- why threshold selection and selective prediction are different;
- why the FSR denominator is `FP + TN`;
- why low FSR can coincide with very low potable recall;
- why Needs Review creates a workload;
- why results do not certify real drinking-water safety.
