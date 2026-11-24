"""Post-hoc risk/coverage analysis of the previously saved 8-split experiment.
Does not retrain models. Test partition for every seed contains 656 samples.
False-safe rate = FP/(FP+TN); positive prediction rate = (TP+FP)/N.
Precision = TP/(TP+FP); undefined if nobody is predicted potable.
"""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
raw=pd.read_csv(ROOT/'potability_corrected_risk_detailed.csv')
N_TEST=656
raw['predicted_potable_count']=raw.test_false_safe+raw.test_potable_true_positive
raw['predicted_potable_pct']=100*raw.predicted_potable_count/N_TEST
raw['potable_precision_pct']=np.where(raw.predicted_potable_count>0,100*raw.test_potable_true_positive/raw.predicted_potable_count,np.nan)
raw['false_safe_pct']=100*raw.test_false_safe_rate
raw['potable_recall_pct']=100*raw.test_potable_recall
raw['negative_predictions_pct']=100-raw.predicted_potable_pct
raw.to_csv(ROOT/'potability_risk_coverage_detailed.csv',index=False)
summary=raw.groupby(['model','rule'],sort=False).agg(
    splits=('seed','size'),
    mean_predicted_potable_pct=('predicted_potable_pct','mean'),
    sd_predicted_potable_pct=('predicted_potable_pct','std'),
    mean_potable_precision_pct=('potable_precision_pct','mean'),
    mean_false_safe_pct=('false_safe_pct','mean'),
    mean_potable_recall_pct=('potable_recall_pct','mean'),
    mean_predicted_potable_count=('predicted_potable_count','mean'),
    mean_false_safe_count=('test_false_safe','mean'),
    mean_true_potable_count=('test_potable_true_positive','mean'),
    splits_exceeding_5pct=('test_exceeds_5pct','sum'),
    fully_negative_splits=('predicted_potable_count',lambda x:(x==0).sum()),
).reset_index()
summary.to_csv(ROOT/'potability_risk_coverage_summary.csv',index=False)
print(summary.round(2).to_string(index=False))
