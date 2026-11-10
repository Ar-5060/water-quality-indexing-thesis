"""Exploratory selective prediction with abstention on Kaggle water_potability.csv.
Model predicts 0/1 only when sufficiently far from p=0.5; otherwise Needs Review.
Validation selects a score cutoff for desired coverage, test is held out.
No drinking-water safety guarantee. Repeats share observations.
"""
import warnings
warnings.filterwarnings('ignore',category=UserWarning)
import numpy as np,pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss
P=Path(__file__).resolve().parent
D=pd.read_csv(P/'water_potability.csv');X=D.drop(columns='Potability');y=D['Potability'].astype(int)
seeds=[11,17,23,29,37,43,51,61]
rows=[]
for seed in seeds:
    Xtv,Xte,ytv,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=seed)
    Xtr,Xva,ytr,yva=train_test_split(Xtv,ytv,test_size=.25,stratify=ytv,random_state=seed+100)
    models={
      'Random Forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=120,min_samples_leaf=3,n_jobs=-1,random_state=seed)),
      'MLP': make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),alpha=1,max_iter=180,early_stopping=True,n_iter_no_change=12,random_state=seed))
    }
    for name,model in models.items():
        model.fit(Xtr,ytr)
        pva=model.predict_proba(Xva)[:,1]
        pte=model.predict_proba(Xte)[:,1]
        yt=np.asarray(yte)
        val_conf=np.abs(pva-.5)
        for intended in [1.0,.75,.5,.25]:
            # Quantile based on validation confidence. Selective coverage on test is not forced.
            cutoff=-1.0 if intended==1 else float(np.quantile(val_conf,1-intended))
            accept=np.abs(pte-.5)>=cutoff
            label=pte>=.5
            accepted=int(accept.sum());review=int((~accept).sum())
            a0=(yt==0)&accept;a1=(yt==1)&accept
            fp=int((a0&label).sum());tn=int((a0&~label).sum())
            tp=int((a1&label).sum());fn=int((a1&~label).sum())
            accepted_neg=int(a0.sum());accepted_pos=int(a1.sum())
            predicted_pos=tp+fp
            rows.append(dict(seed=seed,model=name,target_coverage=intended,validation_conf_cutoff=cutoff,
                 test_size=len(yt),accepted=accepted,needs_review=review,
                 coverage=accepted/len(yt),selective_error=(fp+fn)/accepted if accepted else np.nan,
                 selective_accuracy=(tp+tn)/accepted if accepted else np.nan,
                 false_safe=fp,correct_potable=tp,
                 false_safe_rate_accepted_neg=fp/accepted_neg if accepted_neg else np.nan,
                 false_safe_rate_all_neg=fp/(yt==0).sum(),
                 potable_recall_all=tp/(yt==1).sum(),
                 potable_recall_accepted=tp/accepted_pos if accepted_pos else np.nan,
                 positive_precision=tp/predicted_pos if predicted_pos else np.nan,
                 pred_potable_rate=predicted_pos/len(yt),
                 roc_auc=roc_auc_score(yt,pte),brier=brier_score_loss(yt,pte),
                 accepted_actual_nonpotable=accepted_neg, accepted_actual_potable=accepted_pos))
    print('completed seed',seed,flush=True)
full=pd.DataFrame(rows)
full.to_csv(P/'potability_selective_prediction_detailed.csv',index=False)
agg=full.groupby(['model','target_coverage'],as_index=False).agg(
 repeats=('seed','size'),avg_test_coverage_pct=('coverage',lambda x:x.mean()*100),
 sd_test_coverage_pct=('coverage',lambda x:x.std()*100),
 avg_review_count=('needs_review','mean'),
 avg_selective_error_pct=('selective_error',lambda x:x.mean()*100),
 avg_accepted_false_safe_pct=('false_safe_rate_accepted_neg',lambda x:x.mean()*100),
 avg_all_nonpotable_false_safe_pct=('false_safe_rate_all_neg',lambda x:x.mean()*100),
 avg_potable_recall_all_pct=('potable_recall_all',lambda x:x.mean()*100),
 avg_potable_precision_pct=('positive_precision',lambda x:x.mean()*100),
 avg_correct_potable=('correct_potable','mean'),
 avg_false_safe=('false_safe','mean'),
 avg_auc=('roc_auc','mean'))
agg.to_csv(P/'potability_selective_prediction_summary.csv',index=False)
print(agg.round(2).to_string(index=False))
