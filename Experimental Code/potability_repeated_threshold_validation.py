"""Repeated held-out test threshold evaluation. Not a real drinking water safety classifier.
8 independently seeded stratified 60/20/20 train-validation-test splits.
Threshold is selected ONLY on validation data; preprocessing/calibration fitted ONLY on training.
Overlapping test sets mean repeats are NOT independent observations.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
p=Path(__file__).resolve().parent
raw=pd.read_csv(p/'water_potability.csv'); X=raw.drop(columns='Potability');y=raw.Potability.astype(int)
THRESHOLDS=np.linspace(0,1,1001)
def count(y,p,t):
    yn=np.asarray(y);pred=(p>=t)
    tn=((yn==0)&(~pred)).sum();fp=((yn==0)&pred).sum();fn=((yn==1)&(~pred)).sum();tp=((yn==1)&pred).sum()
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),false_safe_rate=float(fp/(fp+tn)),potable_recall=float(tp/(tp+fn)),predicted_potable=int(fp+tp))
def choose(y,p):
    # find largest number of true positives subject to validation FPR <= 5%;
    # choosing smallest threshold in case of ties. threshold 1.0001 permits abstaining entirely.
    good=[(t,count(y,p,t)) for t in np.r_[THRESHOLDS,1.0001] if count(y,p,t)['false_safe_rate'] <= .05]
    return min(good,key=lambda e:(-e[1]['tp'],e[0]))[0]
rows=[]
for seed in [11,17,23,29,37,43,51,61]:
    Xtrainval,Xte,ytrainval,yte=train_test_split(X,y,stratify=y,test_size=.2,random_state=seed)
    Xtr,Xva,ytr,yva=train_test_split(Xtrainval,ytrainval,stratify=ytrainval,test_size=.25,random_state=seed+100)
    rf=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=120,min_samples_leaf=3,n_jobs=-1,random_state=seed))
    mlp=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),alpha=1,max_iter=180,early_stopping=True,n_iter_no_change=12,random_state=seed))
    for name,model in [('RF raw',rf),('RF calibrated',CalibratedClassifierCV(estimator=rf,method='sigmoid',cv=3)),('MLP raw',mlp)]:
        model.fit(Xtr,ytr)
        pv=model.predict_proba(Xva)[:,1];pt=model.predict_proba(Xte)[:,1];t=choose(yva,pv)
        for method,thr in [('fixed_0.50',.5),('validation_5pct',t)]:
            v=count(yva,pv,thr);te=count(yte,pt,thr)
            rows.append(dict(seed=seed,model=name,rule=method,threshold=thr,validation_fpr=v['false_safe_rate'],validation_recall=v['potable_recall'],test_auc=roc_auc_score(yte,pt),test_brier=brier_score_loss(yte,pt),test_nontpotable=int((yte==0).sum()),test_potable=int((yte==1).sum()),**te))
    print('finished',seed,flush=True)
df=pd.DataFrame(rows);df.to_csv(p/'potability_repeated_threshold_detailed.csv',index=False)
summary=df.groupby(['model','rule']).agg(repeats=('seed','count'),threshold_median=('threshold','median'),mean_false_safe_pct=('false_safe_rate',lambda s:100*s.mean()),sd_false_safe_pct=('false_safe_rate',lambda s:100*s.std()),min_false_safe_pct=('false_safe_rate',lambda s:100*s.min()),max_false_safe_pct=('false_safe_rate',lambda s:100*s.max()),violations_5pct=('false_safe_rate',lambda s:int((s>.05).sum())),mean_potable_recall_pct=('potable_recall',lambda s:100*s.mean()),sd_potable_recall_pct=('potable_recall',lambda s:100*s.std()),mean_fp=('fp','mean'),mean_tp=('tp','mean'),mean_auc=('test_auc','mean'),mean_brier=('test_brier','mean')).reset_index()
summary.to_csv(p/'potability_repeated_threshold_summary.csv',index=False)
print(summary.to_string(index=False))
