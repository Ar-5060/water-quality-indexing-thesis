"""Exploratory repeated held-out evaluation; not a water safety certification.
For each seed, train 60%, select on validation 20%, test on untouched 20%.
One-sided Clopper-Pearson 95% upper bound for P(predict potable | actual non-potable).
Validation-bound is a marginal binomial bound conditional on a fixed threshold;
selecting thresholds on same validation data voids its nominal simultaneous coverage.
The threshold search is therefore exploratory, not a 95%-guaranteed safety method.
"""
import numpy as np, pandas as pd
from scipy.stats import beta
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
BASE=Path(__file__).parent
raw=pd.read_csv(BASE/'water_potability.csv');X=raw.drop(columns='Potability');y=raw.Potability.astype(int)
SEEDS=[11,17,23,29,37,43,51,61]
RISK=0.05

def risk(y,probs,t):
    y=np.asarray(y); pred=probs>=t
    neg=y==0;pos=y==1
    fp=int((neg&pred).sum()); tp=int((pos&pred).sum());nneg=int(neg.sum());npos=int(pos.sum())
    u=float(beta.ppf(.95,fp+1,nneg-fp)) if fp < nneg else 1.
    return {'fp':fp,'tp':tp,'nonpotable':nneg,'potable':npos,'fpr':fp/nneg,'potable_recall':tp/npos,'upper95':u}

def select_threshold(y,p,method):
    # all distinct prediction scores: >= cutoff; cutoff > max means classify nobody potable
    candidates=np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]
    valid=[]
    for t in candidates:
        r=risk(y,p,t)
        ok=(r['fpr']<=RISK) if method=='empirical_5pct' else (r['upper95']<=RISK)
        if ok: valid.append((int(r['tp']),float(t)))
    return sorted(valid,key=lambda x:(-x[0],x[1]))[0][1]

rows=[]
for seed in SEEDS:
    Xtv,Xte,ytv,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=seed)
    Xtr,Xva,ytr,yva=train_test_split(Xtv,ytv,test_size=.25,stratify=ytv,random_state=seed+100)
    models={
        'Random Forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=120,min_samples_leaf=3,n_jobs=-1,random_state=seed)),
        'MLP':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),alpha=1,max_iter=180,early_stopping=True,n_iter_no_change=12,random_state=seed))
    }
    for name,model in models.items():
        model.fit(Xtr,ytr)
        pv=model.predict_proba(Xva)[:,1];pt=model.predict_proba(Xte)[:,1]
        thresholds={'fixed_0.50':.5,'empirical_5pct':select_threshold(yva,pv,'empirical_5pct'),'upper_bound_5pct':select_threshold(yva,pv,'upper_bound_5pct')}
        for rule,t in thresholds.items():
            v=risk(yva,pv,t);test=risk(yte,pt,t)
            rows.append({'seed':seed,'model':name,'rule':rule,'threshold':t,'validation_fp':v['fp'],'validation_fpr':v['fpr'],'validation_upper95':v['upper95'],'test_fp':test['fp'],'test_fpr':test['fpr'],'test_tp':test['tp'],'test_potable_recall':test['potable_recall'],'test_upper95':test['upper95'],'test_roc_auc':roc_auc_score(yte,pt),'test_pr_auc':average_precision_score(yte,pt),'test_brier':brier_score_loss(yte,pt),'test_n_neg':test['nonpotable'],'test_n_pos':test['potable']})
    print('finished split',seed,flush=True)
df=pd.DataFrame(rows);df.to_csv(BASE/'potability_risk_bound_detailed.csv',index=False)
summary=df.groupby(['model','rule']).agg(n_splits=('seed','count'),mean_threshold=('threshold','mean'),mean_test_fpr=('test_fpr','mean'),sd_test_fpr=('test_fpr','std'),test_fpr_exceeds_5pct=('test_fpr',lambda a:int(sum(a>.05))),mean_potable_recall=('test_potable_recall','mean'),mean_test_fp=('test_fp','mean'),mean_test_tp=('test_tp','mean'),mean_test_roc_auc=('test_roc_auc','mean'),mean_test_pr_auc=('test_pr_auc','mean'),mean_test_brier=('test_brier','mean')).reset_index()
summary.to_csv(BASE/'potability_risk_bound_summary.csv',index=False)
print(summary.round(4).to_string(index=False))
