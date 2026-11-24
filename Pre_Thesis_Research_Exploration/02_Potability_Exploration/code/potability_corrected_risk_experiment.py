"""Repeated exploratory experiment: Bonferroni-adjusted one-sided Clopper-Pearson.
Note: Guarantees assume IID calibration/test and an independent trained scoring function.
A family-wise confidence band covers a predeclared grid of fixed thresholds under these assumptions.
Does NOT certify drinking-water safety; original labels lack source verification.
"""
import csv, warnings
from pathlib import Path
import numpy as np
from scipy.stats import beta
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE=Path(__file__).parent
with (BASE/'water_potability.csv').open(newline='') as f:
    rdr=csv.DictReader(f);data=list(rdr);features=[c for c in rdr.fieldnames if c!='Potability']
X=np.array([[float(row[c]) if row[c].strip() else np.nan for c in features] for row in data],dtype=float)
y=np.array([int(float(row['Potability'])) for row in data],dtype=int)
SEEDS=[11,17,23,29,37,43,51,61]
RISK=.05;ALPHA=.05

def metrics(y,p,threshold):
    neg=y==0;pos=y==1
    pred=p>=threshold
    fp=int(np.sum(pred&neg));tp=int(np.sum(pred&pos))
    nneg=int(neg.sum());npos=int(pos.sum())
    return dict(fp=fp,tp=tp,nneg=nneg,npos=npos,fpr=fp/nneg,recall=tp/npos)

def upper_bound(fp,n,alpha):
    return float(beta.ppf(1-alpha,fp+1,n-fp)) if fp < n else 1.

def selected(y,p,rule):
    # Fixed, predeclared grid avoids choosing the candidate threshold family from validation data.
    candidates=np.r_[np.linspace(0.0,1.0,201),np.nextafter(1.0,np.inf)]
    m=len(candidates)
    alpha=ALPHA/m if rule=='bonferroni_5pct' else ALPHA
    opts=[]
    for t in candidates:
        z=metrics(y,p,t)
        ub=upper_bound(z['fp'],z['nneg'],alpha)
        ok=(z['fpr']<=RISK) if rule=='empirical_5pct' else (ub<=RISK)
        if ok:opts.append((-z['tp'],float(t),ub))
    # Rejection of all predicted potable is always included and should satisfy bound if n is adequate.
    if not opts: return float(candidates[-1]),m,False
    return min(opts)[1],m,True

out=[]
with warnings.catch_warnings():
    warnings.filterwarnings('ignore',category=UserWarning)
    for seed in SEEDS:
        indices=np.arange(len(y))
        tv,te=train_test_split(indices,test_size=.2,stratify=y,random_state=seed)
        tr,va=train_test_split(tv,test_size=.25,stratify=y[tv],random_state=seed+100)
        models={
          'Random Forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=120,min_samples_leaf=3,n_jobs=-1,random_state=seed)),
          'MLP':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),alpha=1,max_iter=180,early_stopping=True,n_iter_no_change=12,random_state=seed))
        }
        for name,model in models.items():
            model.fit(X[tr],y[tr]);pv=model.predict_proba(X[va])[:,1];pt=model.predict_proba(X[te])[:,1]
            rules={'fixed_0.50':(.5,1,True)}
            for rule in ['empirical_5pct','uncorrected_bound_5pct','bonferroni_5pct']:
                rules[rule]=selected(y[va],pv,rule)
            for rule,(t,m,feasible) in rules.items():
                v=metrics(y[va],pv,t);z=metrics(y[te],pt,t)
                out.append({'seed':seed,'model':name,'rule':rule,'threshold':t,'candidates':m,'feasible':int(feasible),'validation_false_safe':v['fp'],'validation_false_safe_rate':v['fpr'],'validation_bonferroni_upper':upper_bound(v['fp'],v['nneg'],ALPHA/m),'test_false_safe':z['fp'],'test_false_safe_rate':z['fpr'],'test_potable_true_positive':z['tp'],'test_potable_recall':z['recall'],'test_exceeds_5pct':int(z['fpr']>RISK),'test_roc_auc':roc_auc_score(y[te],pt),'test_pr_auc':average_precision_score(y[te],pt),'test_brier':brier_score_loss(y[te],pt)})
        print('Completed split',seed,flush=True)

with (BASE/'potability_corrected_risk_detailed.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
summary=[]
for model in ['Random Forest','MLP']:
    for rule in ['fixed_0.50','empirical_5pct','uncorrected_bound_5pct','bonferroni_5pct']:
        r=[z for z in out if z['model']==model and z['rule']==rule]
        a=lambda col:np.array([z[col] for z in r],dtype=float)
        summary.append({'model':model,'rule':rule,'splits':len(r),'mean_threshold':a('threshold').mean(),'mean_candidates':a('candidates').mean(),'mean_false_safe_rate':a('test_false_safe_rate').mean(),'sd_false_safe_rate':a('test_false_safe_rate').std(ddof=1),'exceeded_5pct':int(a('test_exceeds_5pct').sum()),'mean_potable_recall':a('test_potable_recall').mean(),'mean_false_safe_count':a('test_false_safe').mean(),'mean_true_potable_count':a('test_potable_true_positive').mean(),'mean_roc_auc':a('test_roc_auc').mean(),'mean_brier':a('test_brier').mean(),'fully_rejecting_splits':int(sum(z['test_potable_true_positive']==0 and z['test_false_safe']==0 for z in r))})
with (BASE/'potability_corrected_risk_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
for s in summary:print(s['model'],s['rule'],'FPR',round(100*s['mean_false_safe_rate'],2),'Recall',round(100*s['mean_potable_recall'],2),'violations',s['exceeded_5pct'],'reject_all',s['fully_rejecting_splits'])
