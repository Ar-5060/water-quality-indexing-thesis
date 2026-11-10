"""Reproducible educational potability risk/threshold benchmark.
Split: stratified train 60% / validation 20% / test 20% (random_state=42).
Calibration is fitted on TRAIN folds only. Threshold is chosen on VALIDATION only.
This is a research benchmark, not a drinking-water safety assessment.
"""
import warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,
                             brier_score_loss,log_loss,confusion_matrix,recall_score)
from sklearn.base import clone
from pathlib import Path
warnings.filterwarnings('ignore', category=UserWarning)
p=Path(__file__).parent
raw=pd.read_csv(p/'water_potability.csv'); X=raw.drop(columns=['Potability']); y=raw.Potability.astype(int)
Xa,Xtest,ya,ytest=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
Xtr,Xval,ytr,yval=train_test_split(Xa,ya,test_size=.25,stratify=ya,random_state=42)
models={
 'Logistic regression':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1500,random_state=42)),
 'Random forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=160,min_samples_leaf=3,n_jobs=-1,random_state=42)),
 'MLP':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),MLPClassifier(hidden_layer_sizes=(32,16),alpha=1.,max_iter=220,early_stopping=True,n_iter_no_change=12,random_state=42))}
def metrics(y,p,t):
    a=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,a,labels=[0,1]).ravel()
    return {'accuracy':accuracy_score(y,a),'balanced_accuracy':balanced_accuracy_score(y,a),
            'nonpotable_mislabeled_potable':int(fp),'false_safe_rate':fp/(fp+tn),
            'potable_recall':tp/(tp+fn),'predicted_potable':int(tp+fp),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
rows=[]; cvrows=[]; selection=[]
for name,pipe in models.items():
    print('Fitting',name,flush=True)
    for method in ('raw','sigmoid'):
        model=clone(pipe) if method=='raw' else CalibratedClassifierCV(estimator=clone(pipe),method='sigmoid',cv=3,n_jobs=1)
        model.fit(Xtr,ytr)
        v=model.predict_proba(Xval)[:,1]; te=model.predict_proba(Xtest)[:,1]
        # False-safe-rate goal on validation, maximizing potable recall; ties prefer lower threshold
        thresholds=np.round(np.linspace(.05,1.,96),2)
        candidates=[(float(t),metrics(yval,v,t)) for t in thresholds]
        valid=[(t, m) for t,m in candidates if m['false_safe_rate']<=.05]
        t_opt=sorted(valid,key=lambda z:(-z[1]['potable_recall'],z[0]))[0][0] if valid else 1.
        selection.append({'model':name,'method':method,'validation_selected_threshold':t_opt,
                          **{'validation_'+k:v for k,v in metrics(yval,v,t_opt).items()}})
        for split,ys,probs in [('validation',yval,v),('test',ytest,te)]:
            for idx,threshold in enumerate([0.5,0.65,0.8,t_opt]):
                result=metrics(ys,probs,threshold)
                rows.append({'model':name,'calibration':method,'split':split,
                    'threshold_choice':'validation_selected' if idx==3 else f'fixed_{threshold}',
                    'threshold':threshold,'roc_auc':roc_auc_score(ys,probs),
                    'average_precision':average_precision_score(ys,probs),
                    'brier_score':brier_score_loss(ys,probs),
                    **result})
    cv=RepeatedStratifiedKFold(n_splits=3,n_repeats=2,random_state=42)
    scores=cross_validate(clone(pipe),Xa,ya,cv=cv,scoring={'roc_auc':'roc_auc','balanced_accuracy':'balanced_accuracy','neg_brier':'neg_brier_score'},n_jobs=1)
    cvrows.append({'model':name,'folds':6,'auc_mean':np.mean(scores['test_roc_auc']),
      'auc_std':np.std(scores['test_roc_auc'],ddof=1),'balanced_accuracy_mean':np.mean(scores['test_balanced_accuracy']),
      'balanced_accuracy_std':np.std(scores['test_balanced_accuracy'],ddof=1),'brier_mean':-np.mean(scores['test_neg_brier'])})
results=pd.DataFrame(rows); results.to_csv(p/'potability_threshold_results.csv',index=False)
pd.DataFrame(selection).to_csv(p/'potability_threshold_selection.csv',index=False)
pd.DataFrame(cvrows).to_csv(p/'potability_repeated_cv.csv',index=False)
print('\nSIZES',len(Xtr),len(Xval),len(Xtest),'class counts:',ytest.value_counts().to_dict())
print('\nTEST FIXED .5');print(results[(results.split=='test')&(results.threshold_choice=='fixed_0.5')][['model','calibration','roc_auc','brier_score','balanced_accuracy','false_safe_rate','potable_recall','fp','tp']].to_string(index=False))
print('\nTEST VALIDATION SELECTED');print(results[(results.split=='test')&(results.threshold_choice=='validation_selected')][['model','calibration','threshold','balanced_accuracy','false_safe_rate','potable_recall','fp','tp']].to_string(index=False))
print('\nCV');print(pd.DataFrame(cvrows).to_string(index=False))
