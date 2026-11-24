"""Reproducible exploratory potability baseline and CSV integrity check.
Not a water-safety certification. All preprocessing is fit on training folds only.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,average_precision_score,confusion_matrix,precision_score,recall_score,f1_score

water=pd.read_csv('/mnt/data/Water quality.csv')
pot=pd.read_csv('/mnt/data/water_potability.csv')
X=pot.drop(columns=['Potability']); y=pot['Potability']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={
 'Majority baseline':DummyClassifier(strategy='most_frequent'),
 'Logistic regression': Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('classifier',LogisticRegression(max_iter=1500,random_state=42))]),
 'Random forest':Pipeline([('imputer',SimpleImputer(strategy='median')),('classifier',RandomForestClassifier(n_estimators=250,min_samples_leaf=2,random_state=42,n_jobs=-1))])
}
rows=[]
for name,model in models.items():
 model.fit(Xtr,ytr)
 pred=model.predict(Xte)
 probability=model.predict_proba(Xte)[:,1]
 tn,fp,fn,tp=confusion_matrix(yte,pred,labels=[0,1]).ravel()
 rows.append(dict(model=name,training_rows=len(Xtr),test_rows=len(Xte),accuracy=accuracy_score(yte,pred),balanced_accuracy=balanced_accuracy_score(yte,pred),f1_potable=f1_score(yte,pred,zero_division=0),recall_potable=recall_score(yte,pred,zero_division=0),false_safe_count=fp,non_potable_test_count=tn+fp,false_safe_rate=fp/(tn+fp),roc_auc=roc_auc_score(yte,probability),pr_auc_potable=average_precision_score(yte,probability)))
pd.DataFrame(rows).to_csv('/mnt/data/potability_provenance_baseline.csv',index=False)
report=[
 dict(dataset='Dhaka WQI',check='Sample count',value=str(len(water)),interpretation='Observed CSV rows'),
 dict(dataset='Dhaka WQI',check='WQI missing',value=str(water.WQI.isna().sum()),interpretation='No missing target'),
 dict(dataset='Dhaka WQI',check='Label reproducible from WQI thresholds 50/75',value=str(((np.where(water.WQI <50,'Good',np.where(water.WQI<75,'Poor','Very Poor')))==water.Pollution_Label).sum())+'/'+str(len(water)),interpretation='Perfect match, but publisher formula remains unverified'),
 dict(dataset='Dhaka WQI',check='COD equal to 70',value=str((water['COD\n(mg/L)']==70).sum()),interpretation='Repeated value needs provenance check'),
 dict(dataset='Dhaka WQI',check='Negative COD',value=str((water['COD\n(mg/L)']<0).sum()),interpretation='Invalid or unusual measurements require source clarification'),
 dict(dataset='Potability Kaggle',check='Sample count',value=str(len(pot)),interpretation='Observed CSV rows'),
 dict(dataset='Potability Kaggle',check='Rows missing at least one predictor',value=str(X.isna().any(axis=1).sum()),interpretation='Must impute in train fold only'),
 dict(dataset='Potability Kaggle',check='Missing target',value=str(y.isna().sum()),interpretation='No missing potability labels'),
 dict(dataset='Potability Kaggle',check='Positive potable labels',value=str((y==1).sum()),interpretation='Reported label, not confirmed independent lab standard'),
 dict(dataset='Potability Kaggle',check='Date/location/lab methodology metadata',value='Not present in CSV',interpretation='External or time/location validation impossible from this file'),
]
pd.DataFrame(report).to_csv('/mnt/data/dataset_target_provenance_audit.csv',index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
