"""Reproducible exploratory target and phosphate-dependence audit.
Uses user-provided Water quality.csv; no WQI formula is assumed.
"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

ROOT=Path('/mnt/data'); d=pd.read_csv(ROOT/'Water quality.csv')
feature_cols=['Temp (°C)','pH','DO (mg/L)','COD\n(mg/L)','Nitrate\n(mg/L)','Turbidity\n(NTU)','Phosphate\n(mg/L)','Rainfall (mm)']
ph='Phosphate\n(mg/L)'; y=d.WQI.to_numpy(); dates=pd.to_datetime(d.Date); stations=d.Latitude.round(5).astype(str)+','+d.Longitude.round(5).astype(str)
idx=np.arange(len(d)); tr,te=train_test_split(idx,test_size=.2,random_state=42)
ttr=np.flatnonzero(dates.dt.year<=2019); tte=np.flatnonzero(dates.dt.year>=2020)
gtr,gte=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(d,y,stations))
splits={'Random 80/20':(tr,te),'Future years 2020-21':(ttr,tte),'Unseen station':(gtr,gte)}
rows=[]
for split,(train,test) in splits.items():
 for mode,features in [('All 8',feature_cols),('Phosphate only',[ph]),('Without phosphate',[c for c in feature_cols if c!=ph]),('Dummy mean',[])]:
  if features:
   m=RandomForestRegressor(n_estimators=150,min_samples_leaf=3,random_state=42,n_jobs=-1)
   m.fit(d.iloc[train][features],y[train]);pred=m.predict(d.iloc[test][features])
  else:
   m=DummyRegressor(strategy='mean'); m.fit(np.zeros((len(train),1)),y[train]); pred=m.predict(np.zeros((len(test),1)))
  rows.append({'Split':split,'Features':mode,'Train n':len(train),'Test n':len(test),'MAE':round(mean_absolute_error(y[test],pred),4),'RMSE':round(root_mean_squared_error(y[test],pred),4),'R2':round(r2_score(y[test],pred),5)})
  if mode=='All 8' and split=='Random 80/20':
   perm=permutation_importance(m,d.iloc[test][features],y[test],n_repeats=5,random_state=42,n_jobs=1,scoring='neg_mean_absolute_error')
   pd.DataFrame({'Feature':features,'Permutation MAE increase':perm.importances_mean,'SD across permutations':perm.importances_std}).sort_values('Permutation MAE increase',ascending=False).to_csv(ROOT/'wqi_permutation_importance.csv',index=False)
res=pd.DataFrame(rows);res.to_csv(ROOT/'wqi_phosphate_ablation_results.csv',index=False)
print(res.to_string(index=False));print('\nCORRELATIONS P WQI',pearsonr(d[ph],d.WQI),spearmanr(d[ph],d.WQI))
print('\nLABEL MATCH',np.mean(np.select([d.WQI<50,d.WQI<75],['Good','Poor'],default='Very Poor')==d.Pollution_Label))
print('\nPHOSPHATE exact SAME value WQI range')
print(d.groupby(ph).WQI.agg(['count','min','max']).sort_values('count',ascending=False).head(10).to_string())
print('\nCOD anomaly',int((d['COD\n(mg/L)']==70).sum()),int((d['COD\n(mg/L)']<0).sum()))
