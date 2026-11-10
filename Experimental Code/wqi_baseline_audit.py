import csv, collections, datetime, json, math
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE='/mnt/data/Water quality.csv'
with open(FILE, encoding='utf-8-sig', newline='') as f: rows=list(csv.DictReader(f))
columns=['Temp (°C)','pH','DO (mg/L)','COD\n(mg/L)','Nitrate\n(mg/L)','Turbidity\n(NTU)','Phosphate\n(mg/L)','Rainfall (mm)']
X=np.array([[float(row[c]) for c in columns] for row in rows])
y=np.array([float(r['WQI']) for r in rows]); labels=np.array([r['Pollution_Label'] for r in rows]); dates=np.array([datetime.date.fromisoformat(r['Date']) for r in rows]); years=np.array([d.year for d in dates]); groups=np.array([r['Latitude']+','+r['Longitude'] for r in rows]);
label_correct=np.array([('Good' if v<50 else 'Poor' if v<75 else 'Very Poor')==lab for v,lab in zip(y,labels)])
cod=X[:,3]
idx=np.arange(len(rows)); train_random, test_random=train_test_split(idx,test_size=.2,random_state=42)
train_temporal=np.where(years<=2019)[0]; test_temporal=np.where(years>=2020)[0]
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
train_group,test_group=next(gss.split(X,y,groups=groups))
splits={'Random 80/20':(train_random,test_random),'Time train 2017-19 / test 2020-21':(train_temporal,test_temporal),'Unseen stations 80/20':(train_group,test_group)}
# Baseline only: use predeclared sensible hyperparameters, never tune on test; no location identifiers, date, WQI-derived label in X.
models={
 'Dummy mean':lambda:DummyRegressor(strategy='mean'),
 'Ridge':lambda:make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
 'Random Forest':lambda:RandomForestRegressor(n_estimators=200,min_samples_leaf=3,random_state=42,n_jobs=-1),
 'HistGradientBoost':lambda:HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=2,learning_rate=.06,random_state=42),
 'MLP':lambda:make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,16),alpha=5,learning_rate_init=.001,max_iter=450,random_state=42,early_stopping=True,n_iter_no_change=20))
}
results=[]
for name,(tr,te) in splits.items():
 for model_name,make in models.items():
  m=make();m.fit(X[tr],y[tr]);p=m.predict(X[te]);
  results.append({'split':name,'model':model_name,'train_n':len(tr),'test_n':len(te),'MAE':round(float(mean_absolute_error(y[te],p)),3),'RMSE':round(float(np.sqrt(mean_squared_error(y[te],p))),3),'R2':round(float(r2_score(y[te],p)),3)})
print('shape',len(rows),len(columns)+8)
print('WQI min median max',np.min(y),np.median(y),np.max(y))
print('Labels',dict(collections.Counter(labels)))
print('label threshold agreement',sum(label_correct),len(rows))
print('COD==70',sum(cod==70),'negative',sum(cod<0),'COD by year', {str(yr):{'n':sum(years==yr),'cod70':sum((years==yr)&(cod==70)),'neg':sum((years==yr)&(cod<0))} for yr in sorted(set(years))})
print('stations',len(set(groups)),'dates',len(set(dates)))
for name,(tr,te) in splits.items():
 print('SPLIT',name,'train',len(tr),'test',len(te),'train_stations',len(set(groups[tr])),'test_stations',len(set(groups[te])),'overlap',len(set(groups[tr]) & set(groups[te])),'train_years',sorted(set(years[tr])),'test_years',sorted(set(years[te])),'test target std',np.std(y[te]))
for r in results: print(r)
with open('/mnt/data/wqi_baseline_results.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
with open('/mnt/data/wqi_audit_summary.json','w') as f:
 json.dump({'n':len(rows),'features':columns,'label_threshold_agreement':int(sum(label_correct)),'COD70':int(sum(cod==70)),'CODnegative':int(sum(cod<0)),'results':results,'splits':{k:{'train':len(tr),'test':len(te),'station_overlap':len(set(groups[tr])&set(groups[te]))} for k,(tr,te) in splits.items()}},f,indent=2)
