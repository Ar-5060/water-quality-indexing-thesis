"""Reproducible, descriptive target integrity audit. No WQI formula is assumed.
Run: python wqi_target_integrity_check.py --csv 'Water quality.csv' --out 'wqi_target_integrity_report.csv'
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--csv',default='Water quality.csv')
p.add_argument('--out',default='wqi_target_integrity_report.csv')
a=p.parse_args()
df=pd.read_csv(a.csv)
wqi=df['WQI']
phos=df['Phosphate\n(mg/L)']
cod=df['COD\n(mg/L)']
labels=np.select([wqi<50,wqi<75],['Good','Poor'],default='Very Poor')
years=pd.to_datetime(df['Date']).dt.year
rows=[
('total_rows',len(df),'count'),
('mean_wqi',wqi.mean(),'score'),
('wqi_min',wqi.min(),'score'),
('wqi_max',wqi.max(),'score'),
('label_match_using_observed_50_75_rule',int((labels==df['Pollution_Label']).sum()),'rows'),
('label_mismatch_using_observed_50_75_rule',int((labels!=df['Pollution_Label']).sum()),'rows'),
('negative_cod',int((cod<0).sum()),'rows'),
('cod_equal_70',int((cod==70).sum()),'rows'),
('phosphate_pearson_wqi',phos.corr(wqi),'correlation'),
('unique_phosphate',phos.nunique(),'unique values'),
('unique_wqi',wqi.nunique(),'unique values'),
('phosphate_le_0.25',int((phos<=0.25).sum()),'rows'),
('phosphate_le_0.25_wqi_below_50',int(((phos<=0.25)&(wqi<50)).sum()),'rows'),
('phosphate_gt_0.4',int((phos>0.4).sum()),'rows'),
('phosphate_gt_0.4_wqi_ge_75',int(((phos>0.4)&(wqi>=75)).sum()),'rows'),
]
for year, sub in df.assign(year=years).groupby('year'):
    rows.append((f'cod_equal_70_year_{year}',int((sub['COD\n(mg/L)']==70).sum()),f'of {len(sub)} rows'))
pd.DataFrame(rows,columns=['measure','value','unit']).to_csv(a.out,index=False)
print(pd.DataFrame(rows,columns=['measure','value','unit']).to_string(index=False))
print('\nIMPORTANT: Matching thresholds establishes association, NOT the documented label-generation procedure.')
print('WQI construction remains unverified until dataset authors provide formula, weights, and standards.')
