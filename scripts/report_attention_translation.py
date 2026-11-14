#!/usr/bin/env python3
"""Supplement frozen paired translation results with plots and diagnostic contrasts."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(sys.argv[1]);df=pd.read_csv(out/'summary.csv');df=df[df.category=='all'];agg=df.groupby(['model','stage','shift']).mean(numeric_only=True).reset_index()
fig,axes=plt.subplots(4,4,figsize=(16,13),sharex=True)
for s in range(4):
 for j,key in enumerate(['agreement','cosine','prediction_consistency','accuracy']):
  for n in ['global','window']:
   d=agg[(agg.model==n)&(agg.stage==s)];axes[s,j].plot(d['shift'],d[key],marker='.',label=n)
  axes[s,j].set(title=f'Stage {s}: {key}',ylim=(0,1.02),xticks=[0,4,8,16,32]);axes[s,j].grid(alpha=.2)
axes[0,0].legend();fig.tight_layout();fig.savefig(out/'translation_stability.png',dpi=160);plt.close(fig)
a=np.load(out/'paired.npz');boot=np.random.default_rng(42).integers(len(a['source_indices']),size=(2000,len(a['source_indices'])));metrics=['agreement','cosine','prediction_consistency','accuracy','final_cosine'];effects=[]
for shift in [1,4,8,16,32]:
 for s in range(4):
  v={n:np.mean([a[k] for k in a.files if k.startswith(f'{n}_{s}_{shift}_')],axis=0) for n in ['global','window']}
  for j,m in enumerate(metrics):
   d=v['window'][:,j]-v['global'][:,j];effects.append(dict(stage=s,shift=shift,metric=m,delta=float(d.mean()),ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist()))
(out/'effects.json').write_text(json.dumps(effects,indent=2))
lines=['','## 分類與方向診斷','|模型|原位160 accuracy|8px accuracy / consistency|16px accuracy / consistency|32px accuracy / consistency|','|---|---|---|---|---|']
for n in ['global','window']:
 d=agg[(agg.model==n)&(agg.stage==0)].set_index('shift');lines.append('|'+n+'|'+f"{d.loc[0,'accuracy']:.2%}"+'|'+ '|'.join(f"{d.loc[s,'accuracy']:.2%} / {d.loc[s,'prediction_consistency']:.2%}" for s in [8,16,32])+'|')
lines+=['','|模型|32px方向|accuracy|prediction consistency|final cosine|','|---|---|---|---|---|']
for _,r in df[(df.stage==0)&(df['shift']==32)].iterrows():lines.append(f'|{r.model}|{r.direction}|{r.accuracy:.2%}|{r.prediction_consistency:.2%}|{r.final_cosine:.4f}|')
lines+=['','![平移曲線](translation_stability.png)']
with (out/'REPORT.md').open('a') as f:f.write('\n'.join(lines))
# Fixed-grid window membership is diagnostic only; global has no local windows.
p=np.load(out/'global_original.npz')['points'];strata=[]
config=json.loads((out/'manifest.json').read_text()).get('attention_config',{}).get('window',{})
selected=config.get('attention_stages');selected=set(range(4) if selected is None else selected)
for s,grid in enumerate([28,14,7,4]):
 if config.get('attention_mode','window')!='window' or s not in selected:continue
 ws=min(config.get('window_size',7),grid);offset=config.get('window_offset',0) if ws<grid else 0
 stride=8*2**s
 for shift in [1,4,8,16,32]:
  for direction,(dx,dy) in {'left':(-shift,0),'right':(shift,0),'up':(0,-shift),'down':(0,shift)}.items():
   moved=p+[dx,dy];oldtoken=np.floor(p/stride).astype(int);newtoken=np.floor(moved/stride).astype(int)
   cross=np.any((oldtoken+offset)//ws!=(newtoken+offset)//ws,axis=1)
   for crossed in [False,True]:
    mask=cross==crossed
    if not mask.any():continue
    for n in ['global','window']:
     v=a[f'{n}_{s}_{shift}_{direction}'][mask]
     strata.append(dict(model=n,stage=s,shift=shift,direction=direction,cross_window=crossed,n=int(mask.sum()),agreement=float(v[:,0].mean()),cosine=float(v[:,1].mean())))
pd.DataFrame(strata).to_csv(out/'window_boundary_strata.csv',index=False)
checks=dict(sources=len(p),rows=len(df),finite=bool(all(np.isfinite(a[k]).all() for k in a.files)),identity=bool(all(np.allclose(a[f'{n}_{s}_0_original'][:,[0,1,2,4]],1,atol=1e-5) for n in ['global','window'] for s in range(4))),exact_one_token=bool(all(np.all(np.abs(np.floor((p+[shift,0])/(8*2**s))-np.floor(p/(8*2**s)))==[1,0]) for s,shift in enumerate([8,16,32]))))
assert checks['finite'] and checks['identity'] and checks['exact_one_token'];(out/'validation_checks.json').write_text(json.dumps(checks,indent=2))

import hashlib
m=json.loads((out/'manifest.json').read_text());m['artifacts_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.name!='manifest.json'};m['report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(out/'manifest.json').write_text(json.dumps(m,indent=2))
