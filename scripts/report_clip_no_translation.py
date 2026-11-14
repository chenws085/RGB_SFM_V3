#!/usr/bin/env python3
"""Matched centered-object CE vs deep-CLIP report, no translation conditions."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);data={};configs={};stats={};fig,axes=plt.subplots(1,2,figsize=(10,4))
for name in ['CE','CLIP']:
 folder=root/name;cfg=json.loads((folder/'config.json').read_text());configs[name]=cfg
 assert cfg['no_translation'] and json.loads((folder/'status.json').read_text())['status']=='done'
 ev=json.loads((folder/'scale_evaluation.json').read_text());assert ev['checkpoint_sha256']==hashlib.sha256((folder/'best.pth').read_bytes()).hexdigest();assert all(r['shift']==[0,0] for r in ev['conditions'])
 p=np.load(folder/'scale_predictions.npz');data[name]={'indices':p['source_indices'],'labels':p['labels']}
 for key,lengths in [('reference',[96]),('heldout',[80,112])]:
  data[name][key]=np.mean([p[f'length_{s}_shift_0_0'].argmax(1)==p['labels'] for s in lengths],axis=0)
 h=json.loads((folder/'history.json').read_text());stats[name]=dict(reference_accuracy=float(data[name]['reference'].mean()),heldout_accuracy=float(data[name]['heldout'].mean()),minimum_accuracy=min(r['accuracy'] for r in ev['conditions']),training_seconds=sum(r['seconds'] for r in h))
 axes[0].plot([r['length'] for r in ev['conditions']],[100*r['accuracy'] for r in ev['conditions']],marker='o',label=name)
 axes[1].plot([r['epoch'] for r in h],[r['loss'] for r in h],label=name)
for k in ['splits','seed','epochs','batch_size','model_args','pred_weight','region_weight','clip_stages','no_translation']:assert configs['CE'][k]==configs['CLIP'][k],k
assert np.array_equal(data['CE']['indices'],data['CLIP']['indices'])
rng=np.random.default_rng(42);effects={}
for key in ['reference','heldout']:
 d=data['CLIP'][key]-data['CE'][key];ci=np.quantile([d[rng.integers(len(d),size=len(d))].mean() for _ in range(1000)],[.025,.975]);effects[key]=dict(mean=float(d.mean()),source_bootstrap_95ci=ci.tolist())
result=dict(statistics=stats,clip_minus_CE=effects,limitations='Single training seed; exploratory validation used for checkpoint selection; synthetic solid background with contour cutouts; scale augmentation remains, translation disabled. CIs omit training-seed variation.')
(root/'results.json').write_text(json.dumps(result,indent=2));axes[0].set(xlabel='Centered object long edge (px)',ylabel='Validation accuracy (%)');axes[1].set(xlabel='Epoch',ylabel='Training loss (objectives differ)')
for ax in axes:ax.legend();ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150)
lines=['# 無平移需求：CLIP訊號對訓練的幫助','','兩組皆全程物件置中，不做訓練或評估平移。保留物件尺度64/96/128增強及合成純色背景。CE only與CE+深層Stage3/4 CLIP cosine0.2，共同初始化／seed42／6902train／1768validation／100epochs／batch16。','', '| 組別 | 置中96 accuracy | 未見80/112置中平均 | 最差尺度accuracy |','|---|---:|---:|---:|']
for n,r in stats.items():lines.append(f"| {n} | {r['reference_accuracy']:.2%} | {r['heldout_accuracy']:.2%} | {r['minimum_accuracy']:.2%} |")
lines+=['','CLIP−CE配對來源bootstrap差值（百分點）：']
for key,r in effects.items():lines.append(f"- {key}: {100*r['mean']:+.2f}，95% CI [{100*r['source_bootstrap_95ci'][0]:+.2f}, {100*r['source_bootstrap_95ci'][1]:+.2f}]")
verdict='本次單seed對照支持CLIP提高置中分類準確率。' if effects['reference']['source_bootstrap_95ci'][0]>0 else ('本次單seed對照顯示置中分類下降。' if effects['reference']['source_bootstrap_95ci'][1]<0 else '本次對照未提供置中分類有明確增益的證據。')
lines+=['','## 判讀','',verdict,'置中原尺度為主要指標，未見尺度為次要指標。Best checkpoint依共同置中80 validation選取；非untouched final test。沒有自然背景原圖泛化或多seed證據，CI不含seed變異。','','![Comparison](comparison.png)'];(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
