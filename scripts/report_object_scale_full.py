#!/usr/bin/env python3
"""Matched A/B/H report with paired source-level bootstrap, exploratory validation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
branches=['A','B','H'];configs={};data={};rows=[];rng=np.random.default_rng(42)
fig,ax=plt.subplots(figsize=(7,4))
for branch in branches:
 folder=args.run/branch
 assert json.loads((folder/'status.json').read_text())['status']=='done'
 cfg=json.loads((folder/'config.json').read_text());configs[branch]=cfg
 ev=json.loads((folder/'scale_evaluation.json').read_text());pred=np.load(folder/'scale_predictions.npz')
 assert ev['checkpoint_sha256']==hashlib.sha256((folder/'best.pth').read_bytes()).hexdigest()
 assert pred['source_indices'].tolist()==cfg['splits']['valid']
 data[branch]=dict(pred)
 correct=[]
 for length in [80,112]:
  for dx,dy in [(0,0),(7,-5)]:
   correct.append(pred[f'length_{length}_shift_{dx}_{dy}'].argmax(1)==pred['labels'])
 data[branch]['heldout_per_source']=np.mean(correct,axis=0)
 ref=pred['length_96_shift_0_0'].argmax(1)
 dominance=np.bincount(ref,minlength=101).max()/len(ref)
 rows.append(dict(branch=branch,reference_accuracy=float(np.mean(ref==pred['labels'])),
  heldout_interpolation_accuracy=float(np.mean(correct)),minimum_condition_accuracy=min(r['accuracy'] for r in ev['conditions']),
  reference_max_predicted_class_fraction=float(dominance)))
 subset=[r for r in ev['conditions'] if r['shift']==[7,-5]]
 ax.plot([r['length'] for r in subset],[100*r['accuracy'] for r in subset],marker='o',label=branch)
for branch in branches[1:]:
 for key in ['splits','seed','epochs','batch_size','model_args','source_sha256']:
  assert configs[branch][key]==configs['A'][key],(branch,key)
 assert np.array_equal(data[branch]['labels'],data['A']['labels'])
assert configs['A']['pred_weight']==configs['A']['region_weight']==0
assert configs['B']['pred_weight']==configs['H']['pred_weight']
assert configs['B']['region_weight']==configs['H']['region_weight']
assert not configs['A']['hybrid_merging'] and not configs['B']['hybrid_merging'] and configs['H']['hybrid_merging']
effects={}
for control in ['A','B']:
 d=data['H']['heldout_per_source']-data[control]['heldout_per_source']
 boot=np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(1000)])
 effects['H-'+control]=dict(mean=float(d.mean()),source_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist())
result=dict(rows=rows,paired_effects=effects,n_train=len(configs['A']['splits']['train']),n_valid=len(configs['A']['splits']['valid']),
  limitation='single training seed; exploratory validation used for checkpoint selection; CI excludes training-seed uncertainty; synthetic backgrounds and approximate contour masks')
(args.run/'comparison.json').write_text(json.dumps(result,indent=2))
ax.set(xlabel='Object long edge (pixels), shift (7,-5)',ylabel='Validation accuracy (%)');ax.legend();ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(args.run/'comparison.png',dpi=150)
lines=['# 完整來源物件縮放／平移 A/B/H 比較','','A：CE only；B：原 merging＋一致性；H：gated residual merging＋一致性。',
 '', '| 模型 | 原位96 accuracy | 未見內插尺度80/112平均 accuracy | 最差條件 accuracy | 最大預測類占比 |','|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['branch']} | {r['reference_accuracy']:.2%} | {r['heldout_interpolation_accuracy']:.2%} | {r['minimum_condition_accuracy']:.2%} | {r['reference_max_predicted_class_fraction']:.2%} |")
lines+=['',f"資料：{result['n_train']} train／{result['n_valid']} validation。共同來源、相同seed／預算；80/112平均包含原位及(7,-5)平移。",'', 'H 的配對來源影像 bootstrap 差值（百分點）：']
for control,r in effects.items():lines.append(f"- {control}: {100*r['mean']:.2f}，95% CI [{100*r['source_bootstrap_95ci'][0]:.2f}, {100*r['source_bootstrap_95ci'][1]:.2f}]")
lines+=['','單 seed、探索／選模 validation、近似輪廓與合成背景；不能推論自然場景或嚴格尺度不變。CI 不含訓練 seed 不確定性。','','![comparison](comparison.png)']
(args.run/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
