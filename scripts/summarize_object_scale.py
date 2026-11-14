#!/usr/bin/env python3
"""Produce exploratory B/D comparison from actual checkpoint evaluations."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
fig,axes=plt.subplots(1,2,figsize=(11,4))
lines=['# 物件獨立縮放／平移 pilot 結果','','同一來源 cutout 貼在固定背景，單一 seed、每類 16 張訓練／4 張驗證。驗證集曾用於選 checkpoint，不是 final test。', '',
       '| 組別 | 原位長邊96 accuracy | 長邊80＋平移 accuracy | 最差條件 accuracy | 參考條件最大預測類占比 |',
       '|---|---:|---:|---:|---:|']
for branch in ['B','D']:
 folder=args.run/branch
 status=json.loads((folder/'status.json').read_text());assert status['status']=='done'
 report=json.loads((folder/'scale_evaluation.json').read_text());rows=report['conditions']
 predictions=np.load(folder/'scale_predictions.npz');pred=predictions['length_96_shift_0_0'].argmax(1)
 counts=np.bincount(pred,minlength=101);dominance=counts.max()/len(pred)
 ref=next(r for r in rows if r['length']==96 and r['shift']==[0,0])
 joint=next(r for r in rows if r['length']==80 and r['shift']==[7,-5])
 lines.append(f"| {branch} | {ref['accuracy']:.2%} | {joint['accuracy']:.2%} | {min(r['accuracy'] for r in rows):.2%} | {dominance:.2%} |")
 h=json.loads((folder/'history.json').read_text())
 axes[0].plot([r['epoch'] for r in h],[r['loss'] for r in h],label=branch)
 for shift,style in [([0,0],'-'),([7,-5],'--')]:
  subset=[r for r in rows if r['shift']==shift]
  axes[1].plot([r['length'] for r in subset],[100*r['accuracy'] for r in subset],style,marker='o',label=f'{branch} shift {shift}')
axes[0].set(xlabel='Epoch',ylabel='Paired training loss');axes[1].set(xlabel='Object long edge (pixels)',ylabel='Validation accuracy (%)')
for ax in axes:ax.legend();ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(args.run/'pilot_curves.png',dpi=150);plt.close(fig)
lines+=['','B：原 merging＋成對分類／預測／region 一致性；D：內容加權 merging＋相同損失。',
        '', '訓練長邊 64/96/128，位移每軸 -16..16；80/112 是未見離散尺度的內插，48/144 是尺度外插，測試 (7,-5) 位移本身屬訓練範圍。',
        '', '準確率偏低時不能只憑預測一致率宣稱不變性。本輪僅驗證可訓練性及初步學習；尚缺無一致性 A 組、充足訓練、多 seeds、精確部件標註與自然背景控制。',
        '', '![Pilot curves](pilot_curves.png)']
(args.run/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:10]))
