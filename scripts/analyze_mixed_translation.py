#!/usr/bin/env python3
"""Build an independent, paired-source analysis of the completed translation grid."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
MODELS=['baseline','aligned_8_16_32','augment_1_7']
NAMES={'baseline':'Baseline','aligned_8_16_32':'A：8/16/32 增強','augment_1_7':'T1：1–7 增強'}
GROUPS={'non_aligned_1_7':list(range(1,8)),'patch_aligned_8_16_32':[8,16,32]}
CATS=['all_selected','airplanes','chandelier','brain']
METRICS=['accuracy','prediction_consistency','cluster_agreement','token_cosine','spherical_margin']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(v):return f'{v*100:.2f}%'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'plots/kmeans/caltech_mixed_translation_1020');a=p.parse_args();out=a.output_dir
 frames=[];summaries=[];manifests=[];verification={}
 for name in MODELS:
  d=out/name;manifest=json.loads((d/'run_manifest.json').read_text());manifests.append(manifest)
  assert manifest['status']=='complete' and manifest['verified_pairs']==33128
  checkpoint_dir=ROOT/'runs/train'/dict(baseline='mergingvit_caltech101_letterbox160',aligned_8_16_32='mergingvit_caltech101_aligned_translation_1014',augment_1_7='mergingvit_caltech101_non_aligned_restart_1015')[name]
  assert sha(checkpoint_dir/'MergingViT_best.pth')==manifest['checkpoint_sha256']
  training_config=json.loads((checkpoint_dir/'run_config.json').read_text())
  for key in ['train_indices','valid_indices']:
   if key in training_config:assert sorted(training_config[key])==sorted(manifest[key])
  assert sha(d/'spherical_centers.npz')==manifest['centers_sha256']
  for file,value in manifest['artifacts_sha256'].items():assert sha(d/file)==value
  df=pd.read_csv(d/'paired_metrics.csv');assert len(df)==33128
  assert not df.duplicated(['source_index','stage','shift_pixels','direction']).any()
  assert set(df.shift_pixels)=={0,1,2,3,4,5,6,7,8,16,32}
  for s in [0,1,2,3]:
   for shift in [0,1,2,3,4,5,6,7,8,16,32]:
    sub=df[(df.stage==s)&(df.shift_pixels==shift)]
    assert len(sub)==202*(1 if shift==0 else 4) and sub.source_index.nunique()==202
  z=df[df.shift_pixels==0];assert z.same_cluster.all() and z.prediction_consistent.all() and np.allclose(z.cosine,1,atol=1e-5)
  # Check classifier values are exactly repeated across readout stages.
  assert df.groupby(['source_index','shift_pixels','direction']).moved_prediction.nunique().max()==1
  frames.append(df);summaries.append(pd.read_csv(d/'summary.csv'))
  old=ROOT/'plots/kmeans/caltech_non_aligned_restart_1015'/name
  oldmanifest=json.loads((old/'run_manifest.json').read_text())
  assert oldmanifest['checkpoint_sha256']==manifest['checkpoint_sha256']
  prior=pd.read_csv(old/'summary.csv');now=summaries[-1]
  merged=prior.merge(now,on=['model','category','stage','shift_pixels'],suffixes=('_old','_new'))
  verification[name]={'rows':len(df),'checkpoint_matches_historical':True,'small_shift_max_abs_delta':{k:float(np.max(np.abs(merged[k+'_old']-merged[k+'_new']))) for k in METRICS}}
 for field in ['evaluation_indices','train_indices','valid_indices','data_fingerprint','extractor_sha256','coordinate_sha256','shifts']:
  assert all(m[field]==manifests[0][field] for m in manifests)
 full=pd.concat(frames,ignore_index=True);summary=pd.concat(summaries,ignore_index=True);summary.to_csv(out/'all_summary.csv',index=False)
 group_rows=[];image_rows=[];occupancy=[]
 for name in MODELS:
  d=full[full.model==name]
  for group,shifts in GROUPS.items():
   moved=d[d.shift_pixels.isin(shifts)]
   for cat in CATS:
    sub=moved if cat=='all_selected' else moved[moved.category==cat]
    for stage in range(4):
     q=sub[sub.stage==stage]
     vals={'accuracy':q.moved_correct.mean(),'prediction_consistency':q.prediction_consistent.mean(),'cluster_agreement':q.same_cluster.mean(),'token_cosine':q.cosine.mean(),'spherical_margin':q.spherical_margin.mean()}
     counts=q.moved_cluster.value_counts();prob=counts/counts.sum()
     group_rows.append(dict(model=name,group=group,category=cat,stage=stage,source_images=q.source_index.nunique(),pairs=len(q),**vals,used_clusters=len(counts),cluster_entropy=float(-(prob*np.log(prob)).sum()),max_cluster_share=prob.max(),padding_contact=q.padding_contact.mean(),token_changed=q.token_changed.mean()))
   for (source,cat,stage),q in moved.groupby(['source_index','category','stage']):
    image_rows.append(dict(model=name,group=group,source_index=source,category=cat,stage=stage,accuracy=q.moved_correct.mean(),prediction_consistency=q.prediction_consistent.mean(),cluster_agreement=q.same_cluster.mean(),token_cosine=q.cosine.mean()))
 groupdf=pd.DataFrame(group_rows);groupdf.to_csv(out/'group_summary.csv',index=False)
 images=pd.DataFrame(image_rows);images.to_csv(out/'per_image_metrics.csv',index=False)
 contrasts=[];rng=np.random.default_rng(42);source_ids=sorted(full.source_index.unique());bootstrap=rng.integers(0,202,size=(5000,202))
 for group in GROUPS:
  for stage in range(4):
   for metric in ['accuracy','prediction_consistency','cluster_agreement','token_cosine']:
    wide=images[(images.group==group)&(images.stage==stage)].pivot(index='source_index',columns='model',values=metric).reindex(source_ids)
    for before,after in [('baseline','aligned_8_16_32'),('baseline','augment_1_7'),('aligned_8_16_32','augment_1_7')]:
     delta=(wide[after]-wide[before]).to_numpy();ci=np.quantile(delta[bootstrap].mean(1),[.025,.975])
     contrasts.append(dict(group=group,stage=stage,metric=metric,before=before,after=after,delta=delta.mean(),ci_low=ci[0],ci_high=ci[1],bootstrap_replicates=5000,source_images=202))
 contrast=pd.DataFrame(contrasts);contrast.to_csv(out/'paired_contrasts.csv',index=False)
 direction=full.groupby(['model','category','stage','shift_pixels','direction']).agg(pairs=('source_index','size'),accuracy=('moved_correct','mean'),prediction_consistency=('prediction_consistent','mean'),cluster_agreement=('same_cluster','mean'),token_cosine=('cosine','mean')).reset_index();direction.to_csv(out/'direction_summary.csv',index=False)
 contingency=full[full.shift_pixels>0].groupby(['model','stage','shift_pixels','same_cluster','prediction_consistent']).size().reset_index(name='pairs');contingency.to_csv(out/'cluster_prediction_contingency.csv',index=False)
 # Plots use categorical x positions, to keep all measured shifts legible.
 shifts=[0,1,2,3,4,5,6,7,8,16,32];colors=['#555555','#1479b8','#d45a26']
 fig,axes=plt.subplots(2,2,figsize=(13,8))
 for ax,cat in zip(axes.ravel(),CATS):
  for model,color in zip(MODELS,colors):
   q=summary[(summary.model==model)&(summary.category==cat)&(summary.stage==0)].set_index('shift_pixels').loc[shifts]
   ax.plot(range(len(shifts)),q.accuracy*100,'o-',label=model,color=color)
  ax.axvline(7.5,color='#aaaaaa',ls='--');ax.set_xticks(range(len(shifts)),shifts);ax.set_title(cat);ax.set_ylabel('Accuracy (%)');ax.set_xlabel('Shift (px; categorical spacing)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'classification_comparison.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(13,8))
 for stage,ax in enumerate(axes.ravel()):
  for model,color in zip(MODELS,colors):
   q=summary[(summary.model==model)&(summary.category=='all_selected')&(summary.stage==stage)].set_index('shift_pixels').loc[shifts]
   ax.plot(range(len(shifts)),q.token_cosine,'o-',label=model,color=color)
  ax.axvline(7.5,color='#aaaaaa',ls='--');ax.set_xticks(range(len(shifts)),shifts);ax.set_title(f'Stage {stage}, grid {8*2**stage}px');ax.set_ylim(0,1.02);ax.set_ylabel('Token cosine');ax.set_xlabel('Shift (px; categorical spacing)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'token_cosine_comparison.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(13,8))
 for stage,ax in enumerate(axes.ravel()):
  for model,color in zip(MODELS,colors):
   q=summary[(summary.model==model)&(summary.category=='all_selected')&(summary.stage==stage)].set_index('shift_pixels').loc[shifts]
   ax.plot(range(len(shifts)),q.cluster_agreement,'o-',label=model,color=color)
  ax.axvline(7.5,color='#aaaaaa',ls='--');ax.set_xticks(range(len(shifts)),shifts);ax.set_title(f'Stage {stage}, grid {8*2**stage}px');ax.set_ylim(0,1.02);ax.set_ylabel('Cluster agreement');ax.set_xlabel('Shift (px; categorical spacing)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'cluster_agreement_comparison.png',dpi=160);plt.close(fig)
 historical=pd.read_csv(ROOT/'plots/kmeans/caltech_aligned_translation_1014/translation_summary.csv')
 for name,oldname in [('baseline','baseline'),('aligned_8_16_32','augmented')]:
  old=historical[(historical.model==oldname)&(historical.stratum=='all')].groupby(['stage','shift_pixels'])[['classification_accuracy','prediction_consistency','matched_token_cosine']].mean().reset_index()
  now=summary[(summary.model==name)&(summary.category=='airplanes')]
  check=old.merge(now,on=['stage','shift_pixels'],suffixes=('_old','_new'))
  delta_accuracy=float(np.max(np.abs(check.classification_accuracy-check.accuracy)))
  delta_consistency=float(np.max(np.abs(check.prediction_consistency_old-check.prediction_consistency_new)))
  delta_cosine=float(np.max(np.abs(check.matched_token_cosine-check.token_cosine)))
  assert delta_accuracy<1e-12 and delta_consistency<1e-12 and delta_cosine<1e-5
  verification[name]['aligned_historical_airplanes']={'accuracy_max_delta':delta_accuracy,'prediction_consistency_max_delta':delta_consistency,'token_cosine_max_delta':delta_cosine,'note':'Cluster assignments intentionally not compared: Euclidean versus spherical.'}
 write_json(out/'verification.json',verification)
 def g(model,group,stage=0,cat='all_selected'):
  return groupdf[(groupdf.model==model)&(groupdf.group==group)&(groupdf.stage==stage)&(groupdf.category==cat)].iloc[0]
 lines=['# Caltech-101 對齊／非對齊平移混合分析','',f'更新：{datetime.now(timezone.utc).isoformat()}。執行位置為遠端 remote140165：`/home/ref/RGB_SFM_V3`。','',
 '## 問題與比較範圍','',
 '補齊三個既有 checkpoint 在原位、1–7 px 非對齊小位移、8／16／32 px 對齊初始 patch 位移的完整交叉評估。這裡「混合」指兩類測試條件的合併分析；沒有訓練同時混合兩種位移的新模型。原 epoch-10 暫停模型不納入完成模型比較。','',
 '## 模型、資料與統一方法','',
 '三模型使用 seed-42 split：training 6,907、validation 1,770；分析固定 validation airplanes 160／chandelier 22／brain 20，共 202 張。三模型本輪全部在同一遠端環境重新推論與 fitting，沒有直接拼接歷史 Euclidean 群指派。T1 從本機共享 inbox 下載至遠端並核對 SHA256，不代表本機專案已更新。','',
 table(['模型','訓練平移','最佳 epoch','原位全類別 validation accuracy','checkpoint SHA256'],[[NAMES[n],shift,epoch,acc,'`'+m['checkpoint_sha256']+'`'] for n,shift,epoch,acc,m in zip(MODELS,['無新增平移','原位或四方向 × 8/16/32 px','原位或四方向 × 1–7 px'],[66,84,78],['62.09%','65.71%','68.36%'],manifests)]),'',
 '160-in-224；完整影像整數平移，最大 32 px，不做平移插值或裁切來源影像。使用修正 annotation bbox offset 後的輪廓點平均中心，取各 stage 對應中心 token。每模型、每 stage 只用未平移全類別 training fitting 一個固定 codebook：FAISS spherical K-means，K=31、niter=100、nredo=50、seed=42。L2-normalized features、每輪 centroid 正規化；margin 為 top-1 減 top-2 cosine。不同模型 cluster ID 不對應。','',
 '每模型 202 × (1 + 4×10) = 8,282 個影像條件；四 stage 共 33,128 筆配對。三模型合計 99,384 筆 token 配對。沒有執行實際 64 px 位移。','',
 table(['位移','Stage 0：8 px','Stage 1：16 px','Stage 2：32 px','Stage 3：64 px'],[['1–7 px','不對齊','不對齊','不對齊','不對齊'],['8 px','對齊','不對齊','不對齊','不對齊'],['16 px','對齊','對齊','不對齊','不對齊'],['32 px','對齊','對齊','對齊','不對齊']]),'',
 '## 三類合併分類結果','',
 '各方向等權；兩組各自對位移量等權。合併 accuracy 依影像數加權，飛機占 160/202；macro accuracy 為三類 accuracy 等權，避免只看飛機主導的合併值。','']
 for group,label in [('non_aligned_1_7','非對齊：1–7 px 平均'),('patch_aligned_8_16_32','對齊初始 patch：8／16／32 px 平均')]:
  rows=[]
  for name in MODELS:
   r=g(name,group);orig=summary[(summary.model==name)&(summary.category=='all_selected')&(summary.stage==0)&(summary.shift_pixels==0)].iloc[0].accuracy
   macro=np.mean([g(name,group,cat=c).accuracy for c in CATS[1:]])
   rows.append([NAMES[name],pct(orig),pct(r.accuracy),pct(macro),pct(r.prediction_consistency),f'{(r.accuracy-orig)*100:+.2f} pp'])
  lines += ['### '+label,'',table(['模型','原位 accuracy','平移 accuracy','三類 macro accuracy','預測一致率','相對原位變化'],rows),'']
 lines+=['### 兩組等權的混合分數','', '先分別計算非對齊與對齊組平均，再各給 50% 權重；此權重是分析約定，不代表真實使用時的位移分布。不可把十個位移直接平均後當成兩組等權。','',table(['模型','混合 accuracy','混合預測一致率'],[[NAMES[n],pct(np.mean([g(n,k).accuracy for k in GROUPS])),pct(np.mean([g(n,k).prediction_consistency for k in GROUPS]))] for n in MODELS]),'',
 '## 完整逐位移分類表','']
 for cat in CATS:
  rows=[]
  for shift in shifts:
   vals=[]
   for n in MODELS:
    r=summary[(summary.model==n)&(summary.category==cat)&(summary.stage==0)&(summary.shift_pixels==shift)].iloc[0]
    vals.extend([pct(r.accuracy),pct(r.prediction_consistency)])
   rows.append([str(shift)]+vals)
  lines += ['### '+cat,'',table(['位移 px','B accuracy','B 一致率','A accuracy','A 一致率','T1 accuracy','T1 一致率'],rows),'']
 lines += ['![分類比較](classification_comparison.png)','', '## 各 stage 表徵穩定性','']
 for group,label in [('non_aligned_1_7','非對齊 1–7 px'),('patch_aligned_8_16_32','對齊初始 patch 8／16／32 px')]:
  rows=[]
  for stage in range(4):
   for n in MODELS:
    r=g(n,group,stage)
    rows.append([stage,NAMES[n],pct(r.cluster_agreement),f'{r.token_cosine:.4f}',f'{r.spherical_margin:.4f}',int(r.used_clusters),f'{r.cluster_entropy:.4f}',pct(r.max_cluster_share)])
  lines += ['### '+label,'',table(['Stage','模型','群一致率','Cosine','Spherical margin','群數','群熵','最大群占比'],rows),'']
 lines += ['群數／熵／最大群占比是在該模型、stage、整組位移方向的 pooled assignments 計算；不是逐位移群熵的平均。表徵表均為三類合併。逐位移、各類別、各 stage 完整值見 `all_summary.csv`。','',
 '![Token cosine 比較](token_cosine_comparison.png)','', '![群一致率比較](cluster_agreement_comparison.png)','', '## 影像層級配對差值與不確定性','',
 '同一影像先平均同組所有位移與方向，再將模型差值按來源影像 paired bootstrap；5,000 次、seed 42、202 張影像，報告 percentile 95% CI。沒有把重複位移、方向或 stage 當成獨立樣本。CI 僅描述本批影像抽樣不確定性，不包含重訓 seed；下列為描述性區間，未做多重比較校正。','']
 rows=[]
 for _,r in contrast[(contrast.stage==0)&(contrast.metric.isin(['accuracy','prediction_consistency']))].iterrows():
  rows.append([r.group,NAMES[r.after]+' − '+NAMES[r.before],r.metric,f'{r.delta*100:+.2f}',f'[{r.ci_low*100:+.2f}, {r.ci_high*100:+.2f}]'])
 lines += [table(['測試組','模型差值','指標','差值 pp','95% CI pp'],rows),'',
 '四 stage 的群一致率／cosine 配對區間見 `paired_contrasts.csv`；分類在四 stage 重複，不應解讀為四次獨立驗證。','',
 '## 驗證、歷史關係與限制','',
 '已驗證：三模型 train/validation/source IDs、資料位元組 fingerprint 與前處理程式 hash 相同；train/validation 分離；每個位移／stage 的筆數與唯一鍵完整；0 px cosine=1、群指派及預測 identity；中心位移精確等於要求值；centroid 單位 norm；summary/CSV/圖與 manifest SHA256 相符。每模型 codebook 在所有位移間固定。','',
 '歷史 0–7 px spherical 結果的 checkpoint hash 已對上；本輪與歷史 summary 的逐指標最大差異保存在 `verification.json`，以區分跨硬體浮點及 fitting 敏感性。10/14 對齊報告仍是原 Euclidean 結果，本報告的新 spherical 群指派不覆寫或重命名舊實驗。','',
 '對齊組和非對齊組的位移大小不同，沒有以相同位移量隔離 alignment 的因果效果。7 與 8 px 鄰近比較也會同時改變位移量與內容涵蓋。對齊某 stage 也不代表對齊所有 stage，亦不保證感受野完全相同。單中心 token 不能代表整圖；高分類一致率、較高 cosine 或群一致率不構成人類語意概念或嚴格平移不變性的證明。','',
 '只有一個 training seed，validation 已參與選模，三類為探索性子集。對 A 而言 8/16/32 是訓練內位移；對 T1 而言 1–7 是訓練內位移、8/16/32 是未見位移量；反向亦然。跨範圍結果僅限本次四方向與影像，不是普遍泛化保證。','',
 '## 重現與產物','',
 '```bash\n# 在 remote140165 專案執行；需三模型權重、Caltech-101 與既有 .venv\nMIXED_DEVICE=cuda:1 bash scripts/run_mixed_translation_1020.sh\n.venv/bin/python scripts/analyze_mixed_translation.py\n```','',
 '執行 log/status：`overnight_logs/mixed_translation_1020/`。每模型目錄含 `paired_metrics.csv`、`summary.csv`、`run_manifest.json`、codebook 與 feature caches。彙總：`all_summary.csv`、`group_summary.csv`、`direction_summary.csv`、`per_image_metrics.csv`、`paired_contrasts.csv`、`cluster_prediction_contingency.csv`、`verification.json`。原模型與 cache 不納入 Git。','']
 def value(model,shift,stage,key):
  return float(summary[(summary.model==model)&(summary.category=='all_selected')&(summary.stage==stage)&(summary.shift_pixels==shift)].iloc[0][key])
 key_results=['## 主要發現','',
  f"T1 在非對齊 1–7 px 的三類合併 accuracy 為 {pct(g('augment_1_7','non_aligned_1_7').accuracy)}，高於 Baseline {pct(g('baseline','non_aligned_1_7').accuracy)} 與 A {pct(g('aligned_8_16_32','non_aligned_1_7').accuracy)}；但對齊初始 patch 的 8/16/32 px 平均，A 為 {pct(g('aligned_8_16_32','patch_aligned_8_16_32').accuracy)}，高於 T1 的 {pct(g('augment_1_7','patch_aligned_8_16_32').accuracy)}。兩組等權時 A 最佳。資料支持增強效果與訓練位移範圍有關，尚不支持小位移增強取代大位移增強。",'']
 lines[4:4]=key_results
 interpretation=['## 綜合解讀與後續判斷','',
  '### 1. 小位移改善可以延伸至 8 px，但不能延伸解讀到 32 px','',
  f"T1 從 7 px 到未見的 8 px，accuracy 為 {pct(value('augment_1_7',7,0,'accuracy'))} → {pct(value('augment_1_7',8,0,'accuracy'))}，顯示鄰近位移仍穩健；到 16 px 降為 {pct(value('augment_1_7',16,0,'accuracy'))}，32 px 降為 {pct(value('augment_1_7',32,0,'accuracy'))}。A 在 32 px 為 {pct(value('aligned_8_16_32',32,0,'accuracy'))}。T1 對整組 8/16/32 的平均改善不能掩蓋 32 px 的明顯失效。",'',
  '### 2. 對齊帶來局部表徵回升，不保證分類同步回升','',
  f"Baseline 的 Stage 0 cosine 在 7 → 8 px 由 {value('baseline',7,0,'token_cosine'):.4f} 回升到 {value('baseline',8,0,'token_cosine'):.4f}，但合併 accuracy 為 {pct(value('baseline',7,0,'accuracy'))} → {pct(value('baseline',8,0,'accuracy'))}。8 px 只對齊 Stage 0，後面 stage 仍不對齊；因此不能用單層高 cosine 推論分類不變。這與網格相位敏感性的解讀一致，但尚未隔離位置編碼、merging 或 attention 的因果作用。",'',
  '### 3. 固定 32 px 時，最後一層仍呈現明顯變動','',
  table(['模型','Stage 2 cosine：網格32','Stage 3 cosine：網格64','Stage 3 群一致率','32 px accuracy'],[[NAMES[n],f"{value(n,32,2,'token_cosine'):.4f}",f"{value(n,32,3,'token_cosine'):.4f}",pct(value(n,32,3,'cluster_agreement')),pct(value(n,32,0,'accuracy'))] for n in MODELS]),'',
  '固定同一個 32 px 位移，Stage 2 對齊而 Stage 3 不對齊；三模型的最後一層 cosine 都下降。A 的分類仍較穩定，說明中心 token 的局部穩定性與全圖分類輸出並非一對一。這個跨層差異也同時涉及層深與 token 涵蓋範圍，不能單獨歸因網格對齊。','',
  '### 4. 群一致率需與分群集中程度一起解讀','',
  f"非對齊組 T1 的 Stage 2 最大群占比為 {pct(g('augment_1_7','non_aligned_1_7',2).max_cluster_share)}，Baseline 為 {pct(g('baseline','non_aligned_1_7',2).max_cluster_share)}；T1 的群熵也較低。因此較高群一致率可能同時受到分群集中程度影響。相對地，cosine 和分類 accuracy 的同步提高提供額外證據。現有群數未顯示單群完全坍縮，但不能據此證明語意品質提升。",'',
  '### 5. 下一個實驗應檢驗混合訓練，而非假設已具備跨範圍穩健性','',
  '如果目標同時涵蓋 1–7 與 8/16/32 px，現有結果支持將「同時包含小位移與大位移的增強」列為下一個候選，並用相同訓練預算、明確的小／大位移取樣權重及多個 seed 驗證；本輪沒有執行此訓練。另需獨立 held-out 資料與更廣類別，才能評估是否可推廣。','']
 lines[lines.index('## 驗證、歷史關係與限制'):lines.index('## 驗證、歷史關係與限制')]=interpretation
 overall=[]
 for name in MODELS:
  d=full[(full.model==name)&(full.stage==0)]
  overall.append(dict(model=name,original=d[d.shift_pixels==0].moved_correct.mean(),all_10_shifts=d[d.shift_pixels>0].moved_correct.mean(),groups_50_50=np.mean([g(name,k).accuracy for k in GROUPS]),original_plus_40_transforms=d.moved_correct.mean()))
 pd.DataFrame(overall).to_csv(out/'overall_accuracy.csv',index=False)
 section=['## Overall accuracy：所有已測條件合併','',
  '以下仍是固定三類202張影像，非101類平移結果。十個非零位移量（1–8/16/32）及四方向等權，共每模型8,080個平移條件；非對齊組因此占70%、對齊組占30%。另列兩組各50%的分數；含原位欄則每張影像的原位與40個平移條件各占1/41。這些是同來源影像的重複測量，不是8,080張獨立影像。','',
  table(['模型','原位','所有平移等權','兩組各50%','原位+40平移條件等權'],[[NAMES[r['model']],pct(r['original']),pct(r['all_10_shifts']),pct(r['groups_50_50']),pct(r['original_plus_40_transforms'])] for r in overall]),'']
 position=lines.index('## 完整逐位移分類表');lines[position:position]=section
 matched_rows=[];matched_records=[]
 for stage,shift in [(0,8),(1,16),(2,32)]:
  aligned=summary[(summary.category=='all_selected')&(summary.stage==stage)&(summary.shift_pixels==shift)].set_index('model')
  nonaligned=groupdf[(groupdf.category=='all_selected')&(groupdf.stage==stage)&(groupdf.group=='non_aligned_1_7')].set_index('model')
  for label,frame,model in [(f'對齊增強 → 對齊 {shift} px',aligned,'aligned_8_16_32'),('非對齊增強 → 非對齊 1–7 px 平均',nonaligned,'augment_1_7')]:
   cells=[]
   record=dict(stage=stage,condition=label,model=model)
   for key in ['cluster_agreement','accuracy','token_cosine']:
    before=float(frame.loc['baseline',key]);after=float(frame.loc[model,key]);delta=after-before
    record.update({key+'_before':before,key+'_after':after,key+'_delta':delta})
    cells.append(f'{before:.4f} → {after:.4f}（{delta:+.4f}）' if key=='token_cosine' else f'{pct(before)} → {pct(after)}（{delta*100:+.2f} pp）')
   matched_rows.append([stage,label]+cells);matched_records.append(record)
 pd.DataFrame(matched_records).to_csv(out/'matched_training_test_comparison.csv',index=False)
 section=['## 配對比較：對齊增強→對齊測試／非對齊增強→非對齊測試','',
  '各列為相同測試條件下 Baseline → 對應增強模型，括號為增強減Baseline。對齊列依stage選8/16/32 px；非對齊列固定1–7 px平均。統一三類202張影像、四方向平均與spherical codebooks。分類accuracy來自完整模型輸出，不是該stage的分類probe；非對齊列重複的accuracy是同一結果。','',
  '\n\n'.join('### '+title+'\n\n'+table(['Stage','訓練→測試條件','增強前','增強後','改善量'],[[r['stage'],r['condition'],f"{r[key+'_before']:.4f}" if key=='token_cosine' else pct(r[key+'_before']),f"{r[key+'_after']:.4f}" if key=='token_cosine' else pct(r[key+'_after']),f"{r[key+'_delta']:+.4f}" if key=='token_cosine' else f"{r[key+'_delta']*100:+.2f} pp"] for r in matched_records]) for key,title in [('cluster_agreement','群一致率'),('accuracy','分類正確率'),('token_cosine','Token cosine 相似度')]),'',
  '在Stage 0–2，非對齊增強的群一致率與cosine絕對增益較大，但增強後的表徵穩定程度仍低於對齊增強在相應stage對齊測試下的結果。分類改善量則依位移量而異：對齊16/32 px的增益明顯大於非對齊1–7 px平均。兩類條件的起點、位移量與token涵蓋內容不同，不能將增益大小直接解讀為訓練方法的因果優劣。Stage 3沒有64 px對齊測試，故不納入此表。','']
 position=lines.index('## 三類合併分類結果');lines[position:position]=section
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 (out/'REPORT.txt').write_text('\n'.join(line.lstrip('#').strip() if line.startswith('#') else line for line in lines)+'\n')
 files=[p for p in out.rglob('*') if p.is_file() and p.suffix in ['.csv','.png','.md','.txt','.json'] and p.name!='analysis_manifest.json']
 write_json(out/'analysis_manifest.json',dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),bootstrap_seed=42,bootstrap_replicates=5000,models=MODELS,verified_pairs=99384,files={str(p.relative_to(out)):sha(p) for p in files}))
 print(groupdf[(groupdf.category=='all_selected')&(groupdf.stage==0)][['model','group','accuracy','prediction_consistency']].to_string(index=False))
 print('Analysis complete:',out/'REPORT.md')

if __name__=='__main__':main()
