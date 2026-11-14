#!/usr/bin/env python3
"""Figures, tabular evidence and Chinese text for stratified all-class translations."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_mixed_translation import ROOT, MODELS, NAMES, GROUPS, sha, table, pct, write_json

SHIFTS=[0,1,2,3,4,5,6,7,8,16,32]
COLORS=['#555555','#1479b8','#d45a26']
LABELS=['Baseline','Aligned augmentation (8/16/32px)','Non-aligned augmentation (1-7px)']
KEYS={'accuracy':'moved_correct','prediction_consistency':'prediction_consistent','cluster_agreement':'same_cluster','token_cosine':'cosine','spherical_margin':'spherical_margin'}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ROOT/'plots/kmeans/caltech_allclass_translation_1020');args=parser.parse_args();out=args.output_dir
 selection=json.loads((out/'selection_manifest.json').read_text());ids=sorted(i for row in selection['selection'] for i in row['source_indices']);classes=[r['category'] for r in selection['selection']];n=len(ids)
 assert len(classes)==101 and len(ids)==len(set(ids))==505 and all(r['selected']==5 for r in selection['selection'])
 frames=[];summaries=[];manifests=[];verification={}
 expected=n*41*4
 for model in MODELS:
  d=out/model;manifest=json.loads((d/'run_manifest.json').read_text());manifests.append(manifest)
  assert manifest['status']=='complete' and manifest['verified_pairs']==expected and manifest['evaluation_indices']==ids
  for file,digest in manifest['artifacts_sha256'].items():assert sha(d/file)==digest,file
  assert sha(d/'spherical_centers.npz')==manifest['centers_sha256']
  source=ROOT/manifest['codebook_source'];prior_manifest=json.loads((source/'run_manifest.json').read_text())
  assert sha(source/'run_manifest.json')==manifest['codebook_source_manifest_sha256']
  assert prior_manifest['checkpoint_sha256']==manifest['checkpoint_sha256']
  with np.load(source/'spherical_centers.npz') as old,np.load(d/'spherical_centers.npz') as now:
   for stage in range(4):assert np.array_equal(old[f'stage{stage}'],now[f'stage{stage}'])
  frame=pd.read_csv(d/'paired_metrics.csv');assert len(frame)==expected and set(frame.source_index)==set(ids)
  assert not frame.duplicated(['source_index','stage','shift_pixels','direction']).any()
  assert frame.groupby(['source_index','shift_pixels','direction']).moved_prediction.nunique().max()==1
  for stage in range(4):
   for shift in SHIFTS:
    q=frame[(frame.stage==stage)&(frame.shift_pixels==shift)]
    assert len(q)==n*(1 if shift==0 else 4) and q.source_index.nunique()==n
  zero=frame[frame.shift_pixels==0];assert zero.same_cluster.all() and zero.prediction_consistent.all() and np.allclose(zero.cosine,1,atol=1e-5)
  old=pd.read_csv(source/'paired_metrics.csv');common=old.merge(frame,on=['model','category','source_index','stage','shift_pixels','direction'],suffixes=('_old','_new'))
  assert len(common)==15*41*4
  for key in ['moved_prediction','original_prediction','same_cluster']:assert np.array_equal(common[key+'_old'],common[key+'_new'])
  cosine_delta=float(np.max(np.abs(common.cosine_old-common.cosine_new)));assert cosine_delta<1e-5
  verification[model]=dict(verified_pairs=len(frame),overlapping_historical_pairs=len(common),overlap_predictions_and_clusters_identical=True,max_overlap_cosine_delta=cosine_delta,frozen_centers_exactly_identical=True)
  frames.append(frame);summaries.append(pd.read_csv(d/'summary.csv'))
 for key in ['sampling','evaluation_indices','train_indices','valid_indices','data_fingerprint','extractor_sha256','coordinate_sha256']:
  assert all(m[key]==manifests[0][key] for m in manifests)
 full=pd.concat(frames,ignore_index=True);summary=pd.concat(summaries,ignore_index=True);summary.to_csv(out/'all_summary.csv',index=False)
 group_rows=[];image_rows=[]
 for model in MODELS:
  for group,shifts in GROUPS.items():
   moved=full[(full.model==model)&full.shift_pixels.isin(shifts)]
   for stage in range(4):
    stage_rows=moved[moved.stage==stage]
    for category in ['all_selected']+classes:
     q=stage_rows if category=='all_selected' else stage_rows[stage_rows.category==category]
     counts=q.moved_cluster.value_counts();prob=counts/counts.sum()
     group_rows.append(dict(model=model,group=group,stage=stage,category=category,source_images=q.source_index.nunique(),**{key:float(q[col].mean()) for key,col in KEYS.items()},used_clusters=len(counts),cluster_entropy=float(-(prob*np.log(prob)).sum()),max_cluster_share=float(prob.max())))
   for (source,category,stage),q in moved.groupby(['source_index','category','stage']):
    image_rows.append(dict(model=model,group=group,stage=stage,source_index=source,category=category,**{key:float(q[col].mean()) for key,col in KEYS.items()}))
 groups=pd.DataFrame(group_rows);groups.to_csv(out/'group_summary.csv',index=False)
 images=pd.DataFrame(image_rows);images.to_csv(out/'per_image_metrics.csv',index=False)
 # Stratified resampling keeps the same 101 classes, sampling images within each class.
 source_order=[i for item in selection['selection'] for i in item['source_indices']]
 rng=np.random.default_rng(42);draws=rng.integers(0,5,size=(5000,101,5))+np.arange(101)[None,:,None]*5;draws=draws.reshape(5000,n)
 contrasts=[]
 for group in GROUPS:
  for stage in range(4):
   for metric in ['accuracy','prediction_consistency','cluster_agreement','token_cosine']:
    wide=images[(images.group==group)&(images.stage==stage)].pivot(index='source_index',columns='model',values=metric).reindex(source_order)
    for before,after in [('baseline','aligned_8_16_32'),('baseline','augment_1_7'),('aligned_8_16_32','augment_1_7')]:
     delta=(wide[after]-wide[before]).to_numpy();ci=np.quantile(delta[draws].mean(1),[.025,.975])
     contrasts.append(dict(group=group,stage=stage,metric=metric,before=before,after=after,delta=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1])))
 contrast=pd.DataFrame(contrasts);contrast.to_csv(out/'paired_contrasts.csv',index=False)
 direction=full.groupby(['model','stage','shift_pixels','direction']).agg(pairs=('source_index','size'),accuracy=('moved_correct','mean'),prediction_consistency=('prediction_consistent','mean'),cluster_agreement=('same_cluster','mean'),token_cosine=('cosine','mean')).reset_index();direction.to_csv(out/'direction_summary.csv',index=False)
 curves=summary[summary.category=='all_selected'].copy();macro=summary[summary.category!='all_selected'].groupby(['model','stage','shift_pixels'])[['accuracy','cluster_agreement','token_cosine']].mean().reset_index()
 aligned=curves.merge(macro,on=['model','stage','shift_pixels'],suffixes=('','_macro'))
 for key in ['accuracy','cluster_agreement','token_cosine']:assert np.allclose(aligned[key],aligned[key+'_macro'],atol=1e-12)
 curves.to_csv(out/'plot_data.csv',index=False)
 # Categorical x axis displays each tested shift without implying even physical spacing.
 def decorate(ax):
  ax.axvline(7.5,color='#aaaaaa',ls='--');ax.set_xticks(range(11),SHIFTS);ax.set_xlabel('Shift (px; categorical spacing)');ax.grid(alpha=.2)
 fig,ax=plt.subplots(figsize=(9,5))
 for model,color,label in zip(MODELS,COLORS,LABELS):
  q=curves[(curves.model==model)&(curves.stage==0)].set_index('shift_pixels').loc[SHIFTS]
  ax.plot(range(11),q.accuracy*100,'o-',color=color,label=label)
 ax.set_title('101 classes, 5 validation images/class (n=505)');ax.set_ylabel('Accuracy (%; micro = macro)');ax.set_ylim(0,100)
 decorate(ax);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/'classification_comparison.png',dpi=180);plt.close(fig)
 for key,title,file in [('cluster_agreement','Cluster agreement (%)','cluster_agreement_comparison.png'),('token_cosine','Token cosine','token_cosine_comparison.png')]:
  fig,axes=plt.subplots(2,2,figsize=(13,8))
  for stage,ax in enumerate(axes.ravel()):
   for model,color,label in zip(MODELS,COLORS,LABELS):
    q=curves[(curves.model==model)&(curves.stage==stage)].set_index('shift_pixels').loc[SHIFTS]
    ax.plot(range(11),q[key]*(100 if key=='cluster_agreement' else 1),'o-',color=color,label=label)
   decorate(ax);ax.set_title(f'Stage {stage}, grid {8*2**stage}px');ax.set_ylabel(title);ax.set_ylim(0,102 if key=='cluster_agreement' else 1.02)
  axes[0,0].legend(fontsize=8);fig.suptitle('101 classes, 5 validation images/class (n=505)',fontsize=13);fig.tight_layout();fig.savefig(out/file,dpi=180);plt.close(fig)
 def g(model,group,stage=0):return groups[(groups.model==model)&(groups.group==group)&(groups.stage==stage)&(groups.category=='all_selected')].iloc[0]
 def v(model,shift,stage,key):return float(curves[(curves.model==model)&(curves.shift_pixels==shift)&(curves.stage==stage)].iloc[0][key])
 def ci(group,before,after,metric='accuracy',stage=0):return contrast[(contrast.group==group)&(contrast.before==before)&(contrast.after==after)&(contrast.metric==metric)&(contrast.stage==stage)].iloc[0]
 overall=[]
 for model in MODELS:
  d=full[(full.model==model)&(full.stage==0)]
  overall.append(dict(model=model,original_accuracy=v(model,0,0,'accuracy'),non_aligned_accuracy=g(model,'non_aligned_1_7').accuracy,patch_aligned_accuracy=g(model,'patch_aligned_8_16_32').accuracy,all_shift_accuracy=d[d.shift_pixels>0].moved_correct.mean(),equal_group_accuracy=np.mean([g(model,k).accuracy for k in GROUPS])))
 pd.DataFrame(overall).to_csv(out/'overall_accuracy.csv',index=False)
 matched=[]
 for stage,shift in [(0,8),(1,16),(2,32)]:
  for condition,model,group in [(f'對齊增強→{shift}px','aligned_8_16_32',None),('非對齊增強→1–7px平均','augment_1_7','non_aligned_1_7')]:
   row=dict(stage=stage,condition=condition)
   for key in ['accuracy','cluster_agreement','token_cosine']:
    before=v('baseline',shift,stage,key) if group is None else float(g('baseline',group,stage)[key]);after=v(model,shift,stage,key) if group is None else float(g(model,group,stage)[key])
    row.update({key+'_before':before,key+'_after':after,key+'_delta':after-before})
   matched.append(row)
 pd.DataFrame(matched).to_csv(out/'matched_training_test_comparison.csv',index=False)
 # Did category-level changes have the same sign, rather than relying on pooled average only?
 category_effects=[]
 for group in GROUPS:
  q=groups[(groups.group==group)&(groups.stage==0)&(groups.category!='all_selected')].pivot(index='category',columns='model',values='accuracy')
  for before,after in [('baseline','aligned_8_16_32'),('baseline','augment_1_7'),('aligned_8_16_32','augment_1_7')]:
   delta=q[after]-q[before]
   category_effects.append(dict(group=group,before=before,after=after,better=int((delta>1e-12).sum()),same=int((delta.abs()<=1e-12).sum()),worse=int((delta< -1e-12).sum())))
 pd.DataFrame(category_effects).to_csv(out/'category_effect_counts.csv',index=False)
 write_json(out/'verification.json',verification)
 lines=['# Caltech-101 全類別分層採樣：平移分類與表徵穩定性','',f'完成：{datetime.now(timezone.utc).isoformat()}；執行位置：remote140165 `/home/ref/RGB_SFM_V3`。','',
 '## 採樣與比較方式','',
 '涵蓋101類，每類從固定seed-42 validation split無放回抽5張，共505張；採樣seed42，三模型使用完全相同來源ID。來源清單、每類可用與採樣數見 `selection_manifest.json`。這是全類別涵蓋的平衡採樣，不是全部1,770張validation，也不是另設的untouched test。每類樣本很少，單類估計不穩定。','',
 '原位與四方向×1–8/16/32px；160-in-224，整數平移、無來源內容裁切、無平移插值；修正annotation offset後的中心token。沒有重新訓練。Baseline/A/T1分別為歷史無新增平移、8/16/32增強、完成版1–7增強（best epochs66/84/78）；不是epoch-10暫停模型。','',
 '沿用三類混合評估的同一組已驗證、固定training spherical codebooks。每模型各stage K31/niter100/nredo50/seed42；codebook由未平移全類別6,907張training中心token fitting，沒有用這505張重fit。不同模型cluster ID不對應；同模型跨位移字典固定。','',
 f'每模型505×41={505*41:,}個影像條件，四stage共{expected:,}筆token配對；三模型合計{expected*3:,}筆。各類樣本數相同，micro與macro在本採樣上相等；不代表原始資料分布的整體accuracy。Stage3網格64px沒有實際64px對齊測試。','',
 '## 總體分類結果','',
 table(['模型','原位','1–7px平均','8/16/32px平均','十個位移等權','兩組各50%'],[[NAMES[r['model']]]+[pct(r[k]) for k in ['original_accuracy','non_aligned_accuracy','patch_aligned_accuracy','all_shift_accuracy','equal_group_accuracy']] for r in overall]),'',
 '十個非零位移各等權時，小位移組占70%、對齊初始patch組占30%；兩組各50%則先各自彙整再平均。所有位移都先對四方向等權。','',
 '![全類別分類](classification_comparison.png)','',
 '## 群一致率','',
 '![全類別群一致率](cluster_agreement_comparison.png)','',
 '## Token cosine相似度','',
 '![全類別cosine](token_cosine_comparison.png)','',
 '## 對應訓練→測試條件的增強前後比較','',
 '每列Baseline→對應增強模型；對齊測試依stage採8/16/32px，非對齊測試皆1–7px平均。分類來自完整模型輸出，不是各stage獨立probe；非對齊accuracy各stage相同。','']
 for key,title in [('cluster_agreement','群一致率'),('accuracy','分類正確率'),('token_cosine','Cosine相似度')]:
  lines+=['### '+title,'']
  for aligned,label in [(True,'對齊增強 → 對齊測試'),(False,'非對齊增強 → 非對齊測試')]:
   rows=[]
   for r in matched:
    if r['condition'].startswith('對齊')!=aligned:continue
    f=(lambda x:f'{x:.4f}') if key=='token_cosine' else pct
    delta=f"{r[key+'_delta']:+.4f}" if key=='token_cosine' else f"{r[key+'_delta']*100:+.2f} pp"
    shift=f"{8*2**r['stage']} px" if aligned else '1–7 px 平均'
    rows.append([r['stage'],shift,f(r[key+'_before']),f(r[key+'_after']),delta])
   lines+=['#### '+label,'',table(['Stage','測試位移','增強前','增強後','改善量'],rows),'']
 lines+=['## 逐位移文字表','']
 rows=[]
 for shift in SHIFTS:rows.append([shift]+[pct(v(m,shift,0,'accuracy')) for m in MODELS])
 lines += ['### 分類accuracy','',table(['位移px','Baseline','A：8/16/32增強','T1：1–7增強'],rows),'']
 for key,title in [('cluster_agreement','群一致率'),('token_cosine','Cosine相似度')]:
  for stage in range(4):
   f=pct if key=='cluster_agreement' else lambda x:f'{x:.4f}'
   lines+=['### '+title+f'：Stage {stage}','',table(['位移px','Baseline','A','T1'],[[shift]+[f(v(m,shift,stage,key)) for m in MODELS] for shift in SHIFTS]),'']
 lines+=['## 配對差值與採樣不確定性','',
 '先於每張來源影像平均同組所有位移與方向，再比較模型；固定101類，每類內對5張來源影像有放回分層bootstrap，5,000次、seed42。這維持類別等權；區間不包含訓練seed、類別選擇、模型選模或其他採樣seed的不確定性。不是將平移重複視為獨立樣本；未做多重比較校正。','']
 rows=[]
 for _,r in contrast[(contrast.stage==0)&(contrast.metric=='accuracy')].iterrows():rows.append([r.group,NAMES[r.after]+' − '+NAMES[r.before],f'{r.delta*100:+.2f}',f'[{r.ci_low*100:+.2f}, {r.ci_high*100:+.2f}]'])
 lines += [table(['測試組','模型差值','Accuracy差值pp','95% CI pp'],rows),'',
 '## 分析與限制','',
 f"小位移1–7px：T1的accuracy為{pct(g('augment_1_7','non_aligned_1_7').accuracy)}，A為{pct(g('aligned_8_16_32','non_aligned_1_7').accuracy)}，Baseline為{pct(g('baseline','non_aligned_1_7').accuracy)}。較大8/16/32px組：A為{pct(g('aligned_8_16_32','patch_aligned_8_16_32').accuracy)}、T1為{pct(g('augment_1_7','patch_aligned_8_16_32').accuracy)}。這些是本次全類別平衡採樣的結果，應與三類探索性結果分開讀。",'',
 f"在32px，Baseline/A/T1的accuracy分別為{pct(v('baseline',32,0,'accuracy'))}/{pct(v('aligned_8_16_32',32,0,'accuracy'))}/{pct(v('augment_1_7',32,0,'accuracy'))}；Stage3 cosine分別為{v('baseline',32,3,'token_cosine'):.4f}/{v('aligned_8_16_32',32,3,'token_cosine'):.4f}/{v('augment_1_7',32,3,'token_cosine'):.4f}。分類與局部中心token穩定性是不同測量，不應互相取代。",'',
 '8px只對齊Stage0，16px對齊Stage0/1，32px對齊Stage0/1/2；所有本輪非零位移對Stage3均不對齊。對齊與非對齊組的位移量不同，不能宣稱單獨證明alignment的因果效果；非對齊token的內容涵蓋也會改變。高群一致率可能受分群集中程度影響，群數、熵與最大群占比見group_summary.csv；不能只用群一致率證明語意品質或嚴格平移不變性。','',
 '單一training seed、validation已參與選模；101類涵蓋改善了類別覆蓋，但每類5張仍不足以對單類作強結論。每類分類變好／不變／變差的數量見下表，該數量同樣受小樣本波動影響。','',
 table(['組別','模型差值','改善類數','相同類數','下降類數'],[[r['group'],NAMES[r['after']]+' − '+NAMES[r['before']],r['better'],r['same'],r['worse']] for r in category_effects]),'',
 '## 重現、校驗與產物','',
 '三模型確認來源ID與split相同、train/validation分離、資料位元組fingerprint一致、0px identity、完整筆數及唯一鍵、中心位移、centroid單位norm、codebook矩陣與上一輪逐值相同、artifact hashes。與原三類評估重疊的15張影像在全部位移下分類與群指派一致，cosine差異小於1e-5；詳見verification.json。','',
 '```bash\nALLCLASS_DEVICE=cuda:1 bash scripts/run_allclass_translation_1020.sh\n.venv/bin/python scripts/analyze_allclass_translation.py\n```','',
 '圖片：classification_comparison.png、cluster_agreement_comparison.png、token_cosine_comparison.png。文字：REPORT.md、REPORT.txt。數據：all_summary.csv（含每類）、plot_data.csv、group_summary.csv、direction_summary.csv、per_image_metrics.csv、paired_contrasts.csv、overall_accuracy.csv、matched_training_test_comparison.csv、category_effect_counts.csv；來源selection_manifest.json與逐檔analysis_manifest.json。','',
 '每模型raw paired CSV、feature caches與固定codebook留遠端，不納入Git；compact數據與報告可追蹤。本輪沒有將新產物複製／解壓到Windows/WSL本機專案。舊三類群一致率圖見 ../caltech_mixed_translation_1020/cluster_agreement_comparison.png。','']
 # Keep headline interpretations tied to this balanced sample rather than to the old three-class mixture.
 headline=['## 主要發現','',
  f"全類別平衡採樣下，T1在1–7px的accuracy最高（{pct(g('augment_1_7','non_aligned_1_7').accuracy)}），A在8/16/32px最高（{pct(g('aligned_8_16_32','patch_aligned_8_16_32').accuracy)}）。兩組各50%時A為{pct(overall[1]['equal_group_accuracy'])}、T1為{pct(overall[2]['equal_group_accuracy'])}，點估計相近；這不同於先前三類影像占比失衡的彙總，不應沿用原先的排名差距。未針對此混合分數差值另作顯著性判定。",'',
  '每類5張使本輪原位accuracy接近類別等權的採樣估計，不能與全部1,770張、依原始類別數加權的62.09%/65.71%/68.36%直接比較或解讀為模型退化。','']
 lines[4:4]=headline
 relative=['### 原位差異與平移退化應分開看','',
  table(['模型','原位accuracy','1–7px相對原位變化','8/16/32px相對原位變化'],[[NAMES[m],pct(v(m,0,0,'accuracy')),f"{(g(m,'non_aligned_1_7').accuracy-v(m,0,0,'accuracy'))*100:+.2f} pp",f"{(g(m,'patch_aligned_8_16_32').accuracy-v(m,0,0,'accuracy'))*100:+.2f} pp"] for m in MODELS]),'',
  'T1的絕對accuracy改善同時包含較高原位表現，不能全部歸因於平移穩健性。平移相對原位的降幅與prediction consistency應一起閱讀；後者逐位移數字見plot_data.csv。','',
  '### 表徵改善的範圍','',
  '對Stage0–2各自對齊的8/16/32px，對齊增強後的群一致率與cosine仍高於非對齊增強在1–7px平均的值；非對齊組的絕對表徵增益則較大。這個起點與上限不同的比較應同時報告before/after，不能只以改善量或最終值判斷方法優劣。','',
  f"同樣以1–7px測試時，Stage3的群一致率是T1 {pct(g('augment_1_7','non_aligned_1_7',3).cluster_agreement)}、A {pct(g('aligned_8_16_32','non_aligned_1_7',3).cluster_agreement)}；cosine則是T1 {g('augment_1_7','non_aligned_1_7',3).token_cosine:.4f}、A {g('aligned_8_16_32','non_aligned_1_7',3).token_cosine:.4f}，兩指標排序不一致。這也顯示不能將某個stage或三類子集的排序概括為所有表徵指標。",'']
 position=lines.index('## 分析與限制');lines[position:position]=relative
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 # Plain text preserves the same values, with tab-separated tables and image filenames.
 plain=[]
 for line in lines:
  for text in line.splitlines():
   if text.startswith('|'):
    if set(text)<=set('|-: '):continue
    text='\t'.join(x.strip() for x in text.strip('|').split('|'))
   plain.append(text.lstrip('#').strip() if text.startswith('#') else text)
 (out/'REPORT.txt').write_text('\n'.join(plain)+'\n')
 figure_report=['# Caltech-101 平移增強：分類與表徵穩定性三圖報告','',
 '## 評估設定','',
 '涵蓋101類，每類固定採樣5張validation影像，共505張；採樣seed42。三模型使用相同來源影像，測試原位與上下左右1–8、16、32px平移。各類樣本數相同，以下正確率為此平衡採樣的結果。','',
 '- **Baseline（灰）**：無新增平移增強。\n- **對齊增強（藍）**：8／16／32px訓練增強。\n- **非對齊增強（橘）**：1–7px訓練增強。','',
 '群一致率與cosine取修正annotation中心的對應token；四stage網格依序為8／16／32／64px。每模型、每stage沿用以未平移training特徵建立的固定spherical K-means字典（K=31）。圖片橫軸逐一列出已測位移，並非等比例距離；虛線區分1–7px與8／16／32px，後者只保證對齊初始8px網格。','',
 '## 圖一：分類正確率','',
 '![全類別分類正確率](classification_comparison.png)','',
 table(['模型','原位','非對齊1–7px平均','8／16／32px平均'],[[NAMES[r['model']]]+[pct(r[k]) for k in ['original_accuracy','non_aligned_accuracy','patch_aligned_accuracy']] for r in overall]),'',
 '非對齊增強模型在原位與小位移的正確率較高；對齊增強模型在16／32px下較穩定。非對齊增強的絕對正確率優勢也包含較高原位表現，不能全部歸因於平移穩健性。','',
 '## 圖二：群一致率','',
 '![全類別群一致率](cluster_agreement_comparison.png)','',
 '群一致率表示平移前後對應中心token仍分到同一群的比例。與Baseline相比，非對齊增強改善小位移下的群一致率；各stage達到自身網格對齊的位移時，一致率明顯回升。Stage3網格為64px，本輪所有非零位移均未對齊該層。','',
 '## 圖三：Cosine相似度','',
 '![全類別Token cosine相似度](token_cosine_comparison.png)','',
 'Cosine衡量平移前後中心token向量的方向相似度，不依賴群指派。非對齊增強提高小位移下的相似度；對齊增強對較大位移與Stage3較有利。Cosine與群一致率的模型排序不一定相同，兩者應並讀。','',
 '## 綜合判讀','',
 '對Stage0–2分別使用8／16／32px對齊測試時，對齊增強後的群一致率與cosine仍高於非對齊增強在1–7px平均的結果；非對齊組的表徵改善幅度則較大。兩組的起點、位移量與token涵蓋內容不同，這是各自條件下的比較，不能單獨證明alignment的因果效果或嚴格平移不變性。','',
 '本輪只有一個訓練seed，validation曾參與選模，且每類僅5張；結果不代表全部1,770張validation的分布加權正確率，也不足以對單一類別作強結論。完整數據、配對比較與不確定性見[詳細分析報告](REPORT.md)與[繪圖數據](plot_data.csv)。','']
 (out/'FIGURE_REPORT.md').write_text('\n'.join(figure_report)+'\n')
 files=[p for p in out.rglob('*') if p.is_file() and p.suffix in ['.md','.txt','.csv','.json','.png'] and p.name!='analysis_manifest.json']
 write_json(out/'analysis_manifest.json',dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),helper_sha256=sha(ROOT/'scripts/analyze_mixed_translation.py'),sample_images=n,sample_classes=101,verified_pairs=expected*3,bootstrap_replicates=5000,bootstrap_seed=42,files={str(p.relative_to(out)):sha(p) for p in files}))
 print(pd.DataFrame(overall).to_string(index=False));print(out/'REPORT.md')

if __name__=='__main__':main()
