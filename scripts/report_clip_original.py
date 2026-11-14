#!/usr/bin/env python3
"""Evaluate CE/CLIP on complete natural-background validation images and report."""
import sys,json,hashlib
from pathlib import Path
import torch,numpy as np
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataloader.OriginalImagePairs import OriginalImagePairs
from models.ObjectScaleViT import ObjectScaleViT
root=Path(sys.argv[1]);torch.set_num_threads(4);rows={};correct={};configs={};labels_ref=None
for name in ['CE','CLIP']:
 folder=root/name;cfg=json.loads((folder/'config.json').read_text());configs[name]=cfg
 assert cfg['original_images'] and json.loads((folder/'status.json').read_text())['status']=='done'
 checkpoint=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
 model=ObjectScaleViT(**cfg['model_args']).to('cuda:1').eval();model.load_state_dict(checkpoint['model'])
 ds=OriginalImagePairs(str(ROOT/'data'),cfg['splits']['valid'],training=False);logits=[];labels=[]
 with torch.inference_mode():
  for batch in DataLoader(ds,batch_size=32,num_workers=2):logits.append(model(batch['a'].to('cuda:1')).cpu());labels.append(batch['label'])
 logits=torch.cat(logits).numpy();labels=torch.cat(labels).numpy();c=logits.argmax(1)==labels;correct[name]=c.astype(float)
 if labels_ref is None:labels_ref=labels
 else:assert np.array_equal(labels_ref,labels)
 rows[name]=dict(accuracy=float(c.mean()),macro_accuracy=float(np.mean([c[labels==k].mean() for k in np.unique(labels)])),best_epoch=checkpoint['epoch'],checkpoint_sha256=hashlib.sha256((folder/'best.pth').read_bytes()).hexdigest())
 np.savez_compressed(folder/'original_predictions.npz',logits=logits,labels=labels,source_indices=cfg['splits']['valid'])
for key in ['splits','seed','epochs','batch_size','model_args','pred_weight','region_weight','clip_stages','original_images']:assert configs['CE'][key]==configs['CLIP'][key],key
d=correct['CLIP']-correct['CE'];rng=np.random.default_rng(42);ci=np.quantile([d[rng.integers(len(d),size=len(d))].mean() for _ in range(1000)],[.025,.975]);result=dict(models=rows,clip_minus_CE=dict(mean=float(d.mean()),source_bootstrap95ci=ci.tolist()),limitations='Single seed; exploratory validation used for checkpoint selection; full source images retain natural background; only deterministic aspect-preserving resize/padding; no translation or scale augmentation.')
(root/'results.json').write_text(json.dumps(result,indent=2))
lines=['# 原圖訓練：CLIP監督的幫助','','完整來源影像保留自然背景，無物件去背／隨機縮放／平移／額外增強。僅等比例縮放完整影像到長邊224並置中補邊，不裁切。模型為原merging。兩組從頭訓練seed42、100epochs、6902train／1768validation；沿用原共同索引以匹配教師cache，先前排除的7張不重新納入。','', '| 組別 | 原圖validation accuracy | Macro accuracy | best epoch |','|---|---:|---:|---:|']
for name,r in rows.items():lines.append(f"| {name} | {r['accuracy']:.2%} | {r['macro_accuracy']:.2%} | {r['best_epoch']} |")
lines+=['',f'CLIP−CE：{100*d.mean():+.2f}百分點，來源配對bootstrap95% CI [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]。', '', 'CE為分類損失，CLIP組另加Stage3/4 region到frozen CLIP bbox教師的0.2 cosine監督。CE模型與CLIP模型推論皆只使用完整圖片，不需CLIP教師或GT bbox作分類輸入。', '', '單seed，validation用於選模，非untouched final test；CI不含seed變異。此試驗不測平移／縮放不變性，不能與純色cutout的accuracy當作同一測試分布比較。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
