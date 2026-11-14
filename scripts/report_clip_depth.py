#!/usr/bin/env python3
"""Predefined depth ablation statistics and held-out CLIP readout diagnostics."""
import json,sys,hashlib
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.ObjectScaleViT import ObjectScaleViT,corresponding_regions
from dataloader.ObjectScalePairs import ObjectScalePairs,feature_grid

def main():
 torch.set_num_threads(4)
 root=Path(sys.argv[1]);folders={'A':ROOT/'runs/train/object_scale_full_20261031T093902Z/A','all':ROOT/'runs/train/clip_supervision_1101/full',**{n:root/n for n in ['deep','shallow','deep_half']}}
 output={};per_image={};diagnostics={};configs={}
 teacher=torch.load(ROOT/'datasets/clip_bbox_vitb32_1101.pt',map_location='cpu',weights_only=False)
 for name,folder in folders.items():
  cfg=json.loads((folder/'config.json').read_text());configs[name]=cfg
  assert json.loads((folder/'status.json').read_text())['status']=='done'
  pred=np.load(folder/'scale_predictions.npz');ev=json.loads((folder/'scale_evaluation.json').read_text())
  assert ev['checkpoint_sha256']==hashlib.sha256((folder/'best.pth').read_bytes()).hexdigest()
  correct=np.mean([pred[f'length_{size}_shift_{dx}_{dy}'].argmax(1)==pred['labels'] for size in [80,112] for dx,dy in [(0,0),(7,-5)]],axis=0);per_image[name]=correct
  output[name]=dict(accuracy_reference=float((pred['length_96_shift_0_0'].argmax(1)==pred['labels']).mean()),accuracy_heldout=float(correct.mean()),worst_accuracy=min(r['accuracy'] for r in ev['conditions']),clip_stages=cfg.get('clip_stages',[0,1,2,3]) if name!='A' else [],clip_weight=cfg.get('clip_weight',0) if name!='A' else 0)
  if name=='A':continue
  checkpoint=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
  model=ObjectScaleViT(**cfg['model_args']).to('cuda:1').eval();model.load_state_dict(checkpoint['model'])
  heads=torch.nn.ModuleList([torch.nn.Linear(d,512) for d in cfg['model_args']['embed_dims']]).to('cuda:1').eval();heads.load_state_dict(checkpoint['alignment_heads'])
  active=cfg.get('clip_stages',[0,1,2,3]);vectors={s:[] for s in active}
  ds=ObjectScalePairs(str(ROOT/'data'),cfg['splits']['valid'],training=False)
  with torch.inference_mode():
   for batch in DataLoader(ds,batch_size=32,num_workers=2):
    _,features=model(batch['a'].to('cuda:1'),True)
    for s in active:
     r=corresponding_regions(features[s],feature_grid(batch['points_a'].to('cuda:1'),features[s],8*2**s),batch['region'].to('cuda:1'))
     vectors[s].append(torch.nn.functional.normalize(heads[s](r),dim=-1).cpu())
  target=teacher['features'][cfg['splits']['valid']];mean=torch.nn.functional.normalize(teacher['features'][cfg['splits']['train']].mean(0),dim=0)
  constant=float((target*mean).sum(-1).mean());diagnostics[name]={}
  for s,parts in vectors.items():
   z=torch.cat(parts);diag=(z*target).sum(-1);retrieval=(z@target.T).argmax(-1)==torch.arange(len(z))
   diagnostics[name][str(s)]=dict(validation_teacher_cosine=float(diag.mean()),constant_train_mean_cosine=constant,cosine_above_constant=float(diag.mean())-constant,same_source_teacher_retrieval_top1=float(retrieval.float().mean()))
  print(name,output[name],diagnostics[name],flush=True)
 for name,cfg in configs.items():
  for key in ['splits','seed','epochs','batch_size','model_args','pred_weight','region_weight']:assert cfg[key]==configs['A'][key],(name,key)
 rng=np.random.default_rng(42);effects={}
 for first,second in [('deep','all'),('deep','shallow'),('deep_half','all'),('deep','A')]:
  d=per_image[first]-per_image[second];ci=np.quantile([d[rng.integers(len(d),size=len(d))].mean() for _ in range(1000)],[.025,.975]);effects[first+'-'+second]=dict(mean=float(d.mean()),source_bootstrap_95ci=ci.tolist())
 result=dict(classification=output,effects=effects,semantic_proxy_diagnostics=diagnostics,limitations='Single seed; exploratory validation used for checkpoint selection. CIs omit seed variation. CLIP retrieval is a teacher-alignment proxy, not human concept ground truth. Historical A/all have separate source snapshots; matched configs verified.')
 (root/'results.json').write_text(json.dumps(result,indent=2))
 lines=['# CLIP監督深度消融：本次實驗報告','','假設：深層較適合直接CLIP物件語意監督。比較原merging、共同初始化與資料、6902 train／1768 validation、seed42、100epochs。','', '| 組別 | CLIP直接監督層（1起算） | 權重 | 原尺度accuracy | 未見80/112平均 | 最差條件 |','|---|---|---:|---:|---:|---:|']
 for n,r in output.items():lines.append(f"| {n} | {[s+1 for s in r['clip_stages']]} | {r['clip_weight']} | {r['accuracy_reference']:.2%} | {r['accuracy_heldout']:.2%} | {r['worst_accuracy']:.2%} |")
 lines+=['','未見80/112平均包含原位及(7,-5)平移。原分類head評估，CLIP僅在訓練提供教師。配對來源bootstrap差值（百分點）：']
 for n,r in effects.items():lines.append(f"- {n}: {100*r['mean']:+.2f}，95% CI [{100*r['source_bootstrap_95ci'][0]:+.2f}, {100*r['source_bootstrap_95ci'][1]:+.2f}]")
 lines+=['','## 獨立於training cosine的語意代理評估','','驗證原位96 ROI對canonical bbox教師的cosine、超過training教師均值常數baseline的差距、1768候選內同來源teacher retrieval。數值見results.json。這不是人工語意標註，不能單獨證明深層內生語意更明確。','', '## 判讀限制','','只在deep優於all／shallow且未犧牲原尺度、deep_half仍支持效果時，才較支持監督位置的設計假設。若不成立，明確記錄不支持，不將既有深層群一致率當作語意證明。淺層無直接CLIP head損失，但仍接收深層反傳；不是凍結淺層。單seed探索validation、合成背景／近似輪廓；bootstrap不含training seed變異。']
 probe_file=root/'stage_semantic_probe.json'
 if probe_file.exists():
  probes=json.loads(probe_file.read_text());result['frozen_A_equal_rank_probes']=probes
  lines+=['','## 凍結原模型的等秩語意探測','','各層同rank32、ridge10，只用training fit PCA與線性映射；validation不選超參數。以下是CLIP語意可讀取性的代理證據，不是人工概念真值。','', '| 層（1起算） | validation teacher cosine | 超過常數teacher均值 | 同來源retrieval top1 |','|---|---:|---:|---:|']
  for r in probes['rows']:lines.append(f"| {r['stage']+1} | {r['validation_cosine']:.4f} | {r['cosine_above_constant']:+.4f} | {r['same_source_retrieval_top1']:.2%} |")
 positive=effects['deep-all']['source_bootstrap_95ci'][0]>0 and effects['deep-shallow']['source_bootstrap_95ci'][0]>0
 strength=effects['deep_half-all']['source_bootstrap_95ci'][0]>0
 verdict='主要深度對照支持深層監督較有效；仍需結合原尺度代價與多seed驗證。' if positive else '目前對照不足以支持深層監督普遍優於全層及淺層；不將推斷視為已證實。'
 result['verdict']=dict(depth_comparison=verdict,per_stage_weight_control_supports=strength)
 lines+=['','## 本次判讀','',verdict, '等每層權重控制：'+('支持相較全層的改善。' if strength else '未提供明確改善證據。')]
 (root/'results.json').write_text(json.dumps(result,indent=2))
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
