#!/usr/bin/env python3
"""Full natural-image global/window CE comparison, including frozen spherical dictionaries."""
import hashlib,itertools,json,sys
from pathlib import Path
import faiss,numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataloader.OriginalImageViews import OriginalImageViews
from dataloader.ObjectScalePairs import feature_grid
from models.ObjectScaleViT import ObjectScaleViT,corresponding_regions
from compare_clip_spherical_scale import unit,fit,assign,mapping

BASE=ROOT/'runs/train/clip_original_20261103T065816Z/CE'
SIZES=[112,160,192,224]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 torch.set_num_threads(4);faiss.omp_set_num_threads(4)
 run=Path(sys.argv[1]).resolve();out=run/'spherical';out.mkdir(exist_ok=False)
 folders={'global':BASE,'window':run/'window'}
 configs={n:json.loads((p/'config.json').read_text()) for n,p in folders.items()}
 audit=json.loads((run/'input_checks.json').read_text())
 assert sha(BASE/'best.pth')==audit['baseline_checkpoint_sha256']
 assert sha(ROOT/'dataloader/OriginalImagePairs.py')==audit['original_loader_sha256']
 assert sha(ROOT/'dataloader/OriginalImageViews.py')==audit['view_loader_sha256']
 for k in ['splits','seed','epochs','batch_size','model_args','pred_weight','region_weight','annotation_sha256']:
  assert configs['global'][k]==configs['window'][k],k
 assert all(c['original_images'] and c['clip_cache'] is None for c in configs.values())
 assert configs['window']['attention_mode']=='window'
 values,centers,maps,assignments,classification={},{},{},{},{}
 labels=indices=None
 for name,folder in folders.items():
  assert json.loads((folder/'status.json').read_text())['status']=='done'
  cfg=configs[name];cp=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
  model=ObjectScaleViT(**cfg['model_args'],attention_mode=cfg.get('attention_mode','global'),window_size=cfg.get('window_size',7)).to('cuda:1').eval();model.load_state_dict(cp['model'])
  classification[name]={'best_epoch':cp['epoch']}
  with torch.inference_mode():
   for split,size in [('train',224)]+[('valid',s) for s in SIZES]:
    buckets=[[] for _ in range(4)];got=[];ys=[];pred=[]
    ds=OriginalImageViews(str(ROOT/'data'),cfg['splits'][split],size)
    for batch in DataLoader(ds,batch_size=32,num_workers=2):
     logits,features=model(batch['a'].to('cuda:1'),True);points=batch['points_a'].to('cuda:1');mask=batch['region'].to('cuda:1')
     for s,f in enumerate(features):buckets[s].append(corresponding_regions(f,feature_grid(points,f,8*2**s),mask).cpu().numpy())
     got.extend(batch['index'].tolist());ys.extend(batch['label'].tolist());pred.append(logits.cpu().numpy())
    assert got==cfg['splits'][split]
    d={f'stage{s}':unit(np.concatenate(v)) for s,v in enumerate(buckets)}
    d.update(logits=np.concatenate(pred),labels=np.array(ys),source_indices=np.array(got))
    values[(name,split,size)]=d;np.savez_compressed(out/f'{name}_{split}_L{size}_features.npz',**d)
    if split=='valid':
     if labels is None:labels,indices=d['labels'],d['source_indices']
     else:assert np.array_equal(labels,d['labels']) and np.array_equal(indices,d['source_indices'])
     correct=d['logits'].argmax(1)==labels
     classification[name][str(size)]=dict(accuracy=float(correct.mean()),macro_accuracy=float(np.mean([correct[labels==k].mean() for k in np.unique(labels)])))
    print(name,split,size,'features saved',flush=True)
  del model,cp;torch.cuda.empty_cache()
  train_ids=[]
  for s in range(4):
   center=fit(values[(name,'train',224)][f'stage{s}']);centers[f'{name}_{s}']=center
   train_ids.append(assign(values[(name,'train',224)][f'stage{s}'],center))
   for size in SIZES:assignments[f'{name}_{s}_{size}']=assign(values[(name,'valid',size)][f'stage{s}'],center)
  for i,j in itertools.combinations(range(4),2):maps[f'{name}_{i}_{j}']=mapping(train_ids[i],train_ids[j])
  np.savez_compressed(out/'dictionaries.partial.npz',**centers,**maps,**assignments)
 permutation=np.arange(len(labels))
 for label in np.unique(labels):
  ix=np.flatnonzero(labels==label);assert len(ix)>1;permutation[ix]=np.roll(ix,1)
 rows,per_source=[],{}
 def add(name,metric,stages,size,x,y,second=None):
  same=(x==y).astype('float32');control=(x==y[permutation]).astype('float32')
  if second is not None:
   xx,yy=second;same=(same+(xx==yy))/2;control=(control+(xx==yy[permutation]))/2
  key=f'{name}_{metric}_{stages}_{size}';per_source[key]=same;per_source[key+'_control']=control
  rows.append(dict(model=name,metric=metric,stages=stages,size=size,agreement=float(same.mean()),same_class_other=float(control.mean()),gap=float((same-control).mean()),key=key))
 for name in folders:
  def a(s,size):return assignments[f'{name}_{s}_{size}']
  for s in range(4):
   for size in [160,192]:add(name,'within',[s+1],size,a(s,224),a(s,size))
  for i,j in itertools.combinations(range(4),2):
   m=maps[f'{name}_{i}_{j}']
   add(name,'cross_same',[i+1,j+1],224,a(i,224),m[a(j,224)])
   for size in [160,192]:add(name,'cross_scale',[i+1,j+1],size,a(i,224),m[a(j,size)],(a(i,size),m[a(j,224)]))
   if j-i==1:add(name,'support_ratio',[i+1,j+1],112,a(i,112),m[a(j,224)])
 boot=np.random.default_rng(42).integers(len(labels),size=(2000,len(labels)));summaries,effects=[],[]
 for metric in ['within','cross_same','cross_scale','support_ratio']:
  options=sorted({tuple(r['stages']) for r in rows if r['metric']==metric})+['mean']
  for stages in options:
   scores,controls={},{}
   for name in folders:
    selected=[r for r in rows if r['metric']==metric and r['model']==name and (stages=='mean' or tuple(r['stages'])==stages)]
    scores[name]=np.mean([per_source[r['key']] for r in selected],axis=0);controls[name]=np.mean([per_source[r['key']+'_control'] for r in selected],axis=0)
    summaries.append(dict(model=name,metric=metric,stages=stages,agreement=float(scores[name].mean()),same_class_other=float(controls[name].mean()),gap=float((scores[name]-controls[name]).mean())))
   d=scores['window']-scores['global'];gap=d-(controls['window']-controls['global'])
   effects.append(dict(metric=metric,stages=stages,delta=float(d.mean()),source_bootstrap_95ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist(),gap_delta=float(gap.mean())))
 correct={n:(values[(n,'valid',224)]['logits'].argmax(1)==labels).astype(float) for n in folders}
 d=correct['window']-correct['global'];classification['window_minus_global']=dict(delta=float(d.mean()),source_bootstrap_95ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
 occupancy={}
 for key,a in assignments.items():
  p=np.bincount(a,minlength=31)/len(a);occupancy[key]=dict(occupied=int((p>0).sum()),max_fraction=float(p.max()),effective_clusters=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())))
 assert all(np.isfinite(v).all() for v in per_source.values())
 np.savez_compressed(out/'per_source.npz',source_indices=indices,labels=labels,**per_source)
 np.savez_compressed(out/'dictionaries_assignments.npz',**centers,**maps,**assignments)
 result=dict(classification=classification,rows=rows,summaries=summaries,effects=effects,occupancy=occupancy)
 (out/'results.json').write_text(json.dumps(result,indent=2))
 (out/'manifest.json').write_text(json.dumps(dict(checkpoints={n:dict(path=str(p/'best.pth'),sha256=sha(p/'best.pth')) for n,p in folders.items()},
  splits=configs['global']['splits'],script_sha256=sha(Path(__file__)),source_sha256={s:sha(ROOT/s) for s in ['dataloader/OriginalImageViews.py','dataloader/OriginalImagePairs.py','models/WindowAttention.py','models/ObjectScaleViT.py','scripts/compare_clip_spherical_scale.py']},
  preprocessing='Complete source RGB and natural background; deterministic aspect-preserving resize/center pad224; no cutout, crop, flip, translation or random scale augmentation in training',
  evaluation='Full image including background resized together to long edge112/160/192/224 on224canvas, no shift or crop',
  spherical='Training224 raw bbox ROI; independent per-stage FAISS spherical K31 niter100 nredo50 seed42; Hungarian fitted only on training source pairs',
  baseline='Reused completed original-image CE100epoch checkpoint after config and input parity checks; window trained from scratch, same seed/initialization/budget'),indent=2))
 lines=['# 完整自然原圖：全局／窗口MHA純CE對照', '',
  '訓練保留完整RGB影像與自然背景，只做長邊224等比例縮放、置中補邊與正規化；無去背／裁切／額外增強，single forward。global沿用已完成同設定100epoch純CE模型，window從頭訓練；seed42、batch16、6902training／1768validation、原merging與相同參數量。沿用共同來源索引，歷史排除的7張不加回。', '',
  'window7實際窗口7/7/7/4，後兩stage覆蓋全層；無移窗或relative position bias，非完整Swin。分類輸入不需bbox；bbox僅供8×8整物件ROI特徵讀取，沒有分割mask。分群是原始ROI各層獨立字典＋training Hungarian對應，不是投影共用字典或逐token比較。', '',
  '| 原圖224分類 | 全局MHA | 窗口MHA |','|---|---:|---:|',
  f"| accuracy | {classification['global']['224']['accuracy']:.2%} | {classification['window']['224']['accuracy']:.2%} |",
  f"| macro accuracy | {classification['global']['224']['macro_accuracy']:.2%} | {classification['window']['224']['macro_accuracy']:.2%} |",
  f"| best epoch | {classification['global']['best_epoch']} | {classification['window']['best_epoch']} |"]
 for metric,title in [('within','同層跨尺度：整圖224→160/192'),('cross_same','跨層同原圖224'),('cross_scale','跨層＋跨尺度：整圖224↔160/192'),('support_ratio','相鄰層名義支援比例匹配：淺層整圖112→深層整圖224')]:
  lines+=['','## '+title,'','| 層／配對 | 全局MHA | 窗口MHA | 差值 |','|---|---:|---:|---:|']
  options=[r['stages'] for r in summaries if r['metric']==metric and r['model']=='global']
  for stages in options:
   vals=[next(r['agreement'] for r in summaries if r['metric']==metric and r['stages']==stages and r['model']==n) for n in folders]
   label='平均' if stages=='mean' else '↔'.join(map(str,stages));lines.append(f'| {label} | {vals[0]:.2%} | {vals[1]:.2%} | {100*(vals[1]-vals[0]):+.2f}pp |')
 lines+=['','## 限制','',
  '縮放挑戰改變完整圖片與背景大小，與先前物件獨立縮放不同，不能把兩套accuracy當同一測試分布。名義stride比例匹配不是實測有效RF或逐token支援框匹配。未做224大圖平移相位校正，因為固定224畫布可能裁切原圖，違反完整內容要求。', '',
  '字典只用training224建立並固定，validation不fit／配群，但曾用於選best，非final test。bootstrap2000次以來源為單位，先在影像內平均，不含seed變異，CI未多重比較校正；群佔用、同類異圖對照與CI見results.json。', '',
  '遠端/home/ref/RGB_SFM_V3；研究文件版本及SHA256交付E: inbox不是本機專案匯入，本機root未核實。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print((out/'REPORT.md').read_text(),flush=True)

if __name__=='__main__':main()
