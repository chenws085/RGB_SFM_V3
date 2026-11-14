#!/usr/bin/env python3
"""Frozen natural-image global/window checkpoints, controlled full-image translation."""
import sys,json,hashlib,csv
from pathlib import Path
import numpy as np,torch,faiss
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataloader.OriginalImageViews import OriginalImageViews
from models.ObjectScaleViT import ObjectScaleViT
from compare_clip_spherical_scale import unit,fit,assign
from train_caltech_mergingvit import CenterLetterbox,IMAGENET_MEAN,IMAGENET_STD
from torchvision import transforms

class HistoricalLetterboxViews(OriginalImageViews):
 def __getitem__(self,item):
  result=super().__getitem__(item)
  image,_=self.base[self.indices[item]]
  tensor=transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)(transforms.ToTensor()(CenterLetterbox(160)(image)))
  result['a']=tensor;result['b']=tensor
  return result
FOLDERS={'global':ROOT/'runs/train/clip_original_20261103T065816Z/CE','window':ROOT/'runs/train/window_original_20261109T080440Z/window'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
 historical=len(sys.argv)>2 and sys.argv[2]=='--letterbox-run'
 folders=({'global':ROOT/'runs/train/mergingvit_caltech101_letterbox160','window':Path(sys.argv[3])} if historical else FOLDERS)
 dataset_class=HistoricalLetterboxViews if historical else OriginalImageViews
 checkpoint_name='MergingViT_best.pth' if historical else 'best.pth'
 torch.set_num_threads(4);faiss.omp_set_num_threads(4)
 cfgs={k:json.loads((v/('run_config.json' if historical else 'config.json')).read_text()) for k,v in folders.items()}
 if historical:
  audit=json.loads((ROOT/'overnight_logs/letterbox_window_1109/global_audit.json').read_text())
  for n,c in cfgs.items():
   c['splits']={'train':audit['train_indices'],'valid':audit['valid_indices']};c['model_args']=c['model']['args']
  assert cfgs['window']['train_indices']==audit['train_indices'] and cfgs['window']['valid_indices']==audit['valid_indices']
 for k in ['splits','seed','epochs','model_args']:assert cfgs['global'][k]==cfgs['window'][k]
 attention_config={n:{k:cfg.get(k,default) for k,default in [('attention_mode','global'),('window_size',7),('attention_stages',None),('window_offset',0)]} for n,cfg in cfgs.items()}
 manifest=dict(attention_config=attention_config,source_sha256={q:sha(ROOT/q) for q in ['models/MergingViT.py','models/ObjectScaleViT.py','models/WindowAttention.py','dataloader/OriginalImageViews.py','dataloader/OriginalImagePairs.py','scripts/train_caltech_mergingvit.py']},training_input_protocol='letterbox160 matched' if historical else 'original224; scale shift at evaluation',checkpoints={n:dict(path=str(p/checkpoint_name),sha256=sha(p/checkpoint_name)) for n,p in folders.items()},script_sha256=sha(Path(__file__)),splits=cfgs['global']['splits'],content_max=160,canvas=224,coordinate='absolute bbox center; actual rounded resize ratios',clustering='training160 center tokens only; FAISS spherical K31/100iter/50restarts/seed42',status='running')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 rows=[];paired={};boot=None
 categories=OriginalImageViews(str(ROOT/'data'),[],160).base.categories
 for name,folder in folders.items():
  cfg=cfgs[name];cp=torch.load(folder/checkpoint_name,map_location='cpu',weights_only=False)
  model=ObjectScaleViT(**cfg['model_args'],attention_mode=cfg.get('attention_mode','global'),window_size=cfg.get('window_size',7),attention_stages=cfg.get('attention_stages'),window_offset=cfg.get('window_offset',0)).cuda(1).eval();model.load_state_dict(cp['model_weights'] if historical else cp['model'])
  def extract(split,dx=0,dy=0):
   fs=[[] for _ in range(4)];ys=[];pred=[];ids=[];meta=[];final=[]
   for b in DataLoader(dataset_class(str(ROOT/'data'),cfg['splits'][split],160),batch_size=64,num_workers=4):
    x=b['a'];g=b['geometry'][:,0];assert ((g[:,2]+dx>=0)&(g[:,3]+dy>=0)&(g[:,2]+g[:,0]+dx<=224)&(g[:,3]+g[:,1]+dy<=224)).all()
    x=torch.roll(x,(dy,dx),(-2,-1)).cuda(1)
    points=b['points_a'].mean((1,2))+torch.tensor([dx,dy])
    with torch.inference_mode():
     logits,features=model(x,True)
     for s,f in enumerate(features):
      ix=(points[:,0]//(8*2**s)).long();iy=(points[:,1]//(8*2**s)).long();assert (ix<f.shape[3]).all() and (iy<f.shape[2]).all()
      fs[s].append(f[torch.arange(len(x),device=f.device),:,iy.to(f.device),ix.to(f.device)].cpu().numpy())
     final.append(features[-1].mean((2,3)).cpu().numpy())
    pred.extend(logits.argmax(1).cpu().tolist());ys.extend(b['label'].tolist());ids.extend(b['index'].tolist());meta.append(points.numpy())
   return dict(features=[unit(np.concatenate(f)) for f in fs],final=unit(np.concatenate(final)),labels=np.array(ys),pred=np.array(pred),ids=np.array(ids),points=np.concatenate(meta))
  tr=extract('train');centers=[fit(f) for f in tr['features']];np.savez_compressed(out/f'{name}_centers.npz',**{f'stage{s}':c for s,c in enumerate(centers)})
  orig=extract('valid')
  if historical:assert float((orig['pred']==orig['labels']).mean())==cp['valid_acc'], 'Checkpoint accuracy not reproduced'
  np.savez_compressed(out/f'{name}_original.npz',**{f'stage{s}':f for s,f in enumerate(orig['features'])},**{k:v for k,v in orig.items() if k!='features'})
  if boot is None:boot=np.random.default_rng(42).integers(len(orig['ids']),size=(2000,len(orig['ids'])))
  for shift in [0,1,2,3,4,5,6,7,8,16,32]:
   directions={'original':(0,0)} if shift==0 else {'left':(-shift,0),'right':(shift,0),'up':(0,-shift),'down':(0,shift)}
   for direction,(dx,dy) in directions.items():
    moved=orig if shift==0 else extract('valid',dx,dy);assert np.array_equal(orig['ids'],moved['ids']);assert np.allclose(moved['points']-orig['points'],[dx,dy],atol=2e-5)
    for s in range(4):
     a=assign(orig['features'][s],centers[s]);b=assign(moved['features'][s],centers[s]);same=(a==b);cos=(orig['features'][s]*moved['features'][s]).sum(1);consistent=orig['pred']==moved['pred'];correct=moved['pred']==orig['labels'];fc=(orig['final']*moved['final']).sum(1)
     if shift==0:assert same.all() and consistent.all() and np.allclose(cos,1,atol=1e-5)
     key=f'{name}_{s}_{shift}_{direction}';paired[key]=np.stack([same,cos,consistent,correct,fc],1)
     for category in ['all','airplanes']:
      mask=np.ones(len(a),dtype=bool) if category=='all' else np.array([categories[int(y)]=='airplanes' for y in orig['labels']])
      rows.append(dict(model=name,stage=s,shift=shift,direction=direction,category=category,n=int(mask.sum()),agreement=float(same[mask].mean()),cosine=float(cos[mask].mean()),prediction_consistency=float(consistent[mask].mean()),accuracy=float(correct[mask].mean()),final_cosine=float(fc[mask].mean())))
   print(name,shift,'done',flush=True)
  del model;torch.cuda.empty_cache()
 with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 np.savez_compressed(out/'paired.npz',source_indices=orig['ids'],labels=orig['labels'],**paired)
 lines=['# 全局MHA／局部attention變體平移穩定性',json.dumps(attention_config,ensure_ascii=False),('完整自然圖長邊160、224畫布；training/eval皆使用歷史bilinear resize、整數meanRGB補邊。沿用歷史global，window以相同160設定從頭重訓。各模型training中心token固定spherical字典；修正absolute bbox中心。' if historical else '完整自然圖縮至長邊160、224畫布；所有來源像素保留。既有原圖224訓練checkpoint，無新訓練。各模型training160中心token建固定spherical字典；使用修正absolute bbox中心，非0921錯誤座標。'),f'Stage為0-based。以下為{len(orig["ids"])} validation，四方向平均；CI按來源配對bootstrap，固定模型／字典，不含seed變異。','|Stage|px|Global群一致|Window群一致|差值95%CI(pp)|Global cosine|Window cosine|Global分類一致|Window分類一致|','|---|---|---|---|---|---|---|---|---|']
 for s in range(4):
  for shift in [1,4,8,16,32]:
   vals={n:np.mean([v for k,v in paired.items() if k.startswith(f'{n}_{s}_{shift}_')],axis=0) for n in FOLDERS};g,w=vals['global'],vals['window'];ci=np.quantile((w[:,0]-g[:,0])[boot].mean(1),[.025,.975])*100
   lines.append(f'|{s}|{shift}|{g[:,0].mean():.2%}|{w[:,0].mean():.2%}|[{ci[0]:+.2f},{ci[1]:+.2f}]|{g[:,1].mean():.4f}|{w[:,1].mean():.4f}|{g[:,2].mean():.2%}|{w[:,2].mean():.2%}|')
 lines+=['','單一seed、validation曾選模；變體attention設定見manifest.attention_config；無額外relative bias。局部換群與分類改變需分開判讀，不能僅由換群推論分類錯誤原因。global沿用歷史checkpoint；此次修正bbox中心並使用spherical分群，不與舊座標／Euclidean數值合併。全方向與airplanes子集見summary.csv。']
 (out/'REPORT.md').write_text('\n'.join(lines));manifest['status']='complete';manifest['artifacts_sha256']={p.name:sha(p) for p in out.iterdir() if p.name!='manifest.json'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
