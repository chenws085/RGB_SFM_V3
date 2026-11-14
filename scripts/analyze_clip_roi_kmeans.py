#!/usr/bin/env python3
"""Training-only spherical dictionaries for intermediate object-region readouts."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch,faiss
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataloader.ObjectScalePairs import ObjectScalePairs,feature_grid
from models.ObjectScaleViT import ObjectScaleViT,corresponding_regions

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def main():
 torch.set_num_threads(4);faiss.omp_set_num_threads(4)
 out=ROOT/'plots/kmeans/clip_roi_1103'
 out.mkdir(parents=True,exist_ok=False)
 (out/'status.json').write_text(json.dumps(dict(status='running',pid=__import__('os').getpid())))
 sources={'A':ROOT/'runs/train/object_scale_full_20261031T093902Z/A','CLIP':ROOT/'runs/train/clip_supervision_1101/full'}
 result={};manifests={}
 for name,folder in sources.items():
  cfg=json.loads((folder/'config.json').read_text());cp=folder/'best.pth';checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
  m=ObjectScaleViT(**cfg['model_args']).to('cuda:1').eval();m.load_state_dict(checkpoint['model'])
  heads=None
  if name=='CLIP':
   heads=torch.nn.ModuleList([torch.nn.Linear(d,512) for d in cfg['model_args']['embed_dims']]).to('cuda:1').eval();heads.load_state_dict(checkpoint['alignment_heads'])
  manifests[name]=dict(checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),splits=cfg['splits'])
  collections={}
  with torch.inference_mode():
   for split in ['train','valid']:
    buckets={kind:[[[],[]] for _ in range(4)] for kind in (['raw','projected'] if heads is not None else ['raw'])}
    ds=ObjectScalePairs(str(ROOT/'data'),cfg['splits'][split],training=False)
    for batch in DataLoader(ds,batch_size=32,num_workers=2):
     for vi,key in enumerate(['a','b'] if split=='valid' else ['a']):
      _,features=m(batch[key].to('cuda:1'),True)
      for stage,f in enumerate(features):
       points=batch['points_'+key].to('cuda:1');mask=batch['region'].to('cuda:1')
       r=corresponding_regions(f,feature_grid(points,f,8*2**stage),mask)
       buckets['raw'][stage][vi].append(r.cpu().numpy())
       if heads is not None:buckets['projected'][stage][vi].append(heads[stage](r).cpu().numpy())
    collections[split]=buckets
    print(name,split,'extracted',len(ds),flush=True)
  result[name]={}
  for kind in collections['train']:
   rows=[]
   for stage in range(4):
    train=unit(np.concatenate(collections['train'][kind][stage][0])).astype('float32')
    va,vb=[unit(np.concatenate(x)).astype('float32') for x in collections['valid'][kind][stage]]
    km=faiss.Kmeans(train.shape[1],31,niter=100,nredo=50,seed=42,spherical=True,max_points_per_centroid=len(train));km.train(train)
    ca=(va@km.centroids.T).argmax(1);cb=(vb@km.centroids.T).argmax(1)
    counts=np.bincount(ca,minlength=31);p=counts/counts.sum();entropy=float(-(p[p>0]*np.log(p[p>0])).sum())
    # Compare with a different source of the same class, preventing class imbalance confounding.
    base=ds.base;labels=np.array([base.y[i] for i in cfg['splits']['valid']]);permutation=np.arange(len(labels))
    for label in np.unique(labels):
     ix=np.flatnonzero(labels==label);permutation[ix]=np.roll(ix,1)
    same=float((va*vb).sum(1).mean());control=float((va*vb[permutation]).sum(1).mean())
    row=dict(stage=stage,agreement=float((ca==cb).mean()),paired_cosine=same,same_class_different_image_cosine=control,
      paired_cosine_gap=same-control,occupied_clusters=int((counts>0).sum()),max_cluster_fraction=float(p.max()),effective_clusters=float(np.exp(entropy)))
    rows.append(row)
    np.savez_compressed(out/f'{name}_{kind}_stage{stage}.npz',centers=km.centroids,source_indices=cfg['splits']['valid'],a=va,b=vb,cluster_a=ca,cluster_b=cb)
    print(name,kind,row,flush=True)
   result[name][kind]=rows
  (out/'results.json').write_text(json.dumps(result,indent=2))
 assert manifests['A']['splits']==manifests['CLIP']['splits']
 (out/'manifest.json').write_text(json.dumps(manifests,indent=2))
 lines=['# CLIP監督：中間層物件region spherical K-means','','每層／模型／readout獨立以全部6902張training原位96的region表示建K=31球面字典，100 iterations、50 restarts、seed42。1768張validation比較原位96與80+(7,-5)。這是8×8對應mask加權的整物件region readout，不是逐token分群、部件定位或跨層因果證據。群號不可跨字典對照。','','| 模型／特徵 | Stage index | 群一致率 | 配對cosine | 同類異圖cosine | 最大群占比 | 有效群數 |','|---|---:|---:|---:|---:|---:|---:|']
 for name,kinds in result.items():
  for kind,rows in kinds.items():
   for r in rows:lines.append(f"| {name}/{kind} | {r['stage']} | {r['agreement']:.2%} | {r['paired_cosine']:.4f} | {r['same_class_different_image_cosine']:.4f} | {r['max_cluster_fraction']:.2%} | {r['effective_clusters']:.2f} |")
 lines+=['','單seed探索validation；僅一組尺度／平移比較。高cosine與一致率須配合群多樣性及同類異圖對照，投影改善不能代表原始token改善。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'status.json').write_text(json.dumps(dict(status='done')))
if __name__=='__main__':main()
