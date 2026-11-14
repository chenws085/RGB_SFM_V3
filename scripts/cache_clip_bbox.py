#!/usr/bin/env python3
"""Frozen OpenAI CLIP canonical bbox teacher, preserving bbox aspect ratio."""
import sys,json,hashlib,argparse
from pathlib import Path
import torch
from PIL import Image,ImageOps
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataloader.ObjectScalePairs import ObjectScalePairs

def main():
 import clip
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 torch.set_num_threads(4)
 ref=json.loads((ROOT/'runs/train/object_scale_full_20261031T093902Z/A/config.json').read_text())
 base=ObjectScalePairs(str(ROOT/'data'),[]).base
 teacher,preprocess=clip.load('ViT-B/32',device='cuda:1',download_root=str(ROOT/'weights/clip'))
 teacher.eval().requires_grad_(False)
 ids=ref['splits']['train']+ref['splits']['valid'];features=torch.zeros(len(base),512);entries=[]
 with torch.inference_mode():
  for start in range(0,len(ids),64):
   batch=[]
   for index in ids[start:start+64]:
    image,_=base[index];label=base.y[index]
    mat=Path(base.root)/'Annotations'/base.annotation_categories[label]/f'annotation_{base.index[index]:04d}.mat'
    box=loadmat(mat)['box_coord'].ravel();bounds=(max(0,int(box[2])-1),max(0,int(box[0])-1),min(image.width,int(box[3])),min(image.height,int(box[1])))
    crop=image.convert('RGB').crop(bounds)
    square=Image.new('RGB',(max(crop.size),)*2,(123,117,104));square.paste(crop,((square.width-crop.width)//2,(square.height-crop.height)//2))
    batch.append(preprocess(square));entries.append(dict(index=index,bbox=bounds,annotation_sha256=hashlib.sha256(mat.read_bytes()).hexdigest()))
   z=teacher.encode_image(torch.stack(batch).to('cuda:1')).float();z=torch.nn.functional.normalize(z,dim=-1).cpu()
   features[ids[start:start+64]]=z
   if start%640==0:print('cached',start+len(batch),'/',len(ids),flush=True)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 torch.save(dict(features=features,indices=ids,teacher='OpenAI CLIP ViT-B/32',entries=entries,
  teacher_sha256=hashlib.sha256((ROOT/'weights/clip/ViT-B-32.pt').read_bytes()).hexdigest(),
  preprocessing='original annotation bbox; square pad preserving full crop; official CLIP preprocessing',splits=ref['splits']),args.output)
 print('Cache complete',args.output,flush=True)
if __name__=='__main__':main()
