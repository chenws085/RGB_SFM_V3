#!/usr/bin/env python3
"""Controlled object-scale curves for a completed paired pilot (exploratory validation)."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataloader.ObjectScalePairs import ObjectScalePairs,object_crop,MEAN,STD
from models.ObjectScaleViT import ObjectScaleViT

class Views(Dataset):
 def __init__(self,base,indices,length,shift):self.base,self.indices,self.length,self.shift=base,indices,length,shift
 def __len__(self):return len(self.indices)
 def __getitem__(self,item):
  index=self.indices[item];image,mask,label,_=object_crop(self.base,index)
  a=np.asarray(mask,dtype='float32')/255;c=np.asarray(image,dtype='float32')/255*a[:,:,None]
  scale=self.length/max(image.size);w=max(1,round(image.width*scale));h=max(1,round(image.height*scale))
  ox=(224-w)//2+self.shift[0];oy=(224-h)//2+self.shift[1]
  assert 0<=ox<=224-w and 0<=oy<=224-h
  alpha=np.asarray(Image.fromarray(a).resize((w,h),Image.Resampling.BILINEAR))
  color=np.stack([np.asarray(Image.fromarray(c[:,:,i]).resize((w,h),Image.Resampling.BILINEAR)) for i in range(3)],-1)
  bg=MEAN[:,0,0].numpy();canvas=np.broadcast_to(bg,(224,224,3)).copy()
  canvas[oy:oy+h,ox:ox+w]=color+bg*(1-alpha[:,:,None])
  return (torch.from_numpy(canvas).permute(2,0,1)-MEAN)/STD,label,index

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--no-translation',action='store_true');p.add_argument('--device',default='cuda:1');args=p.parse_args()
 torch.set_num_threads(4)
 cfg=json.loads((args.run/'config.json').read_text());cp=args.run/'best.pth'
 model=ObjectScaleViT(**cfg['model_args'],content_merging=cfg['content_merging'],hybrid_merging=cfg.get('hybrid_merging',False),attention_mode=cfg.get('attention_mode','global'),window_size=cfg.get('window_size',7)).to(args.device)
 model.load_state_dict(torch.load(cp,map_location=args.device,weights_only=False)['model']);model.eval()
 base=ObjectScalePairs(str(ROOT/'data'),[]).base;ids=cfg['splits']['valid'];outputs={};rows=[]
 with torch.inference_mode():
  for length in [48,64,80,96,112,128,144]:
   for shift in ([(0,0)] if args.no_translation else [(0,0),(7,-5)]):
    logits=[];labels=[]
    for x,y,_ in DataLoader(Views(base,ids,length,shift),batch_size=32,num_workers=2):
     logits.append(model(x.to(args.device)).cpu());labels.append(y)
    logits=torch.cat(logits);labels=torch.cat(labels);outputs[(length,shift)]=(logits,labels)
 reference=outputs[(96,(0,0))][0].argmax(1)
 for (length,shift),(logits,labels) in outputs.items():
  pred=logits.argmax(1);correct=pred==labels
  rows.append(dict(length=length,shift=shift,accuracy=correct.float().mean().item(),
    macro_accuracy=torch.stack([correct[labels==k].float().mean() for k in labels.unique()]).mean().item(),
    agreement_with_reference=(pred==reference).float().mean().item()))
 result=dict(checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),source_indices=ids,
             split='exploratory validation used for checkpoint selection; not final test',conditions=rows,
             evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 dest=args.run/'scale_evaluation.json'
 if dest.exists():raise FileExistsError(dest)
 dest.write_text(json.dumps(result,indent=2));print(json.dumps(rows,indent=2))
 np.savez_compressed(args.run/'scale_predictions.npz',source_indices=ids,labels=labels.numpy(),
  **{f'length_{length}_shift_{shift[0]}_{shift[1]}':logits.numpy() for (length,shift),(logits,_) in outputs.items()})
if __name__=='__main__':main()
