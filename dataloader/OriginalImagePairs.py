"""Complete natural-background source image; deterministic resize/pad, no augmentation."""
from pathlib import Path
import numpy as np,torch
from PIL import Image
from scipy.io import loadmat
from torchvision.datasets import Caltech101
from dataloader.ObjectScalePairs import MEAN,STD
class OriginalImagePairs:
 def __init__(self,root,indices,training=True,seed=42,**kwargs):
  self.base=Caltech101(root,target_type=['category','annotation']);self.indices=list(indices);self.epoch=0
 def __len__(self):return len(self.indices)
 def __getitem__(self,item):
  index=self.indices[item];image,(label,_)=self.base[index];image=image.convert('RGB');width,height=image.size
  factor=224/max(width,height);w=max(1,round(width*factor));h=max(1,round(height*factor));ox=(224-w)//2;oy=(224-h)//2
  # Preserve every source pixel region and its natural background; pad only outside image.
  canvas=np.broadcast_to(MEAN[:,0,0].numpy(),(224,224,3)).copy()
  canvas[oy:oy+h,ox:ox+w]=np.asarray(image.resize((w,h),Image.Resampling.BICUBIC),dtype='float32')/255
  tensor=(torch.from_numpy(canvas).permute(2,0,1)-MEAN)/STD
  path=Path(self.base.root)/'Annotations'/self.base.annotation_categories[label]/f'annotation_{self.base.index[index]:04d}.mat'
  box=loadmat(path)['box_coord'].ravel();bounds=np.array([max(0,box[2]-1),max(0,box[0]-1),min(width,box[3]),min(height,box[1])],dtype='float32')
  if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:raise ValueError('Invalid bbox')
  yy,xx=np.meshgrid((np.arange(8)+.5)/8,(np.arange(8)+.5)/8,indexing='ij');points=np.stack([xx,yy],-1).astype('float32')*(bounds[2:]-bounds[:2])+bounds[:2]
  points=points*np.array([w/width,h/height],dtype='float32')+np.array([ox,oy],dtype='float32');points=torch.from_numpy(points)
  return dict(a=tensor,b=tensor,points_a=points,points_b=points,region=torch.ones(1,8,8),label=label,index=index,geometry=torch.tensor([[w,h,ox,oy],[w,h,ox,oy]]))
