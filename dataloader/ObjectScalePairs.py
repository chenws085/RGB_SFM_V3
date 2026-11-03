"""Contour cutouts on fixed synthetic backgrounds; no source background is reused."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.datasets import Caltech101
from torchvision.transforms.functional import pil_to_tensor

MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]


def object_crop(base, index):
    image,(label,contour)=base[index]
    image=image.convert('RGB')
    path=Path(base.root)/'Annotations'/base.annotation_categories[label]/f'annotation_{base.index[index]:04d}.mat'
    box=loadmat(path)['box_coord'].ravel()
    points=contour.astype(float)+np.array([[box[2]],[box[0]]])-1
    if not np.isfinite(points).all() or points.shape[1]<3:
        raise ValueError('Invalid contour')
    # Reject inconsistent coordinates rather than silently clipping an entire contour.
    if points[0].min() < -2 or points[1].min() < -2 or points[0].max()>image.width+2 or points[1].max()>image.height+2:
        raise ValueError('Contour outside source image')
    mask=Image.new('L',image.size)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in points.T],fill=255)
    bounds=mask.getbbox()
    if bounds is None or np.asarray(mask).sum()/255<32:
        raise ValueError('Empty/tiny mask')
    return image.crop(bounds),mask.crop(bounds),label,path


class ObjectScalePairs(Dataset):
    def __init__(self, root, indices, training=True, seed=42, epoch=0):
        self.base=Caltech101(root,target_type=['category','annotation'])
        self.indices=list(indices);self.training=training;self.seed=seed;self.epoch=epoch

    def __len__(self):return len(self.indices)

    def __getitem__(self,item):
        index=self.indices[item]
        image,mask,label,_=object_crop(self.base,index)
        rng=np.random.default_rng(self.seed+index*1009+self.epoch*1000003)
        # Background is class-independent, identical across the two views.
        color=rng.uniform(.25,.75,3).astype('float32') if self.training else np.array([.485,.456,.406],dtype='float32')
        # Premultiplied colors avoid source background bleeding at mask boundaries.
        rgb=np.asarray(image,dtype='float32')/255
        alpha=np.asarray(mask,dtype='float32')/255
        premul=rgb*alpha[:,:,None]
        yy,xx=np.meshgrid((np.arange(8)+.5)/8,(np.arange(8)+.5)/8,indexing='ij')
        canonical=np.stack([xx,yy],-1).astype('float32')
        region=np.asarray(mask.resize((8,8),Image.Resampling.BOX),dtype='float32')/255
        views=[];points=[];metas=[]
        for view in range(2):
            length=int(rng.choice([64,96,128])) if self.training else [96,80][view]
            dx,dy=rng.integers(-16,17,size=2) if self.training else ([0,0] if view==0 else [7,-5])
            scale=length/max(image.size);w=max(1,round(image.width*scale));h=max(1,round(image.height*scale))
            ox=(224-w)//2+int(dx);oy=(224-h)//2+int(dy)
            assert 0<=ox<=224-w and 0<=oy<=224-h
            a=np.asarray(Image.fromarray(alpha).resize((w,h),Image.Resampling.BILINEAR))
            c=np.stack([np.asarray(Image.fromarray(premul[:,:,i]).resize((w,h),Image.Resampling.BILINEAR)) for i in range(3)],-1)
            canvas=np.broadcast_to(color,(224,224,3)).copy()
            canvas[oy:oy+h,ox:ox+w]=c+color*(1-a[:,:,None])
            tensor=torch.from_numpy(canvas).permute(2,0,1)
            views.append((tensor-MEAN)/STD)
            # Pixel-edge coordinates; token centers lie at (j+.5)*stride.
            points.append(torch.from_numpy(canonical*np.array([w,h],dtype='float32')+np.array([ox,oy],dtype='float32')))
            metas.append([w,h,ox,oy])
        return dict(a=views[0],b=views[1],points_a=points[0],points_b=points[1],
                    region=torch.from_numpy(region)[None],label=label,index=index,geometry=torch.tensor(metas))


def feature_grid(pixel_points, features, stride):
    """Account for padded stage extent: H*stride may exceed the image size."""
    extent=pixel_points.new_tensor([features.shape[-1]*stride,features.shape[-2]*stride])
    return 2*pixel_points/extent-1
