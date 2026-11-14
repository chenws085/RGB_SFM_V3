"""Deterministic full-source views; scale image and natural background together."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

from dataloader.OriginalImagePairs import OriginalImagePairs
from dataloader.ObjectScalePairs import MEAN, STD


class OriginalImageViews(OriginalImagePairs):
    def __init__(self, root, indices, long_edge=224):
        super().__init__(root, indices, training=False)
        if not 1 <= long_edge <= 224:
            raise ValueError('Full image long edge must fit the 224 canvas')
        self.long_edge = long_edge

    def __getitem__(self, item):
        if self.long_edge == 224:
            return super().__getitem__(item)
        index = self.indices[item]
        image, (label, _) = self.base[index]
        image = image.convert('RGB')
        width, height = image.size
        factor = self.long_edge/max(width, height)
        w, h = max(1, round(width*factor)), max(1, round(height*factor))
        ox, oy = (224-w)//2, (224-h)//2
        canvas = np.broadcast_to(MEAN[:, 0, 0].numpy(), (224, 224, 3)).copy()
        canvas[oy:oy+h, ox:ox+w] = np.asarray(image.resize((w,h), Image.Resampling.BICUBIC), dtype='float32')/255
        tensor = (torch.from_numpy(canvas).permute(2,0,1)-MEAN)/STD
        path = Path(self.base.root)/'Annotations'/self.base.annotation_categories[label]/f'annotation_{self.base.index[index]:04d}.mat'
        box = loadmat(path)['box_coord'].ravel()
        bounds = np.array([max(0,box[2]-1),max(0,box[0]-1),min(width,box[3]),min(height,box[1])], dtype='float32')
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError('Invalid bbox')
        yy,xx = np.meshgrid((np.arange(8)+.5)/8,(np.arange(8)+.5)/8,indexing='ij')
        points = np.stack([xx,yy],-1).astype('float32')*(bounds[2:]-bounds[:2])+bounds[:2]
        points = points*np.array([w/width,h/height],dtype='float32')+np.array([ox,oy],dtype='float32')
        return dict(a=tensor,b=tensor,points_a=torch.from_numpy(points),points_b=torch.from_numpy(points),
                    region=torch.ones(1,8,8),label=label,index=index,geometry=torch.tensor([[w,h,ox,oy],[w,h,ox,oy]]))
