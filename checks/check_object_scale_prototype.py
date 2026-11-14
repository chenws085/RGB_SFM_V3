"""CPU correctness checks; no claim about learned robustness."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.MergingViT import MergingViT
from models.ObjectScaleViT import (ObjectScaleViT, LocalContentMerging,
                                  render_object, corresponding_regions, paired_loss)
torch.set_num_threads(1)
torch.manual_seed(42)
args=dict(img_size=32,patch_size=4,embed_dims=[8,16,32],num_heads=[2,2,4],
          depths=[1,1,1],num_classes=3,drop_rate=0.,drop_path_rate=0.)
a=MergingViT(**args).eval(); b=ObjectScaleViT(**args).eval()
b.load_state_dict(a.state_dict(),strict=True)
x=torch.randn(2,3,32,32)
assert torch.equal(a(x),b(x)), 'Baseline parity'
for H,W in [(7,5),(1,1),(8,8)]:
 m=LocalContentMerging(8)
 y,h,w=m(torch.randn(2,H*W,8),H,W)
 assert y.shape==(2,h*w,16) and torch.isfinite(y).all()
 assert torch.allclose(m.last_weights.sum(-1),torch.ones(2,h*w))
 assert (m.last_weights[~m.last_valid]==0).all()
mask=torch.zeros(2,1,32,32); mask[:,:,8:24,8:24]=1
bg=torch.randn_like(x)
theta=torch.eye(2,3).unsqueeze(0).repeat(2,1,1)
view,alpha=render_object(x,mask,bg,theta)
assert torch.allclose(view,x*mask+bg*(1-mask),atol=1e-5)
theta[:,0,0]=2; theta[:,1,1]=2 # half-sized object, unchanged background
small,sa=render_object(x,mask,bg,theta)
assert sa.sum()<alpha.sum()
assert torch.equal(small.masked_select(sa.expand_as(x)==0),bg.masked_select(sa.expand_as(x)==0))
m=ObjectScaleViT(**args,content_merging=True)
la,fa=m(view,True); lb,fb=m(small,True)
grid=torch.zeros(2,2,2,2); valid=torch.ones(2,1,2,2)
ra=corresponding_regions(fa[1],grid,valid); rb=corresponding_regions(fb[1],grid,valid)
loss=paired_loss(la,lb,torch.tensor([0,1]),ra,rb);loss.backward()
assert torch.isfinite(loss)
assert m.merges[0].score.weight.grad is not None
assert torch.isfinite(m.merges[0].score.weight.grad).all()
print('PASS: baseline parity, odd/tiny grids, padding, fixed background, object shrink, finite loss and merge gradients')
