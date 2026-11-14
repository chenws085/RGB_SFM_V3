"""Geometry, reproducibility, split and real-cutout overfit checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from dataloader.ObjectScalePairs import ObjectScalePairs,feature_grid
from models.ObjectScaleViT import ObjectScaleViT,LocalContentMerging,paired_loss,corresponding_regions

torch.set_num_threads(2);torch.manual_seed(42)
# Padded 4x4 stage spans 256 pixels, not 224. Coordinate ramp must recover pixels.
stride=64
axis=(torch.arange(4)+.5)*stride
y,x=torch.meshgrid(axis,axis,indexing='ij');ramp=torch.stack([x,y])[None]
points=torch.tensor([[[[80.,90.],[120.,160.]]]])
sampled=F.grid_sample(ramp,feature_grid(points,ramp,stride),align_corners=False)
assert torch.allclose(sampled.permute(0,2,3,1),points,atol=1e-4)
# Interior merging support is centered on the parent's nominal coordinate.
for factor in [2,3]:
 merge=LocalContentMerging(2,factor,factor)
 centers=(torch.arange(12)+.5)
 cy,cx=torch.meshgrid(centers,centers,indexing='ij')
 coords=torch.stack([cx,cy],-1).reshape(1,144,2)
 with torch.no_grad():
  merge.score.weight.zero_();merge.score.bias.zero_()
  captured=[]
  hook=merge.norm.register_forward_pre_hook(lambda module,args: captured.append(args[0].detach()))
  merge(coords,12,12);hook.remove()
  pooled=captured[-1].reshape(1,12//factor,12//factor,2)
  assert torch.allclose(pooled[0,1,1],torch.tensor([1.5*factor,1.5*factor]))
ds=ObjectScalePairs('data',[0,436,1000,5000],seed=42)
samples=[ds[i] for i in range(4)]
assert torch.equal(samples[0]['a'],ds[0]['a'])
assert samples[0]['region'].sum()>0
a=torch.stack([s['a'] for s in samples]);b=torch.stack([s['b'] for s in samples])
labels=torch.tensor([s['label'] for s in samples]);mask=torch.stack([s['region'] for s in samples])
pa=torch.stack([s['points_a'] for s in samples]);pb=torch.stack([s['points_b'] for s in samples])
for content in [False,True]:
 torch.manual_seed(42)
 model=ObjectScaleViT(img_size=224,patch_size=8,embed_dims=[8,16],num_heads=[2,2],depths=[1,1],
                      num_classes=101,drop_rate=0,drop_path_rate=0,content_merging=content)
 opt=torch.optim.AdamW(model.parameters(),lr=.01)
 losses=[]
 for step in range(50):
  la,fa=model(a,True);lb,fb=model(b,True)
  ra=corresponding_regions(fa[1],feature_grid(pa,fa[1],16),mask)
  rb=corresponding_regions(fb[1],feature_grid(pb,fb[1],16),mask)
  loss=paired_loss(la,lb,labels,ra,rb,.2,.2)
  opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
 assert losses[-1]<losses[0]*.3,(content,losses[0],losses[-1])
 assert (la.argmax(1)==labels).all() and (lb.argmax(1)==labels).all()
 print(dict(content_merging=content,initial_loss=losses[0],final_loss=losses[-1],accuracy=1.))
print('PASS: physical feature coordinates including padding, deterministic real cutouts, both branches overfit four paired sources')

# Residual branch starts at the exact baseline function and can learn its gate.
args=dict(img_size=32,patch_size=4,embed_dims=[8,16],num_heads=[2,2],depths=[1,1],num_classes=3,drop_rate=0,drop_path_rate=0)
torch.manual_seed(42);base=ObjectScaleViT(**args).eval()
torch.manual_seed(42);hybrid=ObjectScaleViT(**args,hybrid_merging=True).eval()
x=torch.randn(2,3,32,32)
assert torch.equal(base(x),hybrid(x))
hybrid(x).square().mean().backward()
assert torch.isfinite(hybrid.merges[0].gate.grad) and hybrid.merges[0].gate.grad.abs()>0
print('PASS: residual content branch baseline parity and nonzero gate gradient')
