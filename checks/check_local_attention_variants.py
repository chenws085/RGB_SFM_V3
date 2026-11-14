"""Dense masked references for neighborhood geometry and offset-window edges."""
import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.MergingViT import Attention,MergingViT
from models.WindowAttention import LocalWindowAttention,SlidingNeighborhoodAttention,configure_attention
from scripts.train_caltech_mergingvit import MODEL_ARGS

torch.set_num_threads(4);torch.manual_seed(42);passed=[]
for kind,h,w,ws,off in [('window',5,7,4,2),('window',4,4,7,0),('window',14,14,4,0),('sliding',5,7,4,0),('sliding',5,7,3,0),('sliding',4,4,7,0)]:
 original=Attention(8,2).eval();layer=(LocalWindowAttention(original,h,w,ws,off) if kind=='window' else SlidingNeighborhoodAttention(original,h,w,ws)).eval()
 x=torch.randn(2,h*w,8,requires_grad=True);q,k,v=original.qkv(x).reshape(2,h*w,3,2,4).permute(2,0,3,1,4).unbind(0)
 yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');xy=torch.stack([yy.flatten(),xx.flatten()],1)
 delta=xy[None,:]-xy[:,None]
 mask=((delta>=-((ws-1)//2)) & (delta<=ws//2)).all(-1) if kind=='sliding' else (((xy[:,None]+layer.offset)//layer.window_size)==((xy[None,:]+layer.offset)//layer.window_size)).all(-1)
 score=(q@k.transpose(-2,-1))*original.scale;score=score.masked_fill(~mask,-torch.inf)
 reference=original.proj((score.softmax(-1)@v).transpose(1,2).reshape_as(x));actual=layer(x)
 assert torch.allclose(actual,reference,atol=1e-6),(kind,(actual-reference).abs().max())
 actual.square().mean().backward();assert torch.isfinite(x.grad).all();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())
 passed.append([kind,h,w,ws,off])
variants=[('window',4,None,0),('window',14,None,0),('window',7,None,3),('window',7,[0],0),('sliding',7,[0,1],0),('sliding',4,[0,1],0)]
for mode,ws,stages,offset in variants:
 torch.manual_seed(42);baseline=MergingViT(**MODEL_ARGS);torch.manual_seed(42);model=MergingViT(**MODEL_ARGS);configure_attention(model,mode,ws,stages,offset)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items())
 model.eval();loss=model(torch.randn(2,3,224,224)).square().mean();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
print(json.dumps({'dense_reference_and_gradients':passed,'initialization_and_full_model_backward':variants,'parameters':sum(p.numel() for p in model.parameters())},indent=2))
