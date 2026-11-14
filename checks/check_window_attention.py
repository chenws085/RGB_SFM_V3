"""Attention isolation, padding correctness, and legacy-model compatibility."""
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.MergingViT import Attention, MergingViT
from models.ObjectScaleViT import ObjectScaleViT
from models.WindowAttention import LocalWindowAttention
from scripts.train_caltech_mergingvit import MODEL_ARGS


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    a=Attention(8,2).eval()
    w=LocalWindowAttention(a,4,4,7).eval()
    x=torch.randn(2,16,8)
    assert torch.allclose(a(x),w(x),atol=1e-6)
    a=Attention(8,2).eval()
    w=LocalWindowAttention(a,4,4,2).eval()
    x=torch.randn(1,16,8)
    y=x.clone();y[:,15]+=10
    assert torch.allclose(w(x)[:,0],w(y)[:,0],atol=1e-7)
    assert not torch.allclose(a(x)[:,0],a(y)[:,0],atol=1e-5)
    a=Attention(8,2).eval()
    w=LocalWindowAttention(a,5,7,4).eval()
    x=torch.randn(2,35,8,requires_grad=True)
    actual=w(x)
    reference=torch.zeros_like(x).reshape(2,5,7,8)
    for top in [0,4]:
        for left in [0,4]:
            tile=x.reshape(2,5,7,8)[:,top:top+4,left:left+4]
            h,ww=tile.shape[1:3]
            reference[:,top:top+4,left:left+4]=a(tile.reshape(2,-1,8)).reshape(2,h,ww,8)
    assert torch.allclose(actual,reference.reshape_as(actual),atol=1e-6)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    torch.manual_seed(42);g=ObjectScaleViT(**MODEL_ARGS).eval()
    torch.manual_seed(42);l=ObjectScaleViT(**MODEL_ARGS,attention_mode='window').eval()
    assert g.state_dict().keys()==l.state_dict().keys()
    assert all(torch.equal(v,l.state_dict()[k]) for k,v in g.state_dict().items())
    ref=MergingViT(**MODEL_ARGS).eval();ref.load_state_dict(g.state_dict())
    with torch.inference_mode():
        image=torch.randn(1,3,224,224)
        assert torch.equal(g(image),ref(image))
    print(json.dumps(dict(passed=['one-window global parity','window boundary isolation with global perturbation control',
        'padding matches valid-token-only window reference','finite padding gradients','identical initial state and parameter count',
        'legacy global model exact output parity'],effective_windows=[b[0].attn.window_size for b in l.stages],
        parameters=sum(p.numel() for p in l.parameters())),indent=2))


if __name__=='__main__':
    main()
