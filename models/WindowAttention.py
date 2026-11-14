"""Unshifted window MHA retaining the original Attention's parameter names."""
import torch
from torch import nn
from torch.nn import functional as F


class LocalWindowAttention(nn.Module):
    def __init__(self, original, height, width, window_size=7, offset=0):
        super().__init__()
        if window_size < 1:
            raise ValueError('window_size must be positive')
        self.height, self.width = height, width
        self.window_size = min(window_size, height, width)
        if offset < 0 or offset >= window_size:
            raise ValueError("Offset must lie within the requested window")
        self.offset = offset if self.window_size < max(height,width) else 0
        self.num_heads, self.scale = original.num_heads, original.scale
        # Reuse tensors, rather than initializing additional trainable parameters.
        self.qkv, self.attn_drop = original.qkv, original.attn_drop
        self.proj, self.proj_drop = original.proj, original.proj_drop
        ws = self.window_size
        self.padded_height = ((height+self.offset+ws-1)//ws)*ws
        self.padded_width = ((width+self.offset+ws-1)//ws)*ws
        valid = torch.zeros(1,self.padded_height,self.padded_width,1,dtype=torch.bool)
        valid[:,self.offset:self.offset+height,self.offset:self.offset+width] = True
        valid = self._partition(valid).squeeze(-1)
        assert valid.any(-1).all()
        self.register_buffer('_valid_keys',valid,persistent=False)

    def _partition(self, x):
        b,h,w,c = x.shape
        ws = self.window_size
        return x.reshape(b,h//ws,ws,w//ws,ws,c).permute(0,1,3,2,4,5).reshape(-1,ws*ws,c)

    def _restore(self, x, batch):
        ws,h,w = self.window_size,self.padded_height,self.padded_width
        c = x.shape[-1]
        x = x.reshape(batch,h//ws,w//ws,ws,ws,c).permute(0,1,3,2,4,5).reshape(batch,h,w,c)
        return x[:,self.offset:self.offset+self.height,self.offset:self.offset+self.width].reshape(batch,self.height*self.width,c)

    def forward(self, x):
        b,n,c = x.shape
        if n != self.height*self.width:
            raise ValueError('Input token count does not match configured stage grid')
        image = x.reshape(b,self.height,self.width,c)
        image = F.pad(image,(0,0,self.offset,self.padded_width-self.width-self.offset,self.offset,self.padded_height-self.height-self.offset))
        windows = self._partition(image)
        bw,m,_ = windows.shape
        qkv = self.qkv(windows).reshape(bw,m,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        logits = (q@k.transpose(-2,-1))*self.scale
        valid = self._valid_keys.repeat(b,1)
        logits = logits.masked_fill(~valid[:,None,None,:],-torch.inf)
        heads = self.attn_drop(logits.softmax(-1))@v
        joined = self._restore(heads.transpose(1,2).reshape(bw,m,c),b)
        # Preserve the analysis interface and original whole-grid token ordering.
        self.individual_heads_output = joined.reshape(b,n,self.num_heads,c//self.num_heads)
        self.pre_projection_features = joined
        return self.proj_drop(self.proj(joined))


class SlidingNeighborhoodAttention(nn.Module):
    """Overlapping centered neighborhoods, with no wraparound or position bias."""
    def __init__(self, original, height, width, window_size=7):
        super().__init__()
        if window_size < 1:
            raise ValueError('Sliding neighborhood size must be positive')
        self.height, self.width, self.window_size = height, width, window_size
        self.num_heads, self.scale = original.num_heads, original.scale
        self.qkv, self.attn_drop = original.qkv, original.attn_drop
        self.proj, self.proj_drop = original.proj, original.proj_drop
        self.pad_before = (window_size-1)//2
        self.pad_after = window_size-1-self.pad_before
        valid = self._unfold(torch.ones(1,1,height,width))
        self.register_buffer('_valid_keys',valid.transpose(1,2).bool(),persistent=False)

    def _unfold(self, spatial):
        # Even widths use offsets [-1,0,1,2] for k=4; no symmetric integer center.
        padded=F.pad(spatial,(self.pad_before,self.pad_after,self.pad_before,self.pad_after))
        return F.unfold(padded,self.window_size)

    def forward(self, x):
        b,n,c = x.shape
        if n != self.height*self.width:
            raise ValueError('Input token count does not match stage grid')
        heads,dim = self.num_heads,c//self.num_heads
        q,k,v = self.qkv(x).reshape(b,n,3,heads,dim).permute(2,0,3,1,4).unbind(0)
        def neighborhoods(t):
            spatial=t.permute(0,1,3,2).reshape(b,c,self.height,self.width)
            return self._unfold(spatial).reshape(b,heads,dim,self.window_size**2,n).permute(0,1,4,3,2)
        kn,vn=neighborhoods(k),neighborhoods(v)
        scores=torch.einsum('bhnd,bhnkd->bhnk',q,kn)*self.scale
        scores=scores.masked_fill(~self._valid_keys[:,None],-torch.inf)
        out=torch.einsum('bhnk,bhnkd->bhnd',self.attn_drop(scores.softmax(-1)),vn)
        joined=out.transpose(1,2).reshape(b,n,c)
        self.individual_heads_output=joined.reshape(b,n,heads,dim)
        self.pre_projection_features=joined
        return self.proj_drop(self.proj(joined))


def configure_attention(model, attention_mode='global', window_size=7,
                        attention_stages=None, window_offset=0):
    """Replace only chosen stages; retain tensor names and initialization."""
    if attention_mode not in ('global','window','sliding'):
        raise ValueError('Unsupported attention mode')
    selected=set(range(len(model.stages)) if attention_stages is None else attention_stages)
    if not selected.issubset(range(len(model.stages))):
        raise ValueError('Invalid attention stage')
    if attention_mode=='global':
        return model
    height,width=model.patch_embed.grid_h,model.patch_embed.grid_w
    for stage,blocks in enumerate(model.stages):
        if stage in selected:
            for block in blocks:
                if attention_mode=='window':
                    block.attn=LocalWindowAttention(block.attn,height,width,window_size,window_offset)
                else:
                    block.attn=SlidingNeighborhoodAttention(block.attn,height,width,window_size)
        merge=model.merges[stage]
        if not isinstance(merge,nn.Identity):
            height,width=model.get_next_resolution(height,width,merge.m_h,merge.m_w)
    return model
