"""Opt-in object correspondence prototype; existing checkpoints remain unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from models.MergingViT import MergingViT


class LocalContentMerging(nn.Module):
    """Fixed output grid, overlapping content-weighted neighborhoods (not equivariant)."""
    def __init__(self, dim, m_h=2, m_w=2):
        super().__init__()
        self.m_h, self.m_w = m_h, m_w
        self.kernel = (m_h+2, m_w+2)
        self.score = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, 2*dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        oh, ow = (H+self.m_h-1)//self.m_h, (W+self.m_w-1)//self.m_w
        # Symmetric one-child halo around each parent cell's nominal support.
        ph, pw = 1, 1
        right = (ow-1)*self.m_w+self.kernel[1]-W-pw
        bottom = (oh-1)*self.m_h+self.kernel[0]-H-ph
        padding = (pw, right, ph, bottom)
        image = x.transpose(1, 2).reshape(B, C, H, W)
        patches = F.unfold(F.pad(image, padding), self.kernel, stride=(self.m_h,self.m_w))
        patches = patches.reshape(B,C,-1,oh*ow).permute(0,3,2,1)
        valid = F.unfold(F.pad(x.new_ones(B,1,H,W), padding), self.kernel,
                         stride=(self.m_h,self.m_w)).transpose(1,2).bool()
        scores = self.score(self.norm(patches)).squeeze(-1).masked_fill(~valid, -torch.inf)
        weights = scores.softmax(-1)
        # Detached tracing avoids retaining training graphs; offsets follow unfold order.
        self.last_weights = weights.detach()
        self.last_valid = valid.detach()
        pooled = (patches*weights.unsqueeze(-1)).sum(2)
        return self.reduction(self.norm(pooled)), oh, ow


class ResidualContentMerging(nn.Module):
    """Preserve ordered local details; learn a gated content pooling residual."""
    def __init__(self, base, dim):
        super().__init__()
        self.base = base
        self.m_h, self.m_w = base.m_h, base.m_w
        self.content = LocalContentMerging(dim, self.m_h, self.m_w)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x, H, W):
        original, h, w = self.base(x,H,W)
        content, ch, cw = self.content(x,H,W)
        assert (h,w)==(ch,cw)
        return original+self.gate.tanh()*content, h, w


class ObjectScaleViT(MergingViT):
    def __init__(self, *args, content_merging=False, hybrid_merging=False,
                 attention_mode='global', window_size=7, attention_stages=None, window_offset=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_mode = attention_mode
        self.window_size = window_size
        from models.WindowAttention import configure_attention
        configure_attention(self,attention_mode,window_size,attention_stages,window_offset)
        if content_merging or hybrid_merging:
            for i, merge in enumerate(self.merges):
                if isinstance(merge, nn.Identity):
                    continue
                dim = self.pos_embeds[i].shape[-1]
                self.merges[i] = (ResidualContentMerging(merge, dim) if hybrid_merging else
                                  LocalContentMerging(dim, merge.m_h, merge.m_w))

    def forward(self, x, return_stages=False):
        x = self.patch_embed(x)
        H, W = self.patch_embed.grid_h, self.patch_embed.grid_w
        stages = []
        for i, blocks in enumerate(self.stages):
            x = x+self.pos_embeds[i]
            for block in blocks:
                x = block(x)
            stages.append(x.transpose(1,2).reshape(x.shape[0],-1,H,W))
            if not isinstance(self.merges[i], nn.Identity):
                x,H,W = self.merges[i](x,H,W)
        logits = self.head(self.head_drop(self.norm_final(x.mean(1))))
        return (logits, stages) if return_stages else logits


def render_object(foreground, mask, background, theta):
    """theta [B,2,3] maps view coordinates to source coordinates, align_corners=False.

    Inputs share a canvas. Background is never transformed. Premultiplied alpha
    prevents source background leaking into interpolated object boundaries.
    """
    grid = F.affine_grid(theta, foreground.shape, align_corners=False)
    alpha = F.grid_sample(mask, grid, align_corners=False).clamp(0,1)
    color = F.grid_sample(foreground*mask, grid, align_corners=False)
    return color+background*(1-alpha), alpha


def corresponding_regions(features, source_to_view_grid, source_mask):
    """Sample at common source points; grid [B,h,w,2] uses normalized view coordinates.

    Caller must supply common visible object/part mask [B,1,h,w], excluding
    clipped areas and padded feature cells. Pooling does not prove token invariance.
    """
    aligned = F.grid_sample(features, source_to_view_grid, align_corners=False)
    mass = source_mask.sum((2,3))
    if (mass <= 0).any():
        raise ValueError('Correspondence region must contain visible object pixels')
    return (aligned*source_mask).sum((2,3))/mass


def paired_loss(logits_a, logits_b, labels, region_a, region_b, pred_weight=1., region_weight=1.):
    ce = (F.cross_entropy(logits_a,labels)+F.cross_entropy(logits_b,labels))/2
    # Symmetric prediction consistency with a detached target in each direction.
    pred = (F.kl_div(logits_a.log_softmax(-1),logits_b.detach().softmax(-1),reduction='batchmean')+
            F.kl_div(logits_b.log_softmax(-1),logits_a.detach().softmax(-1),reduction='batchmean'))/2
    region = (1-F.cosine_similarity(region_a,region_b,dim=-1)).mean()
    return ce+pred_weight*pred+region_weight*region
