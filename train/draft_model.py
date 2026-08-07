"""Shared EAGLE draft head (imported by train + eval so they can't diverge).
One transformer block: merge(target feature, next-token embedding) -> RoPE self-attn -> SwiGLU ->
predict next feature. RoPE matches Granite (the top fix — its features are deeply positional)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

def rope_cache(positions, head_dim, base, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(positions.float(), inv)                # [T, hd/2]
    emb = torch.cat([freqs, freqs], -1)                        # [T, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], -1)

def apply_rope(x, cos, sin):                                   # x [B,nh,T,hd], cos/sin [T,hd]
    return x * cos + rotate_half(x) * sin

class DraftHead(nn.Module):
    # EAGLE-3-style multi-layer fusion: input is the target's fused (low+mid+high) feature [fuse*dim] plus the
    # next-token embedding; the block runs in dim; fout predicts the next FUSED feature so it can autoregress.
    # The token head reads the HIGH slice (last dim) of the fused feature. fuse=1 recovers last-layer-only.
    def __init__(self, dim, nh, inter, fuse=1, rope_base=1e7):
        super().__init__()
        self.nh, self.hd, self.rope_base = nh, dim // nh, rope_base
        self.dim, self.fuse, self.fdim = dim, fuse, fuse * dim
        self.merge = nn.Linear(self.fdim + dim, dim, bias=False)
        self.n1 = nn.Parameter(torch.ones(dim)); self.n2 = nn.Parameter(torch.ones(dim))
        self.q = nn.Linear(dim, dim, bias=False); self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False); self.o = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, inter, bias=False); self.up = nn.Linear(dim, inter, bias=False)
        self.down = nn.Linear(inter, dim, bias=False)
        self.fout = nn.Linear(dim, self.fdim, bias=False)     # predict next FUSED feature

    def forward(self, feat, emb, positions=None):             # feat [B,T,fdim], emb [B,T,dim]
        B, T, _ = feat.shape
        if positions is None:
            positions = torch.arange(T, device=feat.device)
        x = self.merge(torch.cat([feat, emb], -1))            # [B,T,dim]
        h = rmsnorm(x, self.n1)
        q = self.q(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        cos, sin = rope_cache(positions, self.hd, self.rope_base, feat.device, feat.dtype)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(att.transpose(1, 2).reshape(B, T, self.dim))
        h = rmsnorm(x, self.n2)
        h = x + self.down(F.silu(self.gate(h)) * self.up(h))  # [B,T,dim]
        return self.fout(h)                                   # [B,T,fdim] predicted next fused feature
