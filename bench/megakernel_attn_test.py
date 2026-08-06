#!/usr/bin/env python3
"""Correctness of the attention megakernel vs a matched reference (rmsnorm+QKV+RoPE+GQA+SDPA+Wo+residual),
with a pre-filled KV cache so real attention-over-history is exercised. Perf is end-to-end, not here."""
import os, torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128; DIM = 2560; NH = 40; NKV = 8; HD = 64; SCALE = 0.015625; ROPE = 1e7; RESID = 0.22
POS = 5   # current position; cache pre-filled for t=0..4

def bitcast_i32(p): return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).contiguous()
def qpc(W, G=128):
    OUT, IN = W.shape
    Wg = W.float().view(OUT, IN // G, G); s = Wg.abs().amax(-1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(Wg / s), -8, 7)
    qu = (q + 8).to(torch.int64).view(OUT, IN // 8, 8)
    sh = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    return bitcast_i32((qu << sh).sum(-1)), s.squeeze(-1).to(torch.bfloat16).contiguous(), (q * s).view(OUT, IN).to(torch.bfloat16)

def rope(x, pos):  # x [heads, HD], rotate-half, matches kernel
    i = torch.arange(HD // 2, device=x.device, dtype=torch.float32)
    freq = ROPE ** (-2.0 * i / HD); ang = pos * freq
    c, sn = torch.cos(ang), torch.sin(ang)
    x0, x1 = x[..., :HD // 2].float(), x[..., HD // 2:].float()
    return torch.cat([x0 * c - x1 * sn, x1 * c + x0 * sn], -1)

if __name__ == "__main__":
    dev = "cuda"
    m = load(name="flint_mka", sources=[os.path.join(HERE, "kernels/megakernel_attn.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    torch.manual_seed(0)
    QKV = DIM + 2 * NKV * HD
    Wqkv = torch.randn(QKV, DIM, device=dev, dtype=torch.bfloat16) * 0.02
    Wo = torch.randn(DIM, DIM, device=dev, dtype=torch.bfloat16) * 0.02
    nw = torch.randn(DIM, device=dev, dtype=torch.bfloat16).abs() + 0.5
    h0 = torch.randn(DIM, device=dev, dtype=torch.bfloat16)
    maxseq = 16
    kc0 = torch.randn(maxseq, NKV, HD, device=dev, dtype=torch.bfloat16) * 0.1
    vc0 = torch.randn(maxseq, NKV, HD, device=dev, dtype=torch.bfloat16) * 0.1

    Wqkv_q, Wqkv_s, Wqkv_dq = qpc(Wqkv); Wo_q, Wo_s, Wo_dq = qpc(Wo)

    # reference
    rms = torch.rsqrt(h0.float().pow(2).mean() + 1e-5)
    xn = (h0.float() * rms * nw.float())
    qkv = Wqkv_dq.float() @ xn
    q = qkv[:DIM].view(NH, HD); k = qkv[DIM:DIM + NKV * HD].view(NKV, HD); v = qkv[DIM + NKV * HD:].view(NKV, HD)
    q = rope(q, POS); k = rope(k, POS)
    kc = kc0.clone().float(); vc = vc0.clone().float()
    kc[POS] = k; vc[POS] = v
    ao = torch.zeros(NH, HD, device=dev)
    for qh in range(NH):
        kv = qh // (NH // NKV)
        sc = (q[qh] @ kc[:POS + 1, kv].T) * SCALE
        w = torch.softmax(sc, -1)
        ao[qh] = w @ vc[:POS + 1, kv]
    h_ref = h0.float() + (Wo_dq.float() @ ao.view(DIM)) * RESID

    # kernel
    h = h0.clone(); kc_k = kc0.clone(); vc_k = vc0.clone()
    m.attn_mega(Wqkv_q, Wqkv_s, Wo_q, Wo_s, nw, h, kc_k, vc_k, POS, SCALE, ROPE, RESID)
    torch.cuda.synchronize()
    rel = (h.float() - h_ref).norm().item() / (h_ref.norm().item() + 1e-9)
    cos = F.cosine_similarity(h.float(), h_ref, dim=0).item()
    print(f"attention megakernel  rel-L2 {rel:.4f}  cos {cos:.5f}  {'OK' if rel < 0.03 else 'BAD'}")
    if rel >= 0.03:
        print(f"  h[:4]  ={h[:4].tolist()}")
        print(f"  ref[:4]={h_ref[:4].tolist()}")
