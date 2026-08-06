#!/usr/bin/env python3
"""Correctness foundation for the megakernel build. Verifies the fused-MLP megakernel computes the SAME
thing as a reference SwiGLU MLP (int4 weights). Perf is NOT reported here — the megakernel's payoff is
end-to-end (whole decode fused); isolated perf would mislead. This only answers: is the kernel correct?

    python bench/megakernel_test.py
"""
import os, torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128
DIM, INTER = 2560, 8192
RESID = 0.22

def bitcast_i32(p): return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).contiguous()
def quant_pack_contig(W, G=128):
    OUT, IN = W.shape
    Wg = W.float().view(OUT, IN // G, G); s = Wg.abs().amax(-1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(Wg / s), -8, 7)
    qu = (q + 8).to(torch.int64).view(OUT, IN // 8, 8)
    sh = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    Wq = bitcast_i32((qu << sh).sum(-1))
    Wdq = (q * s).view(OUT, IN).to(torch.bfloat16)
    return Wq, s.squeeze(-1).to(torch.bfloat16).contiguous(), Wdq

if __name__ == "__main__":
    dev = "cuda"
    print("building megakernel_mlp.cu ...")
    m = load(name="flint_mk", sources=[os.path.join(HERE, "kernels/megakernel_mlp.cu")],
             extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    torch.manual_seed(0)
    Wgu = torch.randn(2 * INTER, DIM, device=dev, dtype=torch.bfloat16) * 0.02
    Wd = torch.randn(DIM, INTER, device=dev, dtype=torch.bfloat16) * 0.02
    xn = torch.randn(DIM, device=dev, dtype=torch.bfloat16) * 0.5
    h0 = torch.randn(DIM, device=dev, dtype=torch.bfloat16)

    Wgu_q, Wgu_s, Wgu_dq = quant_pack_contig(Wgu)
    Wd_q, Wd_s, Wd_dq = quant_pack_contig(Wd)

    # reference (same dequantized weights -> isolates kernel correctness)
    gu = Wgu_dq.float() @ xn.float()
    gate, up = gu[:INTER], gu[INTER:]
    act = F.silu(gate) * up
    out = Wd_dq.float() @ act.float()
    h_ref = h0.float() + out * RESID

    h = h0.clone()
    m.mlp_mega(Wgu_q, Wgu_s, Wd_q, Wd_s, xn, h, RESID)
    torch.cuda.synchronize()
    rel = (h.float() - h_ref).norm().item() / (h_ref.norm().item() + 1e-9)
    cos = F.cosine_similarity(h.float(), h_ref, dim=0).item()
    print(f"fused-MLP megakernel  rel-L2 {rel:.4f}  cos {cos:.5f}  {'OK' if rel < 0.02 else 'BAD'}")
    if rel >= 0.02:
        print(f"  h[:4]  ={h[:4].tolist()}")
        print(f"  ref[:4]={h_ref[:4].tolist()}")
