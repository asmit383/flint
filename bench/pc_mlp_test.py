#!/usr/bin/env python3
"""Producer-consumer MLP: does overlapping the Wgu + Wd weight streams (barrier-free) beat the sequential
[gate/up | barrier | down] on MBU? Correctness-checked, prod/consumer split swept, vs sequential (int4_gemv_v)."""
import os, argparse, torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 128; DIM = 2560; INTER = 8192

def bit(p): return torch.where(p >= 2**31, p - 2**32, p).to(torch.int32).contiguous()
def pack(W, G=128):
    OUT, IN = W.shape; Wg = W.float().view(OUT, IN // G, G); s = Wg.abs().amax(-1, keepdim=True) / 7.0
    q = torch.clamp(torch.round(Wg / s), -8, 7)
    qu = (q + 8).to(torch.int64).view(OUT, IN // 8, 8)
    sh = (torch.arange(8, device=W.device) * 4).view(1, 1, 8)
    return bit((qu << sh).sum(-1)), s.squeeze(-1).to(torch.bfloat16).contiguous(), (q * s).view(OUT, IN).to(torch.bfloat16)

def gtime(fn, iters=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    import time; t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1e6

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--peak-bw", type=float, default=2.0e12); a = ap.parse_args()
    dev = "cuda"
    mv = load(name="flint_int4", sources=[os.path.join(HERE, "kernels/int4_gemv.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    mp = load(name="flint_pc", sources=[os.path.join(HERE, "kernels/pc_mlp.cu")], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    torch.manual_seed(0)
    Wgu = torch.randn(2 * INTER, DIM, device=dev, dtype=torch.bfloat16) * 0.02
    Wd = torch.randn(DIM, INTER, device=dev, dtype=torch.bfloat16) * 0.02
    xn = torch.randn(DIM, device=dev, dtype=torch.bfloat16) * 0.5
    Wgu_q, Wgu_s, Wgu_dq = pack(Wgu); Wd_q, Wd_s, Wd_dq = pack(Wd)

    gu = Wgu_dq.float() @ xn.float(); gate, up = gu[:INTER], gu[INTER:]
    act = F.silu(gate) * up; y_ref = Wd_dq.float() @ act

    y = mp.pc_mlp(Wgu_q, Wgu_s, Wd_q, Wd_s, xn, 0.66).float()
    rel = (y - y_ref).norm().item() / (y_ref.norm().item() + 1e-9)
    print(f"pc_mlp correctness  rel-L2 {rel:.4f}  {'OK' if rel < 0.02 else 'BAD'}")

    tot = (2 * INTER * DIM + DIM * INTER) * 0.5 + (DIM // G) * (2 * INTER) * 2 + (INTER // G) * DIM * 2
    # sequential baseline: gate/up GEMV + silu + down GEMV
    def seq():
        g = mv.int4_gemv_v(Wgu_q, Wgu_s, xn.view(DIM), 6)
        gg, uu = g[:INTER], g[INTER:]; ac = (F.silu(gg.float()) * uu.float()).to(torch.bfloat16)
        return mv.int4_gemv_v(Wd_q, Wd_s, ac.view(INTER), 8)
    ts = gtime(seq); print(f"\n  sequential (2 GEMV+silu)  {ts:7.1f} us   MBU {100*tot/(ts*1e-6)/a.peak_bw:4.1f}%")
    for pf in [0.5, 0.6, 0.66, 0.75, 0.8]:
        tc = gtime(lambda pf=pf: mp.pc_mlp(Wgu_q, Wgu_s, Wd_q, Wd_s, xn, pf))
        print(f"  pc_mlp prod={pf:.2f}       {tc:7.1f} us   MBU {100*tot/(tc*1e-6)/a.peak_bw:4.1f}%")
