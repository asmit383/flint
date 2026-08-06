// flint FULL DECODE megakernel — the whole Granite-3B decode step in ONE persistent cooperative launch.
// Loops NL layers (attn block + mlp block), residual h stays in an L2-resident buffer across ALL layers
// (never a DRAM round-trip), then final rmsnorm + LM-head GEMV -> logits. This is the end-to-end kernel:
// one launch advances the whole forward pass for one token. Blocks are the validated attn/mlp bodies.
//
// Granite-3B: DIM=2560 INTER=8192 NH=40 NKV=8 HD=64 VOCAB=100352.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define HD 64
#define NH 40
#define NKV 8
#ifdef HALFBYTES
#define RSTEP 2      // diagnostic: skip every other GEMV row -> read ~half the weight bytes
#else
#define RSTEP 1
#endif

__device__ __forceinline__ float wgemv_row(const uint32_t* __restrict__ Wrow,
    const __nv_bfloat16* __restrict__ srow, const __nv_bfloat16* __restrict__ x, int IN, int lane) {
  const int ncols = IN >> 3; float acc = 0.f;            // single accumulator: keeps registers low (occupancy)
  for (int col = lane; col < ncols; col += 32) {
    const uint32_t w = Wrow[col]; const float sc = __bfloat162float(srow[col >> 4]);
    const int4 xr = __ldg(reinterpret_cast<const int4*>(&x[col << 3]));   // 8 bf16 = one 128-bit load
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xr);
    #pragma unroll
    for (int m = 0; m < 8; m++) acc += (float((w >> (4 * m)) & 0xF) - 8.0f) * sc * __bfloat162float(xb[m]);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;
}

// rmsnorm(h) -> xn.  REDUNDANT per-block reduction: each block sums-of-squares over ALL of h locally
// (h is tiny + L2-hot), reduced with a cheap __syncthreads (block barrier), NOT a grid barrier. Then
// each block normalizes its grid-partitioned slice. 3 grid.sync() -> 1. (Assumes h is ready, i.e. the
// previous phase ended with a grid.sync.) This trades a little redundant math for killing ~162 barriers.
__device__ void rmsnorm(cg::grid_group& grid, const __nv_bfloat16* h, const __nv_bfloat16* nw,
                        __nv_bfloat16* xn, float* red, int DIM, int tid, int nthreads) {
  float ss = 0.f;
  for (int i = threadIdx.x; i < DIM; i += blockDim.x) { float v = __bfloat162float(h[i]); ss += v * v; }
  __shared__ float bs[32]; __shared__ float rms_s;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw_b = blockDim.x >> 5;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
  if (lane == 0) bs[w] = ss;
  __syncthreads();
  if (w == 0) {
    float s = (lane < nw_b) ? bs[lane] : 0.f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) rms_s = rsqrtf(s / DIM + 1e-5f);
  }
  __syncthreads();
  const float r = rms_s;
  for (int i = tid; i < DIM; i += nthreads) xn[i] = __float2bfloat16(__bfloat162float(h[i]) * r * __bfloat162float(nw[i]));
  grid.sync();
}

__device__ void attn_block(cg::grid_group& grid,
    const uint32_t* Wqkv, const __nv_bfloat16* s_qkv, const uint32_t* Wo, const __nv_bfloat16* s_o,
    const __nv_bfloat16* nw, __nv_bfloat16* h, __nv_bfloat16* kc, __nv_bfloat16* vc,
    __nv_bfloat16* xn, __nv_bfloat16* qkv, __nv_bfloat16* ao, float* red,
    float* pm, float* pd, float* pacc, int WPH,          // flash-decode partials: WPH warps per head
    int DIM, int pos, float scale, float rope_base, float resid,
    int tid, int nthreads, int warp, int lane, int nwarps) {
  const int QKV = DIM + 2 * NKV * HD;
  rmsnorm(grid, h, nw, xn, red, DIM, tid, nthreads);
  for (int row = warp; row < QKV; row += RSTEP * nwarps) {
    const float r = wgemv_row(Wqkv + (size_t)row * (DIM >> 3), s_qkv + (size_t)row * (DIM >> 7), xn, DIM, lane);
    if (lane == 0) qkv[row] = __float2bfloat16(r);
  }
  grid.sync();
  const int total = NH + NKV;
  for (int hh = warp; hh < total; hh += nwarps) {
    __nv_bfloat16* base = (hh < NH) ? (qkv + hh * HD) : (qkv + DIM + (hh - NH) * HD);
    for (int i = lane; i < HD / 2; i += 32) {
      const float freq = powf(rope_base, -2.0f * i / HD), ang = pos * freq, c = cosf(ang), sn = sinf(ang);
      const float x0 = __bfloat162float(base[i]), x1 = __bfloat162float(base[i + HD / 2]);
      base[i] = __float2bfloat16(x0 * c - x1 * sn); base[i + HD / 2] = __float2bfloat16(x1 * c + x0 * sn);
    }
  }
  grid.sync();
  for (int i = tid; i < NKV * HD; i += nthreads) {
    kc[(size_t)pos * NKV * HD + i] = qkv[DIM + i];
    vc[(size_t)pos * NKV * HD + i] = qkv[DIM + NKV * HD + i];
  }
  grid.sync();
  const int T = pos + 1;
  // Phase 3a: flash partials. warp -> (head, sub); each warp handles STRIDED timesteps (sub, sub+WPH, ...)
  const int head = warp / WPH, sub = warp % WPH;
  if (head < NH) {
    const int kv = head / (NH / NKV); const __nv_bfloat16* q = qkv + head * HD;
    float mmax = -1e30f, denom = 0.f, outv[2] = {0.f, 0.f};
    for (int t = sub; t < T; t += WPH) {
      const __nv_bfloat16* kt = kc + ((size_t)t * NKV + kv) * HD; float s = 0.f;
      for (int d = lane; d < HD; d += 32) s += __bfloat162float(q[d]) * __bfloat162float(kt[d]);
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
      s = __shfl_sync(0xffffffffu, s, 0) * scale;
      const float m2 = fmaxf(mmax, s), corr = __expf(mmax - m2), w = __expf(s - m2);
      denom = denom * corr + w;
      const __nv_bfloat16* vt = vc + ((size_t)t * NKV + kv) * HD;
      #pragma unroll
      for (int j = 0; j < 2; j++) outv[j] = outv[j] * corr + w * __bfloat162float(vt[lane + j * 32]);
      mmax = m2;
    }
    const int pidx = head * WPH + sub;
    if (lane == 0) { pm[pidx] = mmax; pd[pidx] = denom; }        // UNnormalized partial (flash)
    pacc[(size_t)pidx * HD + lane] = outv[0]; pacc[(size_t)pidx * HD + lane + 32] = outv[1];
  }
  grid.sync();
  // Phase 3b: merge WPH partials per head (flash softmax merge)
  for (int qh = warp; qh < NH; qh += nwarps) {
    float gm = -1e30f;
    for (int s = 0; s < WPH; s++) gm = fmaxf(gm, pm[qh * WPH + s]);
    float gd = 0.f, acc0 = 0.f, acc1 = 0.f;
    for (int s = 0; s < WPH; s++) {
      const int pidx = qh * WPH + s; const float wt = __expf(pm[pidx] - gm);
      gd += pd[pidx] * wt;
      acc0 += pacc[(size_t)pidx * HD + lane] * wt; acc1 += pacc[(size_t)pidx * HD + lane + 32] * wt;
    }
    ao[qh * HD + lane] = __float2bfloat16(acc0 / gd); ao[qh * HD + lane + 32] = __float2bfloat16(acc1 / gd);
  }
  grid.sync();
  for (int row = warp; row < DIM; row += RSTEP * nwarps) {
    const float r = wgemv_row(Wo + (size_t)row * (DIM >> 3), s_o + (size_t)row * (DIM >> 7), ao, DIM, lane);
    if (lane == 0) h[row] = __float2bfloat16(__bfloat162float(h[row]) + r * resid);
  }
  grid.sync();
}

__device__ void mlp_block(cg::grid_group& grid,
    const __nv_bfloat16* nw, const uint32_t* Wgu, const __nv_bfloat16* s_gu,
    const uint32_t* Wd, const __nv_bfloat16* s_d, __nv_bfloat16* h,
    __nv_bfloat16* xn, __nv_bfloat16* gu, __nv_bfloat16* act, float* red,
    int DIM, int INTER, float resid, int tid, int nthreads, int warp, int lane, int nwarps) {
  const int INTER2 = 2 * INTER;
  rmsnorm(grid, h, nw, xn, red, DIM, tid, nthreads);
  for (int row = warp; row < INTER2; row += RSTEP * nwarps) {
    const float r = wgemv_row(Wgu + (size_t)row * (DIM >> 3), s_gu + (size_t)row * (DIM >> 7), xn, DIM, lane);
    if (lane == 0) gu[row] = __float2bfloat16(r);
  }
  grid.sync();
  for (int i = tid; i < INTER; i += nthreads) {
    const float g = __bfloat162float(gu[i]);
    act[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * __bfloat162float(gu[i + INTER]));
  }
  grid.sync();
  for (int row = warp; row < DIM; row += RSTEP * nwarps) {
    const float r = wgemv_row(Wd + (size_t)row * (INTER >> 3), s_d + (size_t)row * (INTER >> 7), act, INTER, lane);
    if (lane == 0) h[row] = __float2bfloat16(__bfloat162float(h[row]) + r * resid);
  }
  grid.sync();
}

__global__ void decode_mega(
    const uint32_t* Wqkv, const __nv_bfloat16* s_qkv, const uint32_t* Wo, const __nv_bfloat16* s_o,
    const __nv_bfloat16* n1, const uint32_t* Wgu, const __nv_bfloat16* s_gu,
    const uint32_t* Wd, const __nv_bfloat16* s_d, const __nv_bfloat16* n2,
    const __nv_bfloat16* nf, const uint32_t* Wlm, const __nv_bfloat16* s_lm,
    __nv_bfloat16* h, __nv_bfloat16* kc, __nv_bfloat16* vc,
    __nv_bfloat16* xn, __nv_bfloat16* qkv, __nv_bfloat16* gu, __nv_bfloat16* act, __nv_bfloat16* ao,
    float* red, float* pm, float* pd, float* pacc, float* logits,
    int DIM, int INTER, int NL, int VOCAB, int MAXSEQ, int pos, int WPH,
    float scale, float rope_base, float resid, float logits_scaling) {
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, nthreads = gridDim.x * blockDim.x;
  const int warp = tid >> 5, lane = tid & 31, nwarps = nthreads >> 5;
  const size_t qkv_st = (size_t)(DIM + 2 * NKV * HD) * (DIM >> 3), sqkv_st = (size_t)(DIM + 2 * NKV * HD) * (DIM >> 7);
  const size_t wo_st = (size_t)DIM * (DIM >> 3), so_st = (size_t)DIM * (DIM >> 7);
  const size_t wgu_st = (size_t)(2 * INTER) * (DIM >> 3), sgu_st = (size_t)(2 * INTER) * (DIM >> 7);
  const size_t wd_st = (size_t)DIM * (INTER >> 3), sd_st = (size_t)DIM * (INTER >> 7);
  const size_t kc_st = (size_t)MAXSEQ * NKV * HD;

  for (int l = 0; l < NL; l++) {
#ifndef NOATTN
    attn_block(grid, Wqkv + l * qkv_st, s_qkv + l * sqkv_st, Wo + l * wo_st, s_o + l * so_st,
               n1 + (size_t)l * DIM, h, kc + l * kc_st, vc + l * kc_st, xn, qkv, ao, red,
               pm, pd, pacc, WPH, DIM, pos, scale, rope_base, resid, tid, nthreads, warp, lane, nwarps);
#endif
#ifndef NOMLP
    mlp_block(grid, n2 + (size_t)l * DIM, Wgu + l * wgu_st, s_gu + l * sgu_st, Wd + l * wd_st, s_d + l * sd_st,
              h, xn, gu, act, red, DIM, INTER, resid, tid, nthreads, warp, lane, nwarps);
#endif
  }
  // final norm + LM head
  rmsnorm(grid, h, nf, xn, red, DIM, tid, nthreads);
#ifndef NOLM
  for (int row = warp; row < VOCAB; row += RSTEP * nwarps) {
    const float r = wgemv_row(Wlm + (size_t)row * (DIM >> 3), s_lm + (size_t)row * (DIM >> 7), xn, DIM, lane);
    if (lane == 0) logits[row] = r / logits_scaling;
  }
#endif
}

torch::Tensor decode_mega_launch(
    torch::Tensor Wqkv, torch::Tensor s_qkv, torch::Tensor Wo, torch::Tensor s_o, torch::Tensor n1,
    torch::Tensor Wgu, torch::Tensor s_gu, torch::Tensor Wd, torch::Tensor s_d, torch::Tensor n2,
    torch::Tensor nf, torch::Tensor Wlm, torch::Tensor s_lm,
    torch::Tensor h, torch::Tensor kc, torch::Tensor vc, int64_t pos,
    double scale, double rope_base, double resid, double logits_scaling) {
  const int DIM = h.size(0), NL = Wqkv.size(0), VOCAB = Wlm.size(0);
  const int INTER = Wd.size(2) * 8, MAXSEQ = kc.size(1);
  auto opt = h.options();
  auto xn = torch::empty({DIM}, opt), qkv = torch::empty({DIM + 2 * NKV * HD}, opt);
  auto gu = torch::empty({2 * INTER}, opt), act = torch::empty({INTER}, opt), ao = torch::empty({DIM}, opt);
  auto red = torch::zeros({1}, opt.dtype(torch::kFloat32));
  auto logits = torch::empty({VOCAB}, opt.dtype(torch::kFloat32));
  int nthreads=256,dev=0; cudaGetDevice(&dev);
  int bps=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, decode_mega, nthreads, 0);
  int nsm=0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  int grid=bps*nsm; int nwarps=grid*(nthreads>>5); int WPH=nwarps/NH; if(WPH<1)WPH=1;
  auto pm = torch::empty({NH * WPH}, opt.dtype(torch::kFloat32));
  auto pd = torch::empty({NH * WPH}, opt.dtype(torch::kFloat32));
  auto pacc = torch::empty({(int64_t)NH * WPH * HD}, opt.dtype(torch::kFloat32));
  auto P = [](torch::Tensor t){ return reinterpret_cast<const uint32_t*>(t.data_ptr<int32_t>()); };
  auto B = [](torch::Tensor t){ return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  auto Bm = [](torch::Tensor t){ return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  const uint32_t *wqkv=P(Wqkv),*wo=P(Wo),*wgu=P(Wgu),*wd=P(Wd),*wlm=P(Wlm);
  const __nv_bfloat16 *sqkv=B(s_qkv),*so=B(s_o),*n1p=B(n1),*sgu=B(s_gu),*sd=B(s_d),*n2p=B(n2),*nfp=B(nf),*slm=B(s_lm);
  __nv_bfloat16 *hp=Bm(h),*kcp=Bm(kc),*vcp=Bm(vc),*xnp=Bm(xn),*qkvp=Bm(qkv),*gup=Bm(gu),*actp=Bm(act),*aop=Bm(ao);
  float *redp=red.data_ptr<float>(),*lgp=logits.data_ptr<float>();
  float *pmp=pm.data_ptr<float>(),*pdp=pd.data_ptr<float>(),*paccp=pacc.data_ptr<float>();
  int D=DIM,I=INTER,nl=NL,V=VOCAB,ms=MAXSEQ,p2=pos,wph=WPH; float sc=scale,rb=rope_base,rm=resid,ls=logits_scaling;
  void* args[]={&wqkv,&sqkv,&wo,&so,&n1p,&wgu,&sgu,&wd,&sd,&n2p,&nfp,&wlm,&slm,&hp,&kcp,&vcp,
                &xnp,&qkvp,&gup,&actp,&aop,&redp,&pmp,&pdp,&paccp,&lgp,&D,&I,&nl,&V,&ms,&p2,&wph,&sc,&rb,&rm,&ls};
  cudaLaunchCooperativeKernel((void*)decode_mega, grid, nthreads, args, 0, at::cuda::getCurrentCUDAStream());
  return logits;
}

std::vector<int64_t> decode_occupancy() {
  cudaFuncAttributes attr; cudaFuncGetAttributes(&attr, (const void*)decode_mega);
  int bps = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, decode_mega, 256, 0);
  int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  // {registers/thread, blocks/SM, #SMs, warps/SM, total warps}
  return {(int64_t)attr.numRegs, (int64_t)bps, (int64_t)nsm, (int64_t)bps * 8, (int64_t)bps * nsm * 8};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_mega_launch", &decode_mega_launch, "full-decode megakernel (persistent, B=1)");
  m.def("decode_occupancy", &decode_occupancy, "regs/occupancy diagnostics");
}
