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

__device__ __forceinline__ float wgemv_row(const uint32_t* __restrict__ Wrow,
    const __nv_bfloat16* __restrict__ srow, const __nv_bfloat16* __restrict__ x, int IN, int lane) {
  const int ncols = IN >> 3; float acc = 0.f;
  for (int col = lane; col < ncols; col += 32) {
    const uint32_t w = Wrow[col]; const float sc = __bfloat162float(srow[col >> 4]); const int k0 = col << 3;
    #pragma unroll
    for (int m = 0; m < 8; m++) acc += (float((w >> (4 * m)) & 0xF) - 8.0f) * sc * __bfloat162float(x[k0 + m]);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;
}

// rmsnorm(h) -> xn  (grid-wide sum-of-squares via atomicAdd on red[0])
__device__ void rmsnorm(cg::grid_group& grid, const __nv_bfloat16* h, const __nv_bfloat16* nw,
                        __nv_bfloat16* xn, float* red, int DIM, int tid, int nthreads) {
  if (tid == 0) red[0] = 0.f;
  grid.sync();
  float p = 0.f;
  for (int i = tid; i < DIM; i += nthreads) { float v = __bfloat162float(h[i]); p += v * v; }
  atomicAdd(&red[0], p);
  grid.sync();
  const float r = rsqrtf(red[0] / DIM + 1e-5f);
  for (int i = tid; i < DIM; i += nthreads) xn[i] = __float2bfloat16(__bfloat162float(h[i]) * r * __bfloat162float(nw[i]));
  grid.sync();
}

__device__ void attn_block(cg::grid_group& grid,
    const uint32_t* Wqkv, const __nv_bfloat16* s_qkv, const uint32_t* Wo, const __nv_bfloat16* s_o,
    const __nv_bfloat16* nw, __nv_bfloat16* h, __nv_bfloat16* kc, __nv_bfloat16* vc,
    __nv_bfloat16* xn, __nv_bfloat16* qkv, __nv_bfloat16* ao, float* red,
    int DIM, int pos, float scale, float rope_base, float resid,
    int tid, int nthreads, int warp, int lane, int nwarps) {
  const int QKV = DIM + 2 * NKV * HD;
  rmsnorm(grid, h, nw, xn, red, DIM, tid, nthreads);
  for (int row = warp; row < QKV; row += nwarps) {
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
  for (int qh = warp; qh < NH; qh += nwarps) {
    const int kv = qh / (NH / NKV); const __nv_bfloat16* q = qkv + qh * HD;
    float mmax = -1e30f, denom = 0.f, outv[2] = {0.f, 0.f};
    for (int t = 0; t < T; t++) {
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
    #pragma unroll
    for (int j = 0; j < 2; j++) ao[qh * HD + lane + j * 32] = __float2bfloat16(outv[j] / denom);
  }
  grid.sync();
  for (int row = warp; row < DIM; row += nwarps) {
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
  for (int row = warp; row < INTER2; row += nwarps) {
    const float r = wgemv_row(Wgu + (size_t)row * (DIM >> 3), s_gu + (size_t)row * (DIM >> 7), xn, DIM, lane);
    if (lane == 0) gu[row] = __float2bfloat16(r);
  }
  grid.sync();
  for (int i = tid; i < INTER; i += nthreads) {
    const float g = __bfloat162float(gu[i]);
    act[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * __bfloat162float(gu[i + INTER]));
  }
  grid.sync();
  for (int row = warp; row < DIM; row += nwarps) {
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
    float* red, float* logits,
    int DIM, int INTER, int NL, int VOCAB, int MAXSEQ, int pos,
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
    attn_block(grid, Wqkv + l * qkv_st, s_qkv + l * sqkv_st, Wo + l * wo_st, s_o + l * so_st,
               n1 + (size_t)l * DIM, h, kc + l * kc_st, vc + l * kc_st, xn, qkv, ao, red,
               DIM, pos, scale, rope_base, resid, tid, nthreads, warp, lane, nwarps);
    mlp_block(grid, n2 + (size_t)l * DIM, Wgu + l * wgu_st, s_gu + l * sgu_st, Wd + l * wd_st, s_d + l * sd_st,
              h, xn, gu, act, red, DIM, INTER, resid, tid, nthreads, warp, lane, nwarps);
  }
  // final norm + LM head
  rmsnorm(grid, h, nf, xn, red, DIM, tid, nthreads);
  for (int row = warp; row < VOCAB; row += nwarps) {
    const float r = wgemv_row(Wlm + (size_t)row * (DIM >> 3), s_lm + (size_t)row * (DIM >> 7), xn, DIM, lane);
    if (lane == 0) logits[row] = r / logits_scaling;
  }
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
  auto P = [](torch::Tensor t){ return reinterpret_cast<const uint32_t*>(t.data_ptr<int32_t>()); };
  auto B = [](torch::Tensor t){ return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  auto Bm = [](torch::Tensor t){ return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  const uint32_t *wqkv=P(Wqkv),*wo=P(Wo),*wgu=P(Wgu),*wd=P(Wd),*wlm=P(Wlm);
  const __nv_bfloat16 *sqkv=B(s_qkv),*so=B(s_o),*n1p=B(n1),*sgu=B(s_gu),*sd=B(s_d),*n2p=B(n2),*nfp=B(nf),*slm=B(s_lm);
  __nv_bfloat16 *hp=Bm(h),*kcp=Bm(kc),*vcp=Bm(vc),*xnp=Bm(xn),*qkvp=Bm(qkv),*gup=Bm(gu),*actp=Bm(act),*aop=Bm(ao);
  float *redp=red.data_ptr<float>(),*lgp=logits.data_ptr<float>();
  int D=DIM,I=INTER,nl=NL,V=VOCAB,ms=MAXSEQ,p2=pos; float sc=scale,rb=rope_base,rm=resid,ls=logits_scaling;
  int nthreads=256,dev=0; cudaGetDevice(&dev);
  int bps=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, decode_mega, nthreads, 0);
  int nsm=0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  int grid=bps*nsm;
  void* args[]={&wqkv,&sqkv,&wo,&so,&n1p,&wgu,&sgu,&wd,&sd,&n2p,&nfp,&wlm,&slm,&hp,&kcp,&vcp,
                &xnp,&qkvp,&gup,&actp,&aop,&redp,&lgp,&D,&I,&nl,&V,&ms,&p2,&sc,&rb,&rm,&ls};
  cudaLaunchCooperativeKernel((void*)decode_mega, grid, nthreads, args, 0, at::cuda::getCurrentCUDAStream());
  return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_mega_launch", &decode_mega_launch, "full-decode megakernel (persistent, B=1)");
}
