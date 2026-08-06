// flint megakernel — attention block (Granite GQA + RoPE, B=1 decode) in ONE cooperative kernel.
// rmsnorm(h) -> Wqkv GEMV -> RoPE(q,k) -> append k,v to cache -> SDPA over cache -> Wo GEMV -> +residual.
// The other half of a fused layer (MLP block is megakernel_mlp.cu). Correctness-first; perf is end-to-end.
//
// Granite-3B: dim=2560, n_head=40, n_kv=8 (GQA 5:1), head_dim=64, attn_scale=0.015625, rope_base=1e7.
// Wqkv [3584, 2560/8] int4 (q=2560, k=512, v=512 rows), Wo [2560, 2560/8] int4.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define HD 64          // head_dim
#define NH 40          // query heads
#define NKV 8          // kv heads

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

__global__ void attn_mega_kernel(
    const uint32_t* __restrict__ Wqkv, const __nv_bfloat16* __restrict__ s_qkv,
    const uint32_t* __restrict__ Wo,   const __nv_bfloat16* __restrict__ s_o,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ h,                       // [DIM] residual in/out
    __nv_bfloat16* __restrict__ kcache, __nv_bfloat16* __restrict__ vcache,  // [max_seq, NKV, HD]
    __nv_bfloat16* __restrict__ xn, __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ ao,
    float* __restrict__ red,                             // [1] scratch for rmsnorm reduction
    int DIM, int pos, float attn_scale, float rope_base, float resid_mult) {
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, nthreads = gridDim.x * blockDim.x;
  const int warp = tid >> 5, lane = tid & 31, nwarps = nthreads >> 5;
  const int QKV = DIM + 2 * NKV * HD;                    // 3584

  // Phase 0: rmsnorm(h) -> xn.  sum of squares via grid atomicAdd.
  if (tid == 0) red[0] = 0.f;
  grid.sync();
  float p = 0.f;
  for (int i = tid; i < DIM; i += nthreads) { float v = __bfloat162float(h[i]); p += v * v; }
  atomicAdd(&red[0], p);
  grid.sync();
  const float rms = rsqrtf(red[0] / DIM + 1e-5f);
  for (int i = tid; i < DIM; i += nthreads) xn[i] = __float2bfloat16(__bfloat162float(h[i]) * rms * __bfloat162float(norm_w[i]));
  grid.sync();

  // Phase 1: qkv GEMV -> qkv[QKV]  (warp per output row)
  for (int row = warp; row < QKV; row += nwarps) {
    const float r = wgemv_row(Wqkv + (size_t)row * (DIM >> 3), s_qkv + (size_t)row * (DIM >> 7), xn, DIM, lane);
    if (lane == 0) qkv[row] = __float2bfloat16(r);
  }
  grid.sync();

  // Phase 2: RoPE on q (heads 0..NH-1) and k (NKV heads), then append k,v to cache at `pos`.
  // one warp per head; lane handles rotary pair (i, i+HD/2).
  const int total_heads = NH + NKV;                      // rope q-heads + k-heads
  for (int hh = warp; hh < total_heads; hh += nwarps) {
    __nv_bfloat16* base = (hh < NH) ? (qkv + hh * HD) : (qkv + DIM + (hh - NH) * HD);
    for (int i = lane; i < HD / 2; i += 32) {
      const float freq = powf(rope_base, -2.0f * i / HD);
      const float ang = pos * freq, c = cosf(ang), sn = sinf(ang);
      const float x0 = __bfloat162float(base[i]), x1 = __bfloat162float(base[i + HD / 2]);
      base[i]          = __float2bfloat16(x0 * c - x1 * sn);
      base[i + HD / 2] = __float2bfloat16(x1 * c + x0 * sn);
    }
  }
  grid.sync();
  // append k,v to cache (one thread per element)
  for (int i = tid; i < NKV * HD; i += nthreads) {
    kcache[(size_t)pos * NKV * HD + i] = qkv[DIM + i];
    vcache[(size_t)pos * NKV * HD + i] = qkv[DIM + NKV * HD + i];
  }
  grid.sync();

  // Phase 3: attention. one warp per query head; loop cache t=0..pos.
  const int T = pos + 1;
  for (int qh = warp; qh < NH; qh += nwarps) {
    const int kv = qh / (NH / NKV);                      // GQA 5:1
    const __nv_bfloat16* q = qkv + qh * HD;
    // online softmax over t, accumulate weighted V
    float mmax = -1e30f, denom = 0.f, acc[HD / 32 ? HD : HD];  // lane owns HD/32=2 dims
    float outv[2] = {0.f, 0.f};                          // HD=64, 32 lanes -> 2 dims/lane
    for (int t = 0; t < T; t++) {
      const __nv_bfloat16* kt = kcache + ((size_t)t * NKV + kv) * HD;
      float s = 0.f;
      for (int d = lane; d < HD; d += 32) s += __bfloat162float(q[d]) * __bfloat162float(kt[d]);
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
      s = __shfl_sync(0xffffffffu, s, 0) * attn_scale;   // broadcast score
      const float m2 = fmaxf(mmax, s), corr = __expf(mmax - m2), w = __expf(s - m2);
      denom = denom * corr + w;
      const __nv_bfloat16* vt = vcache + ((size_t)t * NKV + kv) * HD;
      #pragma unroll
      for (int j = 0; j < 2; j++) { int d = lane + j * 32; outv[j] = outv[j] * corr + w * __bfloat162float(vt[d]); }
      mmax = m2;
    }
    #pragma unroll
    for (int j = 0; j < 2; j++) { int d = lane + j * 32; ao[qh * HD + d] = __float2bfloat16(outv[j] / denom); }
  }
  grid.sync();

  // Phase 4: Wo GEMV(ao) + residual
  for (int row = warp; row < DIM; row += nwarps) {
    const float r = wgemv_row(Wo + (size_t)row * (DIM >> 3), s_o + (size_t)row * (DIM >> 7), ao, DIM, lane);
    if (lane == 0) h[row] = __float2bfloat16(__bfloat162float(h[row]) + r * resid_mult);
  }
}

torch::Tensor attn_mega(torch::Tensor Wqkv, torch::Tensor s_qkv, torch::Tensor Wo, torch::Tensor s_o,
                        torch::Tensor norm_w, torch::Tensor h, torch::Tensor kcache, torch::Tensor vcache,
                        int64_t pos, double attn_scale, double rope_base, double resid_mult) {
  const int DIM = h.size(0);
  auto opt = h.options();
  auto xn = torch::empty({DIM}, opt), qkv = torch::empty({DIM + 2 * NKV * HD}, opt), ao = torch::empty({DIM}, opt);
  auto red = torch::zeros({1}, opt.dtype(torch::kFloat32));
  auto P = [](torch::Tensor t){ return reinterpret_cast<const uint32_t*>(t.data_ptr<int32_t>()); };
  auto B = [](torch::Tensor t){ return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  auto Bm = [](torch::Tensor t){ return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  const uint32_t* wqkv = P(Wqkv); const uint32_t* wo = P(Wo);
  const __nv_bfloat16 *sqkv = B(s_qkv), *so = B(s_o), *nw = B(norm_w);
  __nv_bfloat16 *hp = Bm(h), *kc = Bm(kcache), *vc = Bm(vcache), *xnp = Bm(xn), *qkvp = Bm(qkv), *aop = Bm(ao);
  float* redp = red.data_ptr<float>();
  int as = 0, rb = 0, rm = 0; float asc = attn_scale, rbf = rope_base, rmf = resid_mult; int P2 = pos, D = DIM;
  int nthreads = 256, dev = 0; cudaGetDevice(&dev);
  int bps = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, attn_mega_kernel, nthreads, 0);
  int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  int grid = bps * nsm;
  void* args[] = {&wqkv,&sqkv,&wo,&so,&nw,&hp,&kc,&vc,&xnp,&qkvp,&aop,&redp,&D,&P2,&asc,&rbf,&rmf};
  cudaLaunchCooperativeKernel((void*)attn_mega_kernel, grid, nthreads, args, 0, at::cuda::getCurrentCUDAStream());
  return h;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_mega", &attn_mega, "fused attention megakernel (GQA+RoPE, B=1)");
}
