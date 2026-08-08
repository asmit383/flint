// flint M=K VERIFY megakernel — score M candidate tokens in ONE persistent cooperative int4 launch.
// Same body as decode_mega but with an M dimension: every weight row is read ONCE and applied to all M
// activation vectors (wgemv_row_mk), so the weight stream (the cost) is shared across the K verified tokens.
// This is spec-decode's verify running ON the megakernel — the thing that lets accepted/verify_cost > 1.
// State is [M, *] row-major. Attention is looped per query (cheap); the GEMVs are the M=K flat part.
// Granite-3B: DIM=2560 INTER=8192 NH=40 NKV=8 HD=64 VOCAB=100352.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define HD 64
#define NH 40
#define NKV 8
#define GSYNC() grid.sync()

// read the int4 weight row ONCE, apply to M activation vectors x[M, IN]. out[M] = W_row . x[m].
template<int M>
__device__ __forceinline__ void wgemv_row_mk(const uint32_t* __restrict__ Wrow,
    const __nv_bfloat16* __restrict__ srow, const __nv_bfloat16* __restrict__ x, int IN, int lane, float* out) {
  const int ncols = IN >> 3;
  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.f;
  for (int col = lane; col < ncols; col += 32) {
    const uint32_t w = Wrow[col]; const float sc = __bfloat162float(srow[col >> 4]);
    float wdq[8];                                        // dequant the 8 nibbles ONCE (not M times)
    #pragma unroll
    for (int nib = 0; nib < 8; nib++) wdq[nib] = (float((w >> (4 * nib)) & 0xF) - 8.0f) * sc;
    #pragma unroll
    for (int m = 0; m < M; m++) {
      const int4 xr = __ldg(reinterpret_cast<const int4*>(&x[(size_t)m * IN + (col << 3)]));
      const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xr);
      #pragma unroll
      for (int nib = 0; nib < 8; nib++) acc[m] += wdq[nib] * __bfloat162float(xb[nib]);
    }
  }
  #pragma unroll
  for (int m = 0; m < M; m++) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_down_sync(0xffffffffu, acc[m], o);
    out[m] = acc[m];
  }
}

// rmsnorm for M vectors: h[M, DIM] -> xn[M, DIM]. Redundant per-block reduction (h tiny + L2-hot).
template<int M>
__device__ void rmsnorm_mk(cg::grid_group& grid, const __nv_bfloat16* h, const __nv_bfloat16* nw,
                           __nv_bfloat16* xn, int DIM, int tid, int nthreads) {
  __shared__ float bs[32]; __shared__ float rms_s;
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5, nw_b = blockDim.x >> 5;
  for (int m = 0; m < M; m++) {
    const __nv_bfloat16* hm = h + (size_t)m * DIM;
    float ss = 0.f;
    for (int i = threadIdx.x; i < DIM; i += blockDim.x) { float v = __bfloat162float(hm[i]); ss += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if (lane == 0) bs[wb] = ss;
    __syncthreads();
    if (wb == 0) {
      float s = (lane < nw_b) ? bs[lane] : 0.f;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
      if (lane == 0) rms_s = rsqrtf(s / DIM + 1e-5f);
    }
    __syncthreads();
    const float r = rms_s;
    for (int i = tid; i < DIM; i += nthreads)
      xn[(size_t)m * DIM + i] = __float2bfloat16(__bfloat162float(hm[i]) * r * __bfloat162float(nw[i]));
    __syncthreads();
  }
  GSYNC();
}

template<int M>
__global__ void verify_mega(
    const uint32_t* Wqkv, const __nv_bfloat16* s_qkv, const uint32_t* Wo, const __nv_bfloat16* s_o,
    const __nv_bfloat16* n1, const uint32_t* Wgu, const __nv_bfloat16* s_gu,
    const uint32_t* Wd, const __nv_bfloat16* s_d, const __nv_bfloat16* n2,
    const __nv_bfloat16* nf, const uint32_t* Wlm, const __nv_bfloat16* s_lm,
    __nv_bfloat16* h, __nv_bfloat16* kc, __nv_bfloat16* vc,
    __nv_bfloat16* xn, __nv_bfloat16* qkv, __nv_bfloat16* gu, __nv_bfloat16* act, __nv_bfloat16* ao,
    float* pm, float* pd, float* pacc, float* logits,
    int DIM, int INTER, int NL, int VOCAB, int MAXSEQ, int pos, int WPH,
    float scale, float rope_base, float resid, float logits_scaling) {
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x, nthreads = gridDim.x * blockDim.x;
  const int warp = tid >> 5, lane = tid & 31, nwarps = nthreads >> 5;
  const int QKV = DIM + 2 * NKV * HD, INTER2 = 2 * INTER;
  const size_t qkv_st = (size_t)QKV * (DIM >> 3), sqkv_st = (size_t)QKV * (DIM >> 7);
  const size_t wo_st = (size_t)DIM * (DIM >> 3), so_st = (size_t)DIM * (DIM >> 7);
  const size_t wgu_st = (size_t)INTER2 * (DIM >> 3), sgu_st = (size_t)INTER2 * (DIM >> 7);
  const size_t wd_st = (size_t)DIM * (INTER >> 3), sd_st = (size_t)DIM * (INTER >> 7);
  const size_t kc_st = (size_t)MAXSEQ * NKV * HD;
  float acc[M];

  for (int l = 0; l < NL; l++) {
    // ---- attention ----
    rmsnorm_mk<M>(grid, h, n1 + (size_t)l * DIM, xn, DIM, tid, nthreads);
    for (int row = warp; row < QKV; row += nwarps) {
      wgemv_row_mk<M>(Wqkv + l * qkv_st + (size_t)row * (DIM >> 3), s_qkv + l * sqkv_st + (size_t)row * (DIM >> 7),
                      xn, DIM, lane, acc);
      if (lane == 0) { for (int m = 0; m < M; m++) qkv[(size_t)m * QKV + row] = __float2bfloat16(acc[m]); }
    }
    GSYNC();
    // RoPE per token m (position pos+m), per head
    for (int hh = warp; hh < NH + NKV; hh += nwarps) {
      const int off = (hh < NH) ? hh * HD : DIM + (hh - NH) * HD;
      for (int m = 0; m < M; m++) {
        __nv_bfloat16* base = qkv + (size_t)m * QKV + off; const int posm = pos + m;
        for (int i = lane; i < HD / 2; i += 32) {
          const float freq = powf(rope_base, -2.0f * i / HD), ang = posm * freq, c = cosf(ang), sn = sinf(ang);
          const float x0 = __bfloat162float(base[i]), x1 = __bfloat162float(base[i + HD / 2]);
          base[i] = __float2bfloat16(x0 * c - x1 * sn); base[i + HD / 2] = __float2bfloat16(x1 * c + x0 * sn);
        }
      }
    }
    GSYNC();
    // write all M new K/V to cache at pos+m
    for (int m = 0; m < M; m++) {
      for (int i = tid; i < NKV * HD; i += nthreads) {
        kc[l * kc_st + (size_t)(pos + m) * NKV * HD + i] = qkv[(size_t)m * QKV + DIM + i];
        vc[l * kc_st + (size_t)(pos + m) * NKV * HD + i] = qkv[(size_t)m * QKV + DIM + NKV * HD + i];
      }
    }
    GSYNC();
    // attention: all M queries in 2 grid-syncs (each query has its own partials). query pos+m sees 0..pos+m.
    {
      const int head = warp / WPH, sub = warp % WPH;
      if (head < NH) {
        const int kv = head / (NH / NKV);
        for (int m = 0; m < M; m++) {                    // loop queries INSIDE the warp (no grid sync between)
          const int T = pos + m + 1; const __nv_bfloat16* q = qkv + (size_t)m * QKV + head * HD;
          float mmax = -1e30f, denom = 0.f, outv[2] = {0.f, 0.f};
          for (int t = sub; t < T; t += WPH) {
            const __nv_bfloat16* kt = kc + l * kc_st + ((size_t)t * NKV + kv) * HD; float s = 0.f;
            for (int d = lane; d < HD; d += 32) s += __bfloat162float(q[d]) * __bfloat162float(kt[d]);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
            s = __shfl_sync(0xffffffffu, s, 0) * scale;
            const float m2 = fmaxf(mmax, s), corr = __expf(mmax - m2), w = __expf(s - m2);
            denom = denom * corr + w;
            const __nv_bfloat16* vt = vc + l * kc_st + ((size_t)t * NKV + kv) * HD;
            #pragma unroll
            for (int j = 0; j < 2; j++) outv[j] = outv[j] * corr + w * __bfloat162float(vt[lane + j * 32]);
            mmax = m2;
          }
          const int pidx = (m * NH + head) * WPH + sub;
          if (lane == 0) { pm[pidx] = mmax; pd[pidx] = denom; }
          pacc[(size_t)pidx * HD + lane] = outv[0]; pacc[(size_t)pidx * HD + lane + 32] = outv[1];
        }
      }
      GSYNC();
      for (int mh = warp; mh < M * NH; mh += nwarps) {   // merge all M*NH heads in one phase
        const int m = mh / NH, qh = mh % NH; float gm = -1e30f;
        for (int s = 0; s < WPH; s++) gm = fmaxf(gm, pm[(m * NH + qh) * WPH + s]);
        float gd = 0.f, acc0 = 0.f, acc1 = 0.f;
        for (int s = 0; s < WPH; s++) {
          const int pidx = (m * NH + qh) * WPH + s; const float wt = __expf(pm[pidx] - gm);
          gd += pd[pidx] * wt;
          acc0 += pacc[(size_t)pidx * HD + lane] * wt; acc1 += pacc[(size_t)pidx * HD + lane + 32] * wt;
        }
        ao[(size_t)m * DIM + qh * HD + lane] = __float2bfloat16(acc0 / gd);
        ao[(size_t)m * DIM + qh * HD + lane + 32] = __float2bfloat16(acc1 / gd);
      }
      GSYNC();
    }
    // o proj (M=K) + residual
    for (int row = warp; row < DIM; row += nwarps) {
      wgemv_row_mk<M>(Wo + l * wo_st + (size_t)row * (DIM >> 3), s_o + l * so_st + (size_t)row * (DIM >> 7),
                      ao, DIM, lane, acc);
      if (lane == 0) { for (int m = 0; m < M; m++)
        h[(size_t)m * DIM + row] = __float2bfloat16(__bfloat162float(h[(size_t)m * DIM + row]) + acc[m] * resid); }
    }
    GSYNC();
    // ---- mlp ----
    rmsnorm_mk<M>(grid, h, n2 + (size_t)l * DIM, xn, DIM, tid, nthreads);
    for (int row = warp; row < INTER2; row += nwarps) {
      wgemv_row_mk<M>(Wgu + l * wgu_st + (size_t)row * (DIM >> 3), s_gu + l * sgu_st + (size_t)row * (DIM >> 7),
                      xn, DIM, lane, acc);
      if (lane == 0) { for (int m = 0; m < M; m++) gu[(size_t)m * INTER2 + row] = __float2bfloat16(acc[m]); }
    }
    GSYNC();
    for (int m = 0; m < M; m++) {
      for (int i = tid; i < INTER; i += nthreads) {
        const float g = __bfloat162float(gu[(size_t)m * INTER2 + i]);
        act[(size_t)m * INTER + i] = __float2bfloat16((g / (1.0f + __expf(-g))) * __bfloat162float(gu[(size_t)m * INTER2 + i + INTER]));
      }
    }
    GSYNC();
    for (int row = warp; row < DIM; row += nwarps) {
      wgemv_row_mk<M>(Wd + l * wd_st + (size_t)row * (INTER >> 3), s_d + l * sd_st + (size_t)row * (INTER >> 7),
                      act, INTER, lane, acc);
      if (lane == 0) { for (int m = 0; m < M; m++)
        h[(size_t)m * DIM + row] = __float2bfloat16(__bfloat162float(h[(size_t)m * DIM + row]) + acc[m] * resid); }
    }
    GSYNC();
  }
  // final norm + LM head (M=K)
  rmsnorm_mk<M>(grid, h, nf, xn, DIM, tid, nthreads);
  for (int row = warp; row < VOCAB; row += nwarps) {
    wgemv_row_mk<M>(Wlm + (size_t)row * (DIM >> 3), s_lm + (size_t)row * (DIM >> 7), xn, DIM, lane, acc);
    if (lane == 0) { for (int m = 0; m < M; m++) logits[(size_t)m * VOCAB + row] = acc[m] / logits_scaling; }
  }
}

torch::Tensor verify_mega_launch(
    torch::Tensor Wqkv, torch::Tensor s_qkv, torch::Tensor Wo, torch::Tensor s_o, torch::Tensor n1,
    torch::Tensor Wgu, torch::Tensor s_gu, torch::Tensor Wd, torch::Tensor s_d, torch::Tensor n2,
    torch::Tensor nf, torch::Tensor Wlm, torch::Tensor s_lm,
    torch::Tensor h, torch::Tensor kc, torch::Tensor vc, int64_t pos, int64_t M,
    double scale, double rope_base, double resid, double logits_scaling) {
  const int DIM = h.size(1), NL = Wqkv.size(0), VOCAB = Wlm.size(0);
  const int INTER = Wd.size(2) * 8, MAXSEQ = kc.size(1), QKV = DIM + 2 * NKV * HD;
  auto opt = h.options();
  auto xn = torch::empty({M, DIM}, opt), qkv = torch::empty({M, QKV}, opt);
  auto gu = torch::empty({M, 2 * INTER}, opt), act = torch::empty({M, INTER}, opt), ao = torch::empty({M, DIM}, opt);
  auto logits = torch::empty({M, VOCAB}, opt.dtype(torch::kFloat32));
  int nthreads = 256, dev = 0; cudaGetDevice(&dev);
  void* fn = (M == 1) ? (void*)verify_mega<1> : (M == 2) ? (void*)verify_mega<2>
           : (M == 4) ? (void*)verify_mega<4> : (M == 8) ? (void*)verify_mega<8>
           : (M == 6) ? (void*)verify_mega<6> : (void*)verify_mega<3>;
  int bps = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, fn, nthreads, 0);
  int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  int grid = bps * nsm; int nwarps = grid * (nthreads >> 5); int WPH = nwarps / NH; if (WPH < 1) WPH = 1;
  auto pm = torch::empty({M * NH * WPH}, opt.dtype(torch::kFloat32));      // per-query partials (M queries)
  auto pd = torch::empty({M * NH * WPH}, opt.dtype(torch::kFloat32));
  auto pacc = torch::empty({(int64_t)M * NH * WPH * HD}, opt.dtype(torch::kFloat32));
  auto P = [](torch::Tensor t){ return reinterpret_cast<const uint32_t*>(t.data_ptr<int32_t>()); };
  auto Bc = [](torch::Tensor t){ return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  auto Bm = [](torch::Tensor t){ return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()); };
  const uint32_t *wqkv=P(Wqkv),*wo=P(Wo),*wgu=P(Wgu),*wd=P(Wd),*wlm=P(Wlm);
  const __nv_bfloat16 *sqkv=Bc(s_qkv),*so=Bc(s_o),*n1p=Bc(n1),*sgu=Bc(s_gu),*sd=Bc(s_d),*n2p=Bc(n2),*nfp=Bc(nf),*slm=Bc(s_lm);
  __nv_bfloat16 *hp=Bm(h),*kcp=Bm(kc),*vcp=Bm(vc),*xnp=Bm(xn),*qkvp=Bm(qkv),*gup=Bm(gu),*actp=Bm(act),*aop=Bm(ao);
  float *lgp=logits.data_ptr<float>(),*pmp=pm.data_ptr<float>(),*pdp=pd.data_ptr<float>(),*paccp=pacc.data_ptr<float>();
  int D=DIM,I=INTER,nl=NL,V=VOCAB,ms=MAXSEQ,p2=pos,wph=WPH; float sc=scale,rb=rope_base,rm=resid,ls=logits_scaling;
  void* args[]={&wqkv,&sqkv,&wo,&so,&n1p,&wgu,&sgu,&wd,&sd,&n2p,&nfp,&wlm,&slm,&hp,&kcp,&vcp,
                &xnp,&qkvp,&gup,&actp,&aop,&pmp,&pdp,&paccp,&lgp,&D,&I,&nl,&V,&ms,&p2,&wph,&sc,&rb,&rm,&ls};
  cudaLaunchCooperativeKernel(fn, grid, nthreads, args, 0, at::cuda::getCurrentCUDAStream());
  return logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("verify_mega_launch", &verify_mega_launch, "M=K int4 verify megakernel (persistent, B=1 spec-decode)");
}
