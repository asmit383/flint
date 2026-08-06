// Producer-consumer MLP (proof-of-concept for the barrier-free megakernel).
// y = Wd @ (silu(gate) * up),  gate/up = Wgu @ xn.  NO global barrier: producer warps compute gate/up
// tile-by-tile and raise a per-tile flag; consumer warps compute the down GEMV, spinning on each tile's
// flag and accumulating as tiles arrive. Both weight streams (Wgu + Wd) fly concurrently -> more HBM
// requests in flight -> higher MBU than the sequential [gate/up | barrier | down].
// Regular launch sized to co-resident blocks (producers+consumers must all be resident, else deadlock).
// Contiguous int4 layout (== int4_gemv_v). TILE neurons per flag.
//
// VERDICT (measured, H100 PCIe, B=1): DEAD END. Best pc = 79us @ TILE=1024/prod=0.5 (MBU 20.5%) vs
// sequential [gate/up | barrier | down] = 59us (MBU 27.4%) -> pc is 1.34x SLOWER, MBU is LOWER not higher.
// Falsifies "concurrent weight streams beat the barrier". Three structural reasons: (1) splitting warps
// into producers/consumers starves each phase of parallelism; (2) down needs ALL k-tiles (reduction dep)
// so consumers can't finish until producers deliver the last tile -> ~no wall-clock overlap; (3) fence +
// flag-poll sync costs more than the grid.sync() barrier it replaces. The sequential-barrier megakernel
// (284 tok/s) is the single-pass B=1 ceiling; the real lever is EAGLE-3 speculative decode, not this.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#define TILE 1024

__device__ __forceinline__ float dq_row(const uint32_t* __restrict__ Wrow, const __nv_bfloat16* __restrict__ srow,
    const __nv_bfloat16* __restrict__ x, int IN, int lane) {
  const int ncols = IN >> 3; float acc = 0.f;
  for (int col = lane; col < ncols; col += 32) {
    const uint32_t w = Wrow[col]; const float sc = __bfloat162float(srow[col >> 4]);
    const int4 xr = __ldg(reinterpret_cast<const int4*>(&x[col << 3]));
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xr);
    #pragma unroll
    for (int m = 0; m < 8; m++) acc += (float((w >> (4 * m)) & 0xF) - 8.0f) * sc * __bfloat162float(xb[m]);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;
}

__global__ void pc_mlp_kernel(
    const uint32_t* __restrict__ Wgu, const __nv_bfloat16* __restrict__ s_gu,
    const uint32_t* __restrict__ Wd,  const __nv_bfloat16* __restrict__ s_d,
    const __nv_bfloat16* __restrict__ xn, __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ y,
    int* __restrict__ flags, int* __restrict__ done, int DIM, int INTER, int NTILE, int PROD) {
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int total_warps = (gridDim.x * blockDim.x) >> 5;

  if (gwarp < PROD) {                                    // ---- producers: gate/up, all rows round-robin ----
    for (int r = gwarp; r < 2 * INTER; r += PROD) {      // r < INTER: gate; else up
      const float v = dq_row(Wgu + (size_t)r * (DIM >> 3), s_gu + (size_t)r * (DIM >> 7), xn, DIM, lane);
      if (lane == 0) {
        gu[r] = __float2bfloat16(v);
        const int n = (r < INTER) ? r : r - INTER;       // neuron -> its tile
        __threadfence();                                 // gu[r] visible before we count it
        if (atomicAdd(&done[n / TILE], 1) + 1 == 2 * TILE) flags[n / TILE] = 1;  // last of tile: announce
      }
    }
  } else {                                               // ---- consumers: down GEMV, tile by tile ----
    const int CONS = total_warps - PROD, cw = gwarp - PROD;
    for (int row = cw; row < DIM; row += CONS) {
      const uint32_t* Wr = Wd + (size_t)row * (INTER >> 3);
      const __nv_bfloat16* sr = s_d + (size_t)row * (INTER >> 7);
      float acc = 0.f;
      for (int t = 0; t < NTILE; t++) {
        if (lane == 0) while (!((volatile int*)flags)[t]) { }  // cheap volatile poll (no atomic RMW)
        __syncwarp();
        __threadfence();                                 // acquire: gu writes visible before we read them
        const int col0 = (t * TILE) >> 3;                // uint32 col where this tile starts (TILE/8 words)
        for (int c = col0 + lane; c < col0 + (TILE >> 3); c += 32) {
          const uint32_t w = Wr[c]; const float sc = __bfloat162float(sr[c >> 4]); const int k0 = c << 3;
          #pragma unroll
          for (int m = 0; m < 8; m++) {
            const float gg = __bfloat162float(gu[k0 + m]);
            const float a = (gg / (1.0f + __expf(-gg))) * __bfloat162float(gu[INTER + k0 + m]);
            acc += (float((w >> (4 * m)) & 0xF) - 8.0f) * sc * a;
          }
        }
      }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
      if (lane == 0) y[row] = __float2bfloat16(acc);
    }
  }
}

torch::Tensor pc_mlp(torch::Tensor Wgu, torch::Tensor s_gu, torch::Tensor Wd, torch::Tensor s_d,
                     torch::Tensor xn, double prod_frac) {
  const int DIM = Wd.size(0), INTER = Wd.size(1) * 8;
  const int NTILE = INTER / TILE;
  auto opt = xn.options();
  auto gu = torch::empty({2 * INTER}, opt), y = torch::empty({DIM}, opt);
  auto flags = torch::zeros({NTILE}, opt.dtype(torch::kInt32));
  auto done = torch::zeros({NTILE}, opt.dtype(torch::kInt32));
  const uint32_t* wgu = reinterpret_cast<const uint32_t*>(Wgu.data_ptr<int32_t>());
  const uint32_t* wd = reinterpret_cast<const uint32_t*>(Wd.data_ptr<int32_t>());
  const __nv_bfloat16* sgu = reinterpret_cast<const __nv_bfloat16*>(s_gu.data_ptr<at::BFloat16>());
  const __nv_bfloat16* sd = reinterpret_cast<const __nv_bfloat16*>(s_d.data_ptr<at::BFloat16>());
  const __nv_bfloat16* xnp = reinterpret_cast<const __nv_bfloat16*>(xn.data_ptr<at::BFloat16>());
  __nv_bfloat16* gup = reinterpret_cast<__nv_bfloat16*>(gu.data_ptr<at::BFloat16>());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  int* fp = flags.data_ptr<int>(); int* dp = done.data_ptr<int>();
  int nthreads = 256, dev = 0; cudaGetDevice(&dev);
  int bps = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, pc_mlp_kernel, nthreads, 0);
  int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  int grid = bps * nsm;                                  // all blocks co-resident (no deadlock)
  int total_warps = grid * (nthreads >> 5);
  int PROD = (int)(total_warps * prod_frac); if (PROD < 1) PROD = 1; if (PROD >= total_warps) PROD = total_warps - 1;
  pc_mlp_kernel<<<grid, nthreads, 0, at::cuda::getCurrentCUDAStream()>>>(wgu, sgu, wd, sd, xnp, gup, yp, fp, dp, DIM, INTER, NTILE, PROD);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("pc_mlp", &pc_mlp, "producer-consumer MLP (barrier-free)"); }
