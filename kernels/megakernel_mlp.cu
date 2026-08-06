// flint megakernel — STEP 1 prototype: fused MLP block (Granite SwiGLU) in ONE cooperative kernel.
//
// Normally the MLP is ~5 kernels: [gate GEMV][up GEMV][silu*mul][down GEMV] with the 16384-dim
// intermediate round-tripping HBM twice. Here it's ONE persistent cooperative kernel: gate/up produce
// the intermediate into an L2-resident scratch, a grid.sync(), then down consumes it — the intermediate
// NEVER hits DRAM, and there are zero inter-op kernel relaunches. This is the megakernel principle in
// miniature (build plan step 1); steps 2-4 fold in attention, the residual, and all 40 layers.
//
// v1 status: UNTESTED (written CPU-side; compile+correctness+perf pending GPU, after training frees it).
// Layout matches int4_gemv_v (contiguous): Wgu [2*INTER, DIM/8] uint32, s_gu [2*INTER, DIM/128] bf16;
// Wd [DIM, INTER/8] uint32, s_d [DIM, INTER/128] bf16. xn = normed input (rmsnorm fused later, step 3).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// One warp computes one output row: warp-reduce sum_k dequant(int4) * x[k], x read from on-chip scratch.
__device__ __forceinline__ float warp_gemv_row(
    const uint32_t* __restrict__ Wrow, const __nv_bfloat16* __restrict__ srow,
    const __nv_bfloat16* __restrict__ x, int IN, int lane) {
  const int ncols = IN >> 3;                          // uint32 words (8 int4 each)
  float acc = 0.f;
  for (int col = lane; col < ncols; col += 32) {
    const uint32_t w = Wrow[col];
    const float sc = __bfloat162float(srow[col >> 4]); // 16 words / 128-group
    const int k0 = col << 3;
    #pragma unroll
    for (int m = 0; m < 8; m++)
      acc += (float((w >> (4 * m)) & 0xF) - 8.0f) * sc * __bfloat162float(x[k0 + m]);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  return acc;                                          // valid on lane 0
}

__global__ void mlp_mega_kernel(
    const uint32_t* __restrict__ Wgu, const __nv_bfloat16* __restrict__ s_gu,
    const uint32_t* __restrict__ Wd,  const __nv_bfloat16* __restrict__ s_d,
    const __nv_bfloat16* __restrict__ xn,               // [DIM] normed input
    __nv_bfloat16* __restrict__ h,                      // [DIM] residual in/out
    __nv_bfloat16* __restrict__ gu,                     // [2*INTER] scratch (L2-resident)
    __nv_bfloat16* __restrict__ act,                    // [INTER] scratch
    float resid_mult, int DIM, int INTER) {
  cg::grid_group grid = cg::this_grid();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nthreads = gridDim.x * blockDim.x;
  const int warp = tid >> 5, lane = tid & 31, nwarps = nthreads >> 5;
  const int INTER2 = 2 * INTER;

  // Phase 1: gate/up GEMV -> gu[2*INTER]  (weights stream from HBM; xn from L2)
  for (int row = warp; row < INTER2; row += nwarps) {
    const float r = warp_gemv_row(Wgu + (size_t)row * (DIM >> 3), s_gu + (size_t)row * (DIM >> 7), xn, DIM, lane);
    if (lane == 0) gu[row] = __float2bfloat16(r);
  }
  grid.sync();                                          // intermediate stays on-chip (L2), no DRAM trip

  // Phase 2: act = silu(gate) * up   (elementwise, fused)
  for (int i = tid; i < INTER; i += nthreads) {
    const float g = __bfloat162float(gu[i]);
    act[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * __bfloat162float(gu[i + INTER]));
  }
  grid.sync();

  // Phase 3: down GEMV + residual   (out[row] = sum_k dequant(Wd) * act[k])
  for (int row = warp; row < DIM; row += nwarps) {
    const float r = warp_gemv_row(Wd + (size_t)row * (INTER >> 3), s_d + (size_t)row * (INTER >> 7), act, INTER, lane);
    if (lane == 0) h[row] = __float2bfloat16(__bfloat162float(h[row]) + r * resid_mult);
  }
}

torch::Tensor mlp_mega(torch::Tensor Wgu, torch::Tensor s_gu, torch::Tensor Wd, torch::Tensor s_d,
                       torch::Tensor xn, torch::Tensor h, double resid_mult) {
  const int DIM = xn.size(0);
  const int INTER = Wd.size(1) * 8;                     // Wd: [DIM, INTER/8]
  auto opt = h.options();
  auto gu = torch::empty({2 * INTER}, opt);
  auto act = torch::empty({INTER}, opt);

  const uint32_t* wgu = reinterpret_cast<const uint32_t*>(Wgu.data_ptr<int32_t>());
  const uint32_t* wd = reinterpret_cast<const uint32_t*>(Wd.data_ptr<int32_t>());
  const __nv_bfloat16* sgu = reinterpret_cast<const __nv_bfloat16*>(s_gu.data_ptr<at::BFloat16>());
  const __nv_bfloat16* sd = reinterpret_cast<const __nv_bfloat16*>(s_d.data_ptr<at::BFloat16>());
  const __nv_bfloat16* xnp = reinterpret_cast<const __nv_bfloat16*>(xn.data_ptr<at::BFloat16>());
  __nv_bfloat16* hp = reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>());
  __nv_bfloat16* gup = reinterpret_cast<__nv_bfloat16*>(gu.data_ptr<at::BFloat16>());
  __nv_bfloat16* actp = reinterpret_cast<__nv_bfloat16*>(act.data_ptr<at::BFloat16>());
  float rm = (float)resid_mult;

  // cooperative launch: grid sized to what co-resides on all SMs (occupancy-limited).
  int nthreads = 256, dev = 0; cudaGetDevice(&dev);
  int blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, mlp_mega_kernel, nthreads, 0);
  int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  int grid = blocks_per_sm * nsm;
  void* args[] = {&wgu, &sgu, &wd, &sd, &xnp, &hp, &gup, &actp, &rm, (void*)&DIM, (void*)&INTER};
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchCooperativeKernel((void*)mlp_mega_kernel, grid, nthreads, args, 0, stream);
  return h;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mlp_mega", &mlp_mega, "fused MLP megakernel (cooperative, B=1)");
}
