// int4 GEMV with cp.async deep software pipeline — the lever to push B=1 MBU toward the bf16 ceiling.
// The wall: int4 reads few bytes, so at B=1 the kernel is latency-bound — not enough concurrent memory
// requests to saturate HBM. cp.async fires async global->shared loads that DON'T block the warp, so we
// keep NSTAGE weight chunks in flight per warp at all times. Many warps × NSTAGE outstanding = saturate.
// Contiguous layout (== int4_gemv_v): Wq [OUT, IN/8] uint32, scales [OUT, IN/128] bf16.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

template<int NSTAGE>
__global__ void gemv_cpa(const uint32_t* __restrict__ Wq, const __nv_bfloat16* __restrict__ scales,
                         const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, int IN, int OUT) {
  extern __shared__ uint32_t sm[];                       // [warps_per_block * NSTAGE * 32]
  const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + wib;
  if (row >= OUT) return;
  const int ncols = IN >> 3, nchunks = (ncols + 31) >> 5;
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * ncols;
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (IN >> 7);
  uint32_t* buf = sm + wib * NSTAGE * 32;                // this warp's ring buffer

  #pragma unroll
  for (int s = 0; s < NSTAGE - 1; s++) {                 // prime the pipeline
    if (s < nchunks) __pipeline_memcpy_async(&buf[s * 32 + lane], &Wrow[s * 32 + lane], sizeof(uint32_t));
    __pipeline_commit();
  }
  float acc = 0.f;
  for (int c = 0; c < nchunks; c++) {
    const int pf = c + NSTAGE - 1;
    if (pf < nchunks) __pipeline_memcpy_async(&buf[(pf % NSTAGE) * 32 + lane], &Wrow[pf * 32 + lane], sizeof(uint32_t));
    __pipeline_commit();
    __pipeline_wait_prior(NSTAGE - 1);                   // chunk c now ready in shared
    __syncwarp();
    const int col = c * 32 + lane;
    if (col < ncols) {
      const uint32_t w = buf[(c % NSTAGE) * 32 + lane];
      const float sc = __bfloat162float(srow[col >> 4]);
      const int4 xr = __ldg(reinterpret_cast<const int4*>(&x[col << 3]));
      const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xr);
      #pragma unroll
      for (int m = 0; m < 8; m++) acc += (float((w >> (4 * m)) & 0xF) - 8.0f) * sc * __bfloat162float(xb[m]);
    }
    __syncwarp();
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) y[row] = __float2bfloat16(acc);
}

torch::Tensor gemv_cpasync(torch::Tensor Wq, torch::Tensor scales, torch::Tensor x, int64_t nstage) {
  const int OUT = Wq.size(0), ncols = Wq.size(1), IN = ncols * 8;
  auto y = torch::empty({OUT}, x.options());
  const uint32_t* wq = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  const int nthreads = 256, wpb = nthreads >> 5;
  const int grid = (OUT + wpb - 1) / wpb;
  auto stream = at::cuda::getCurrentCUDAStream();
  #define LAUNCH(NS) gemv_cpa<NS><<<grid, nthreads, (size_t)wpb * NS * 32 * sizeof(uint32_t), stream>>>(wq, sc, xp, yp, IN, OUT)
  switch (nstage) { case 2: LAUNCH(2); break; case 3: LAUNCH(3); break; case 4: LAUNCH(4); break;
                    case 6: LAUNCH(6); break; case 8: LAUNCH(8); break; default: TORCH_CHECK(false, "nstage 2/3/4/6/8"); }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemv_cpasync", &gemv_cpasync, "cp.async pipelined int4 GEMV"); }
