// M=K int4 GEMV — the core primitive for spec-decode verify. Each weight word is read ONCE and applied to
// M activation vectors (the K tokens being verified), so the weight-stream (the memory bottleneck) is shared
// across all M. This measures verify_cost(M) = time(M)/time(1): how much a K-token verify pass costs vs a
// single token. Memory-bound => should stay near 1 for small M (extra compute hides in the latency slack).
// Contiguous layout (== int4_gemv_v): Wq [OUT, IN/8] uint32, scales [OUT, IN/128] bf16, X [M, IN] bf16.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

template<int M>
__global__ void gemv_mk(const uint32_t* __restrict__ Wq, const __nv_bfloat16* __restrict__ scales,
                        const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y, int IN, int OUT) {
  const int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + wib;
  if (row >= OUT) return;
  const int ncols = IN >> 3;
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * ncols;
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (IN >> 7);
  float acc[M];
  #pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.f;
  for (int col = lane; col < ncols; col += 32) {
    const uint32_t w = Wrow[col]; const float sc = __bfloat162float(srow[col >> 4]);   // weight read ONCE
    #pragma unroll
    for (int m = 0; m < M; m++) {
      const int4 xr = __ldg(reinterpret_cast<const int4*>(&X[(size_t)m * IN + (col << 3)]));
      const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xr);
      #pragma unroll
      for (int nib = 0; nib < 8; nib++)
        acc[m] += (float((w >> (4 * nib)) & 0xF) - 8.0f) * sc * __bfloat162float(xb[nib]);
    }
  }
  #pragma unroll
  for (int m = 0; m < M; m++) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_down_sync(0xffffffffu, acc[m], o);
    if (lane == 0) Y[(size_t)m * OUT + row] = __float2bfloat16(acc[m]);
  }
}

torch::Tensor gemv_mk_launch(torch::Tensor Wq, torch::Tensor scales, torch::Tensor X, int64_t M) {
  const int OUT = Wq.size(0), ncols = Wq.size(1), IN = ncols * 8;
  auto Y = torch::empty({M, OUT}, X.options());
  const uint32_t* wq = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>());
  const int nthreads = 256, wpb = nthreads >> 5, grid = (OUT + wpb - 1) / wpb;
  auto stream = at::cuda::getCurrentCUDAStream();
  #define LAUNCH(K) gemv_mk<K><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT)
  switch (M) { case 1: LAUNCH(1); break; case 2: LAUNCH(2); break; case 4: LAUNCH(4); break;
               case 8: LAUNCH(8); break; default: TORCH_CHECK(false, "M in {1,2,4,8}"); }
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemv_mk", &gemv_mk_launch, "M=K int4 GEMV (weight read once, M vectors)"); }
