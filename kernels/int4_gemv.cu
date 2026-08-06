// flint — CUDA-core int4 weight-only GEMV for B=1 decode.
//
// Why not tensor cores: at B=1 the "matmul" is a matrix-vector product (M=1). tinygemm pads M=1->16
// for its m16n8k16 MMA and throws away 15/16 of the tensor-core compute -> it's compute/dequant-bound
// (measured: SM 46% > DRAM 32%). This kernel uses plain CUDA cores: no M-padding, so every FMA is real
// work, and we spend the freed compute on hiding the int4->bf16 dequant behind memory latency.
//
// Layout (we own it, chosen for coalescing):
//   Wq      [OUT, IN/8]  uint32  — 8 offset-int4 per word; element k in bits [4k,4k+4), value=(q-8)
//   scales  [OUT, IN/G]  bf16    — one groupwise scale per G columns of the input dim
//   x       [IN]         bf16    — staged once per block into shared (reused by every row in the block)
//   y       [OUT]        bf16
//
// One WARP computes one output row: the 32 lanes split the K (input) dim, each reading a contiguous
// uint32 (=8 weights) so the warp's 32 reads form one 128-byte coalesced transaction. NACC independent
// accumulators break the dequant->FMA dependency chain (the single biggest lever for a dequant-bound
// kernel — a B=1 GEMV has no other math to hide the unpack behind).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

template<int NACC>
__global__ void int4_gemv_kernel(
    const uint32_t* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int IN, int OUT, int G) {
  extern __shared__ float xs[];                 // [IN], shared by all rows in the block
  const int tid = threadIdx.x, nthreads = blockDim.x;
  for (int i = tid; i < IN; i += nthreads) xs[i] = __bfloat162float(x[i]);
  __syncthreads();

  const int lane = tid & 31;
  const int row  = blockIdx.x * (nthreads >> 5) + (tid >> 5);
  if (row >= OUT) return;

  const int ncols = IN >> 3;                    // uint32 columns
  const int cols_per_group = G >> 3;            // uint32 columns per scale group (=16 for G=128)
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * ncols;
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (IN / G);

  float acc[NACC];
  #pragma unroll
  for (int a = 0; a < NACC; a++) acc[a] = 0.f;

  // Each iteration consumes NACC*32 uint32 columns; lane l owns columns {base+a*32+l}. The a-loop is
  // fully independent -> NACC loads/dequants in flight before the first FMA needs its result.
  for (int base = lane; base < ncols; base += 32 * NACC) {
    #pragma unroll
    for (int a = 0; a < NACC; a++) {
      const int col = base + a * 32;
      if (col < ncols) {
        const uint32_t w = Wrow[col];
        const float sc = __bfloat162float(srow[col / cols_per_group]);
        const int k0 = col << 3;                // first input index this word covers
        #pragma unroll
        for (int k = 0; k < 8; k++) {
          const int q = (w >> (4 * k)) & 0xF;
          acc[a] += (float(q) - 8.0f) * sc * xs[k0 + k];
        }
      }
    }
  }

  float sum = 0.f;
  #pragma unroll
  for (int a = 0; a < NACC; a++) sum += acc[a];
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) y[row] = __float2bfloat16(sum);
}

torch::Tensor int4_gemv(torch::Tensor Wq, torch::Tensor scales, torch::Tensor x,
                        int64_t G, int64_t nacc) {
  const int OUT = Wq.size(0);
  const int ncols = Wq.size(1);
  const int IN = ncols * 8;
  auto y = torch::empty({OUT}, x.options());

  const uint32_t* wq = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());

  const int nthreads = 256;
  const int grid = (OUT + (nthreads >> 5) - 1) / (nthreads >> 5);
  const size_t shmem = (size_t)IN * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  switch (nacc) {
    case 2: int4_gemv_kernel<2><<<grid, nthreads, shmem, stream>>>(wq, sc, xp, yp, IN, OUT, G); break;
    case 4: int4_gemv_kernel<4><<<grid, nthreads, shmem, stream>>>(wq, sc, xp, yp, IN, OUT, G); break;
    case 6: int4_gemv_kernel<6><<<grid, nthreads, shmem, stream>>>(wq, sc, xp, yp, IN, OUT, G); break;
    case 8: int4_gemv_kernel<8><<<grid, nthreads, shmem, stream>>>(wq, sc, xp, yp, IN, OUT, G); break;
    default: TORCH_CHECK(false, "nacc must be 2/4/6/8");
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_gemv", &int4_gemv, "int4 weight-only GEMV (B=1)");
}
