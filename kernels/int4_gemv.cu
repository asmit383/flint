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

  // Interleaved layout: K split into blocks of 256; within a block lane l owns the 8 indices
  // {l, l+32, ..., l+224} packed into one uint32 (weight m at bits [4m,4m+4) -> K = blk*256+l+32m).
  // Word col = blk*32+lane -> consecutive lanes = consecutive words (coalesced 128B load), and at a
  // fixed m consecutive lanes read consecutive x (banks 0..31, zero conflicts). Each word straddles
  // two 128-scale groups: m<4 -> scale[2*blk], m>=4 -> scale[2*blk+1].
  const int nblk = IN >> 8;                      // IN / 256
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * (nblk * 32);
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (nblk * 2);

  float acc[NACC];
  #pragma unroll
  for (int a = 0; a < NACC; a++) acc[a] = 0.f;

  for (int b = 0; b < nblk; b += NACC) {
    #pragma unroll
    for (int a = 0; a < NACC; a++) {
      const int blk = b + a;
      if (blk < nblk) {
        const uint32_t w = Wrow[blk * 32 + lane];
        const int base = (blk << 8) + lane;      // blk*256 + lane
        float xv[8];                             // preload: 8 independent LDS in flight, then compute
        #pragma unroll
        for (int m = 0; m < 8; m++) xv[m] = xs[base + (m << 5)];
        const float sc_lo = __bfloat162float(srow[2 * blk]);
        const float sc_hi = __bfloat162float(srow[2 * blk + 1]);
        float plo = 0.f, phi = 0.f;
        #pragma unroll
        for (int m = 0; m < 4; m++) plo += (float((w >> (4 * m)) & 0xF) - 8.0f) * xv[m];
        #pragma unroll
        for (int m = 4; m < 8; m++) phi += (float((w >> (4 * m)) & 0xF) - 8.0f) * xv[m];
        acc[a] += sc_lo * plo + sc_hi * phi;
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

// Split-K: blockIdx.y = k-split. Each block reduces only its K-slice for its rows, then atomicAdds the
// partial into a float accumulator. More blocks -> fills the SMs (fixes the 0.47-waves underfill on
// tall-K/few-row matmuls like down_proj); smaller x-slice in shared -> higher occupancy.
template<int NACC>
__global__ void int4_gemv_sk_kernel(
    const uint32_t* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ yf,
    int IN, int OUT, int nblk_per_split) {
  extern __shared__ float xs[];                  // [nblk_per_split*256], this split's x-slice
  const int tid = threadIdx.x, nthreads = blockDim.x;
  const int nblk = IN >> 8;
  const int blk0 = blockIdx.y * nblk_per_split;
  const int x_off = blk0 << 8;
  const int slice = nblk_per_split << 8;
  for (int i = tid; i < slice; i += nthreads) xs[i] = __bfloat162float(x[x_off + i]);
  __syncthreads();

  const int lane = tid & 31;
  const int row  = blockIdx.x * (nthreads >> 5) + (tid >> 5);
  if (row >= OUT) return;
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * (nblk * 32);
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (nblk * 2);

  float acc[NACC];
  #pragma unroll
  for (int a = 0; a < NACC; a++) acc[a] = 0.f;

  for (int b = blk0; b < blk0 + nblk_per_split; b += NACC) {
    #pragma unroll
    for (int a = 0; a < NACC; a++) {
      const int blk = b + a;
      if (blk < blk0 + nblk_per_split) {
        const uint32_t w = Wrow[blk * 32 + lane];
        const int base = ((blk << 8) + lane) - x_off;
        float xv[8];
        #pragma unroll
        for (int m = 0; m < 8; m++) xv[m] = xs[base + (m << 5)];
        const float sc_lo = __bfloat162float(srow[2 * blk]);
        const float sc_hi = __bfloat162float(srow[2 * blk + 1]);
        float plo = 0.f, phi = 0.f;
        #pragma unroll
        for (int m = 0; m < 4; m++) plo += (float((w >> (4 * m)) & 0xF) - 8.0f) * xv[m];
        #pragma unroll
        for (int m = 4; m < 8; m++) phi += (float((w >> (4 * m)) & 0xF) - 8.0f) * xv[m];
        acc[a] += sc_lo * plo + sc_hi * phi;
      }
    }
  }
  float sum = 0.f;
  #pragma unroll
  for (int a = 0; a < NACC; a++) sum += acc[a];
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) atomicAdd(&yf[row], sum);
}

torch::Tensor int4_gemv_sk(torch::Tensor Wq, torch::Tensor scales, torch::Tensor x,
                           int64_t splitk, int64_t nacc) {
  const int OUT = Wq.size(0);
  const int nblk = Wq.size(1) / 32;
  const int IN = nblk * 256;
  TORCH_CHECK(nblk % splitk == 0, "splitk must divide nblk (", nblk, ")");
  auto yf = torch::zeros({OUT}, x.options().dtype(torch::kFloat32));

  const uint32_t* wq = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  float* yp = yf.data_ptr<float>();

  const int nthreads = 256;
  const int nbps = nblk / splitk;
  dim3 grid((OUT + (nthreads >> 5) - 1) / (nthreads >> 5), splitk);
  const size_t shmem = (size_t)nbps * 256 * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (nacc) {
    case 2: int4_gemv_sk_kernel<2><<<grid, nthreads, shmem, stream>>>(wq, sc, xp, yp, IN, OUT, nbps); break;
    case 4: int4_gemv_sk_kernel<4><<<grid, nthreads, shmem, stream>>>(wq, sc, xp, yp, IN, OUT, nbps); break;
    case 8: int4_gemv_sk_kernel<8><<<grid, nthreads, shmem, stream>>>(wq, sc, xp, yp, IN, OUT, nbps); break;
    default: TORCH_CHECK(false, "nacc must be 2/4/8");
  }
  return yf.to(torch::kBFloat16);
}

// No-shared variant: read x straight from global (it's tiny, coalesced, and stays hot in L2). Frees
// all shared memory -> occupancy is no longer shared-limited, which is the wall on down_proj (large IN
// -> 32KB shared -> few blocks/SM). Trades L2 reads (cheap, ~10 TB/s) for parallelism.
template<int NACC>
__global__ void int4_gemv_g_kernel(
    const uint32_t* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int IN, int OUT, int G) {
  const int tid = threadIdx.x, nthreads = blockDim.x;
  const int lane = tid & 31;
  const int row  = blockIdx.x * (nthreads >> 5) + (tid >> 5);
  if (row >= OUT) return;
  const int nblk = IN >> 8;
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * (nblk * 32);
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (nblk * 2);

  float acc[NACC];
  #pragma unroll
  for (int a = 0; a < NACC; a++) acc[a] = 0.f;

  for (int b = 0; b < nblk; b += NACC) {
    #pragma unroll
    for (int a = 0; a < NACC; a++) {
      const int blk = b + a;
      if (blk < nblk) {
        const uint32_t w = Wrow[blk * 32 + lane];
        const int base = (blk << 8) + lane;
        float xv[8];
        #pragma unroll
        for (int m = 0; m < 8; m++) xv[m] = __bfloat162float(__ldg(&x[base + (m << 5)]));
        const float sc_lo = __bfloat162float(srow[2 * blk]);
        const float sc_hi = __bfloat162float(srow[2 * blk + 1]);
        float plo = 0.f, phi = 0.f;
        #pragma unroll
        for (int m = 0; m < 4; m++) plo += (float((w >> (4 * m)) & 0xF) - 8.0f) * xv[m];
        #pragma unroll
        for (int m = 4; m < 8; m++) phi += (float((w >> (4 * m)) & 0xF) - 8.0f) * xv[m];
        acc[a] += sc_lo * plo + sc_hi * phi;
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

torch::Tensor int4_gemv_g(torch::Tensor Wq, torch::Tensor scales, torch::Tensor x, int64_t nacc) {
  const int OUT = Wq.size(0);
  const int nblk = Wq.size(1) / 32;
  const int IN = nblk * 256;
  auto y = torch::empty({OUT}, x.options());
  const uint32_t* wq = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  const int nthreads = 256;
  const int grid = (OUT + (nthreads >> 5) - 1) / (nthreads >> 5);
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (nacc) {
    case 2: int4_gemv_g_kernel<2><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT, 128); break;
    case 4: int4_gemv_g_kernel<4><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT, 128); break;
    case 6: int4_gemv_g_kernel<6><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT, 128); break;
    case 8: int4_gemv_g_kernel<8><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT, 128); break;
    default: TORCH_CHECK(false, "nacc must be 2/4/6/8");
  }
  return y;
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
  m.def("int4_gemv_g", &int4_gemv_g, "int4 weight-only GEMV, x from global/L2 (B=1)");
  m.def("int4_gemv_sk", &int4_gemv_sk, "int4 weight-only GEMV, split-K (B=1)");
}
