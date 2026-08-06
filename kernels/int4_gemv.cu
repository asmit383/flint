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
#include <mma.h>

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

// Global-x + split-K: the two P1/P2 wins together. No shared (high per-block occupancy) AND blockIdx.y
// splits K so the block COUNT multiplies -> fills the SMs on few-row shapes (down launches only 320
// blocks otherwise, ~2.4/SM). Partials atomicAdd into a float accumulator.
template<int NACC>
__global__ void int4_gemv_gsk_kernel(
    const uint32_t* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ yf,
    int IN, int OUT, int nblk_per_split) {
  const int tid = threadIdx.x, nthreads = blockDim.x;
  const int lane = tid & 31;
  const int row  = blockIdx.x * (nthreads >> 5) + (tid >> 5);
  if (row >= OUT) return;
  const int nblk = IN >> 8;
  const int blk0 = blockIdx.y * nblk_per_split;
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
  if (lane == 0) atomicAdd(&yf[row], sum);
}

torch::Tensor int4_gemv_gsk(torch::Tensor Wq, torch::Tensor scales, torch::Tensor x,
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
  dim3 grid((OUT + (nthreads >> 5) - 1) / (nthreads >> 5), splitk);
  const int nbps = nblk / splitk;
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (nacc) {
    case 2: int4_gemv_gsk_kernel<2><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT, nbps); break;
    case 4: int4_gemv_gsk_kernel<4><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT, nbps); break;
    case 8: int4_gemv_gsk_kernel<8><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT, nbps); break;
    default: TORCH_CHECK(false, "nacc must be 2/4/8");
  }
  return yf.to(torch::kBFloat16);
}

// Warp-split (intra-block split-K): one BLOCK per output row, W warps split that row's K-reduction,
// combined in shared -> no global atomic / memset / convert (the epilogue that sank int4_gemv_gsk).
// One block/row => OUT blocks (8x more than warp-per-row) => fills the SMs and finally lets occupancy
// hide the Long-Scoreboard load latency that caps glob at 31.7% occupancy / 41% MBU on down.
template<int NACC>
__global__ void int4_gemv_ws_kernel(
    const uint32_t* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int IN, int OUT) {
  const int row = blockIdx.x;
  if (row >= OUT) return;
  const int W = blockDim.x >> 5;                 // warps splitting K for this row
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nblk = IN >> 8;
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * (nblk * 32);
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (nblk * 2);

  float acc[NACC];
  #pragma unroll
  for (int a = 0; a < NACC; a++) acc[a] = 0.f;

  for (int base_b = warp; base_b < nblk; base_b += W * NACC) {
    #pragma unroll
    for (int a = 0; a < NACC; a++) {
      const int blk = base_b + a * W;
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

  __shared__ float wpart[32];                    // one partial per warp, then warp 0 reduces
  if (lane == 0) wpart[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float s = (lane < W) ? wpart[lane] : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if (lane == 0) y[row] = __float2bfloat16(s);
  }
}

torch::Tensor int4_gemv_ws(torch::Tensor Wq, torch::Tensor scales, torch::Tensor x,
                           int64_t splitw, int64_t nacc) {
  const int OUT = Wq.size(0);
  const int nblk = Wq.size(1) / 32;
  const int IN = nblk * 256;
  auto y = torch::empty({OUT}, x.options());
  const uint32_t* wq = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  const int nthreads = splitw * 32;              // W warps per row
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (nacc) {
    case 2: int4_gemv_ws_kernel<2><<<OUT, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    case 4: int4_gemv_ws_kernel<4><<<OUT, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    case 8: int4_gemv_ws_kernel<8><<<OUT, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    default: TORCH_CHECK(false, "nacc must be 2/4/8");
  }
  return y;
}

// Vectorized-x kernel. CONTIGUOUS layout (word j = K-indices [j*8, j*8+8)) instead of interleaved:
// interleaving only existed to dodge SHARED bank conflicts, but x lives in global/L2 now. Contiguous
// means a lane's 8 x-values are adjacent -> load all 8 as ONE 128-bit vector (int4) instead of 8 scalar
// __ldg. That's 8x fewer x-load instructions -> kills the Long-Scoreboard wall (65% of stalls came from
// the scalar x loads). Weight loads stay coalesced; 8 K < 128 so one scale per word.
template<int NACC>
__global__ void int4_gemv_v_kernel(
    const uint32_t* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int IN, int OUT) {
  const int tid = threadIdx.x, nthreads = blockDim.x;
  const int lane = tid & 31;
  const int row  = blockIdx.x * (nthreads >> 5) + (tid >> 5);
  if (row >= OUT) return;
  const int ncols = IN >> 3;                     // uint32 words per row
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * ncols;
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (IN >> 7);

  float acc[NACC];
  #pragma unroll
  for (int a = 0; a < NACC; a++) acc[a] = 0.f;

  for (int base = lane; base < ncols; base += 32 * NACC) {
    #pragma unroll
    for (int a = 0; a < NACC; a++) {
      const int col = base + a * 32;
      if (col < ncols) {
        const uint32_t w = Wrow[col];
        const float sc = __bfloat162float(srow[col >> 4]);          // 16 words per 128-group
        const int4 xr = __ldg(reinterpret_cast<const int4*>(&x[col << 3]));  // 8 bf16 = 128-bit load
        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xr);
        float p = 0.f;
        #pragma unroll
        for (int m = 0; m < 8; m++) p += (float((w >> (4 * m)) & 0xF) - 8.0f) * __bfloat162float(xb[m]);
        acc[a] += sc * p;
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

// Fast-dequant kernel. Once x-loads were vectorized the kernel became COMPUTE-bound (SM 50% > DRAM 22%)
// on the 8 scalar int->float converts per word. This builds the bf16 weight values straight from the
// int4 bits with a magic-number OR (no I2F): bf16 bits (0x4480 | nibble) == 1024 + 8*nibble exactly, so
// (V - 1088) == 8*(q-8). Then multiply-accumulate as bf16x2 (__hmul2) — half the math ops — and fold the
// /8 into the scale. Same contiguous layout as int4_gemv_v.
template<int NACC>
__global__ void int4_gemv_fd_kernel(
    const uint32_t* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int IN, int OUT) {
  const int tid = threadIdx.x, nthreads = blockDim.x;
  const int lane = tid & 31;
  const int row  = blockIdx.x * (nthreads >> 5) + (tid >> 5);
  if (row >= OUT) return;
  const int ncols = IN >> 3;
  const uint32_t* __restrict__ Wrow = Wq + (size_t)row * ncols;
  const __nv_bfloat16* __restrict__ srow = scales + (size_t)row * (IN >> 7);
  const __nv_bfloat162 magic = __float2bfloat162_rn(1088.0f);

  float acc[NACC];
  #pragma unroll
  for (int a = 0; a < NACC; a++) acc[a] = 0.f;

  for (int base = lane; base < ncols; base += 32 * NACC) {
    #pragma unroll
    for (int a = 0; a < NACC; a++) {
      const int col = base + a * 32;
      if (col < ncols) {
        const uint32_t w = Wrow[col];
        const float sc = __bfloat162float(srow[col >> 4]) * 0.125f;    // fold the /8
        const int4 xr = __ldg(reinterpret_cast<const int4*>(&x[col << 3]));
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xr);
        float p = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          uint32_t hb = ((w >> (8 * j)) & 0xFu) | (((w >> (8 * j + 4)) & 0xFu) << 16);
          hb |= 0x44804480u;                                            // two bf16 = 1024 + 8*nibble
          __nv_bfloat162 D = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&hb), magic);  // 8*(q-8)
          __nv_bfloat162 pr = __hmul2(D, x2[j]);
          p += __bfloat162float(pr.x) + __bfloat162float(pr.y);
        }
        acc[a] += sc * p;
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

// Activation-sparse int4 GEMV (down_proj): y = W[OUT,IN] @ h[IN], h ~50% sparse. To turn sparse h into
// FEWER BYTES READ, weights are COLUMN-major (W_col[IN, OUT/8]: input j's OUT weights contiguous). The
// kernel iterates input columns and skips j where |h[j]| < t — a UNIFORM branch (same j for all threads
// in the block) so no warp divergence, and the whole weight column is simply never loaded. j-split over
// blockIdx.y for parallelism (OUT=2560 alone underfills); partials atomicAdd into a float accumulator.
__global__ void int4_spmv_kernel(
    const uint32_t* __restrict__ Wcol,          // [IN, OUT/8]
    const __nv_bfloat16* __restrict__ scale_col,// [IN/128, OUT]
    const __nv_bfloat16* __restrict__ h,        // [IN]
    float thresh, float* __restrict__ yf,
    int IN, int OUT, int jspan) {
  const int OW = OUT >> 3;                        // output words (8 outputs each)
  const int owl = blockIdx.x * blockDim.x + threadIdx.x;
  if (owl >= OW) return;
  const int j0 = blockIdx.y * jspan;
  const int j1 = min(j0 + jspan, IN);

  float acc[8];
  #pragma unroll
  for (int m = 0; m < 8; m++) acc[m] = 0.f;

  int lastg = -1;
  float sc[8];
  for (int j = j0; j < j1; j++) {
    const float hj = __bfloat162float(h[j]);
    if (fabsf(hj) <= thresh) continue;            // keep |h| > t (matches ref/frac); uniform -> no divergence
    const int g = j >> 7;
    if (g != lastg) {                             // group scales constant over 128 j -> load once per group
      #pragma unroll
      for (int m = 0; m < 8; m++) sc[m] = __bfloat162float(scale_col[(size_t)g * OUT + owl * 8 + m]);
      lastg = g;
    }
    const uint32_t w = Wcol[(size_t)j * OW + owl]; // coalesced across threads (consecutive owl)
    #pragma unroll
    for (int m = 0; m < 8; m++)
      acc[m] += hj * (float((w >> (4 * m)) & 0xF) - 8.0f) * sc[m];
  }
  #pragma unroll
  for (int m = 0; m < 8; m++) atomicAdd(&yf[owl * 8 + m], acc[m]);
}

torch::Tensor int4_spmv(torch::Tensor Wcol, torch::Tensor scale_col, torch::Tensor h,
                        double thresh, int64_t OUT, int64_t jsplit) {
  const int IN = Wcol.size(0);
  auto yf = torch::zeros({OUT}, h.options().dtype(torch::kFloat32));
  const uint32_t* wc = reinterpret_cast<const uint32_t*>(Wcol.data_ptr<int32_t>());
  const __nv_bfloat16* scc = reinterpret_cast<const __nv_bfloat16*>(scale_col.data_ptr<at::BFloat16>());
  const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr<at::BFloat16>());
  float* yp = yf.data_ptr<float>();
  const int OW = OUT >> 3;
  const int nthreads = 256;
  const int jspan = (IN + jsplit - 1) / jsplit;
  dim3 grid((OW + nthreads - 1) / nthreads, jsplit);
  auto stream = at::cuda::getCurrentCUDAStream();
  int4_spmv_kernel<<<grid, nthreads, 0, stream>>>(wc, scc, hp, (float)thresh, yp, IN, OUT, jspan);
  return yf.to(torch::kBFloat16);
}

// Tensor-core (WMMA) int4 GEMV. Column-major weights ARE W^T = the MMA's B[K,N] matrix. Each warp owns
// an N-tile of 16 outputs, streams K in 16-chunks: dequant int4->bf16 on CUDA cores into a shared tile,
// put h into row 0 of the A[M=16,K=16] tile (M=1 -> rows 1..15 zero, the fundamental 15/16 MMA waste at
// B=1), mma_sync, accumulate. Reads all weights (the bytes) but offloads the MAC to the tensor core.
using namespace nvcuda;
__global__ void int4_gemv_wmma_kernel(
    const uint32_t* __restrict__ Wcol,          // [IN, OUT/8]  (= W^T packed, 8 outputs/word)
    const __nv_bfloat16* __restrict__ scale_col,// [IN/128, OUT]
    const __nv_bfloat16* __restrict__ h,        // [IN]
    __nv_bfloat16* __restrict__ y,
    int IN, int OUT) {
  const int warp_g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int n0 = warp_g * 16;                    // this warp's 16 outputs
  if (n0 >= OUT) return;
  const int lane = threadIdx.x & 31;
  const int wl = threadIdx.x >> 5;               // warp index within block
  const int OW = OUT >> 3;

  extern __shared__ __nv_bfloat16 sm[];          // per-warp: A[16*16] then B[16*16]
  __nv_bfloat16* At = sm + wl * 512;
  __nv_bfloat16* Bt = At + 256;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;

  for (int k0 = 0; k0 < IN; k0 += 16) {
    const int g = k0 >> 7;
    for (int i = lane; i < 256; i += 32) {       // fill A and B tiles (256 entries each)
      const int r = i >> 4, cc = i & 15;         // r=k-row, cc=col
      At[i] = (r == 0) ? h[k0 + cc] : __float2bfloat16(0.0f);   // A[m=r, k=cc]; only m=0 is real
      const uint32_t w = Wcol[(size_t)(k0 + r) * OW + ((n0 + cc) >> 3)];
      const int q = (w >> (4 * ((n0 + cc) & 7))) & 0xF;
      const float sc = __bfloat162float(scale_col[(size_t)g * OUT + n0 + cc]);
      Bt[i] = __float2bfloat16((float(q) - 8.0f) * sc);          // B[k=r, n=cc] = dequant W^T
    }
    __syncwarp();
    wmma::load_matrix_sync(a, At, 16);
    wmma::load_matrix_sync(b, Bt, 16);
    wmma::mma_sync(c, a, b, c);
    __syncwarp();
  }
  // c row 0 holds y[n0..n0+16]; store via shared float then write
  __shared__ float ctmp[256 * 8];                // per-warp 16x16 float
  float* Ct = ctmp + wl * 256;
  wmma::store_matrix_sync(Ct, c, 16, wmma::mem_row_major);
  if (lane < 16) y[n0 + lane] = __float2bfloat16(Ct[lane]);      // row 0
}

torch::Tensor int4_gemv_wmma(torch::Tensor Wcol, torch::Tensor scale_col, torch::Tensor h, int64_t OUT) {
  const int IN = Wcol.size(0);
  auto y = torch::empty({OUT}, h.options());
  const uint32_t* wc = reinterpret_cast<const uint32_t*>(Wcol.data_ptr<int32_t>());
  const __nv_bfloat16* scc = reinterpret_cast<const __nv_bfloat16*>(scale_col.data_ptr<at::BFloat16>());
  const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(h.data_ptr<at::BFloat16>());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>());
  const int warps_per_block = 8;
  const int nthreads = warps_per_block * 32;
  const int nwarps = (OUT + 15) / 16;
  const int grid = (nwarps + warps_per_block - 1) / warps_per_block;
  const size_t shmem = warps_per_block * 512 * sizeof(__nv_bfloat16);
  auto stream = at::cuda::getCurrentCUDAStream();
  int4_gemv_wmma_kernel<<<grid, nthreads, shmem, stream>>>(wc, scc, hp, yp, IN, OUT);
  return y;
}

torch::Tensor int4_gemv_fd(torch::Tensor Wq, torch::Tensor scales, torch::Tensor x, int64_t nacc) {
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
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (nacc) {
    case 2: int4_gemv_fd_kernel<2><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    case 4: int4_gemv_fd_kernel<4><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    case 6: int4_gemv_fd_kernel<6><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    case 8: int4_gemv_fd_kernel<8><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    default: TORCH_CHECK(false, "nacc must be 2/4/6/8");
  }
  return y;
}

torch::Tensor int4_gemv_v(torch::Tensor Wq, torch::Tensor scales, torch::Tensor x, int64_t nacc) {
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
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (nacc) {
    case 2: int4_gemv_v_kernel<2><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    case 4: int4_gemv_v_kernel<4><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    case 6: int4_gemv_v_kernel<6><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    case 8: int4_gemv_v_kernel<8><<<grid, nthreads, 0, stream>>>(wq, sc, xp, yp, IN, OUT); break;
    default: TORCH_CHECK(false, "nacc must be 2/4/6/8");
  }
  return y;
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
  m.def("int4_gemv_gsk", &int4_gemv_gsk, "int4 weight-only GEMV, global-x + split-K (B=1)");
  m.def("int4_gemv_ws", &int4_gemv_ws, "int4 weight-only GEMV, warp-split intra-block (B=1)");
  m.def("int4_gemv_v", &int4_gemv_v, "int4 weight-only GEMV, contiguous + 128-bit vector x-load (B=1)");
  m.def("int4_gemv_fd", &int4_gemv_fd, "int4 weight-only GEMV, fast bf16x2 dequant (B=1)");
  m.def("int4_spmv", &int4_spmv, "activation-sparse int4 GEMV, column-major skip (B=1)");
  m.def("int4_gemv_wmma", &int4_gemv_wmma, "tensor-core (WMMA) int4 GEMV (B=1)");
}
