// Tensor-core M=K int4 GEMM — the spec-decode verify at M>1. Y[M,OUT] = X[M,IN] @ dequant(W)[IN,OUT].
// int4 weights are dequantized to bf16 tiles on the fly and fed to WMMA (16x16x16 bf16). The M=1 "tensor
// cores waste 15/16" penalty reverses here: at M=8/16 the MMA m-tile is half/fully used. One warp owns a
// 16-wide OUT tile and accumulates over IN. M padded to 16. Contiguous layout (== int4_gemv_v).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_tc(const uint32_t* __restrict__ Wq, const __nv_bfloat16* __restrict__ scales,
                        const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ Y,
                        int IN, int OUT, int M) {
  const int warpid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int n0 = warpid * 16;                              // this warp's OUT columns [n0, n0+16)
  if (n0 >= OUT) return;
  extern __shared__ __nv_bfloat16 smem[];                  // per-warp: 256 W + 256 X (bf16) + 256 Y (as 512 bf16 = float)
  const int wib = threadIdx.x >> 5;
  __nv_bfloat16* Wsh = smem + wib * 1024;                  // [k=16][n=16] row-major
  __nv_bfloat16* Xsh = Wsh + 256;                          // [m=16][k=16] row-major
  float* Ysh = reinterpret_cast<float*>(Xsh + 256);        // [m=16][n=16] row-major (256 floats)

  const int ncw = IN >> 3, nsc = IN >> 7;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k = 0; k < IN; k += 16) {
    for (int e = lane; e < 256; e += 32) {                 // dequant W[n0..+16 (OUT), k..+16 (IN)] -> Wsh[k][n]
      const int i = e >> 4, j = e & 15;                    // i = n (OUT), j = k (IN)
      const int outrow = n0 + i, incol = k + j;
      const uint32_t word = Wq[(size_t)outrow * ncw + (incol >> 3)];
      const int nib = (word >> (4 * (incol & 7))) & 0xF;
      const float sc = __bfloat162float(scales[(size_t)outrow * nsc + (incol >> 7)]);
      Wsh[j * 16 + i] = __float2bfloat16((nib - 8) * sc);
    }
    for (int e = lane; e < 256; e += 32) {                 // load X[0..M, k..+16] -> Xsh[m][k], pad m>=M with 0
      const int i = e >> 4, j = e & 15;                    // i = m, j = k
      Xsh[i * 16 + j] = (i < M) ? X[(size_t)i * IN + (k + j)] : __float2bfloat16(0.0f);
    }
    __syncwarp();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
    wmma::load_matrix_sync(a, Xsh, 16);                    // A = X [m,k]
    wmma::load_matrix_sync(b, Wsh, 16);                    // B = W [k,n]
    wmma::mma_sync(acc, a, b, acc);
    __syncwarp();
  }
  wmma::store_matrix_sync(Ysh, acc, 16, wmma::mem_row_major);
  __syncwarp();
  for (int e = lane; e < M * 16; e += 32) {
    const int i = e >> 4, j = e & 15;
    Y[(size_t)i * OUT + (n0 + j)] = __float2bfloat16(Ysh[i * 16 + j]);
  }
}

torch::Tensor gemm_tc_launch(torch::Tensor Wq, torch::Tensor scales, torch::Tensor X, int64_t M) {
  const int OUT = Wq.size(0), ncols = Wq.size(1), IN = ncols * 8;
  auto Y = torch::empty({M, OUT}, X.options());
  const uint32_t* wq = reinterpret_cast<const uint32_t*>(Wq.data_ptr<int32_t>());
  const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>());
  const int nthreads = 256, wpb = nthreads >> 5;
  const int nwarps = (OUT + 15) / 16, grid = (nwarps + wpb - 1) / wpb;
  const size_t sh = (size_t)wpb * (512 * sizeof(__nv_bfloat16) + 256 * sizeof(float));  // W+X bf16 + Y float
  gemm_tc<<<grid, nthreads, sh, at::cuda::getCurrentCUDAStream()>>>(wq, sc, xp, yp, IN, OUT, (int)M);
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm_tc", &gemm_tc_launch, "tensor-core M=K int4 GEMM"); }
