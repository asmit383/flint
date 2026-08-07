// Is a hand-rolled resident-grid barrier competitive with cooperative-groups grid.sync()? Times the cost
// of ONE grid-wide barrier in isolation. If the hand-rolled path is competitive, hand-rolling on CDNA
// (where cooperative groups may not carry over) is a small step.
//
//   default:              grid.sync()
//   -DHANDROLL:           hand-rolled sense-reversing counter barrier, CUDA C++ atomics (atomicInc + fence)
//   -DHANDROLL -DPTX:     same barrier, core written in inline PTX (atom.inc.u32 / membar.gl / ld.volatile)
// Launched cooperatively so all blocks are co-resident (the barrier deadlocks otherwise).
// STATUS: scaffold — runs on a box; PTX path unverified pending hardware.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#ifdef HANDROLL
// One thread per block arrives on a global counter; the last arriver flips a global sense flag to release
// the spinners. Sense alternates each call so consecutive barriers don't race on a stale flag.
__device__ __forceinline__ void hand_barrier(unsigned int* counter, volatile unsigned int* sense,
                                             int* block_sense, unsigned int ngrid) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned int s = (unsigned int)(!(*block_sense));
    *block_sense = (int)s;
#ifdef PTX
    unsigned int old;
    asm volatile("membar.gl;" ::: "memory");
    asm volatile("atom.inc.u32 %0, [%1], %2;" : "=r"(old) : "l"(counter), "r"(ngrid - 1) : "memory");
    if (old == ngrid - 1) {
      asm volatile("st.global.u32 [%0], %1;" :: "l"((unsigned int*)sense), "r"(s) : "memory");
    } else {
      unsigned int v;
      do { asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"((unsigned int*)sense) : "memory"); }
      while (v != s);
    }
#else
    __threadfence();
    const unsigned int old = atomicInc(counter, ngrid - 1); // wraps 0 at the ngrid-th arrival -> ready again
    if (old == ngrid - 1) *sense = s;                       // last block releases everyone
    else while (*sense != s) { }                            // others spin on the sense flip
#endif
  }
  __syncthreads();
}
#endif

__global__ void barrier_bench(unsigned int* counter, volatile unsigned int* sense, int iters) {
  cg::grid_group grid = cg::this_grid();
  __shared__ int block_sense;
  if (threadIdx.x == 0) block_sense = 0;
  __syncthreads();
  const unsigned int ngrid = gridDim.x;
  for (int i = 0; i < iters; i++) {
#ifdef HANDROLL
    hand_barrier(counter, sense, &block_sense, ngrid);
#else
    grid.sync(); (void)ngrid; (void)block_sense;
#endif
  }
}

// per-barrier time in nanoseconds
double barrier_ns(int64_t iters) {
  int dev = 0; cudaGetDevice(&dev);
  int nthreads = 256, bps = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, barrier_bench, nthreads, 0);
  int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  int grid = bps * nsm;                                    // exactly the co-resident set

  unsigned int *counter, *sense;
  cudaMalloc(&counter, sizeof(unsigned int)); cudaMalloc(&sense, sizeof(unsigned int));
  cudaMemset(counter, 0, sizeof(unsigned int)); cudaMemset(sense, 0, sizeof(unsigned int));
  auto stream = at::cuda::getCurrentCUDAStream();

  int w = 128; void* wargs[] = {&counter, &sense, &w};
  cudaLaunchCooperativeKernel((void*)barrier_bench, grid, nthreads, wargs, 0, stream);
  cudaStreamSynchronize(stream);

  int it = (int)iters; void* args[] = {&counter, &sense, &it};
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, stream);
  cudaLaunchCooperativeKernel((void*)barrier_bench, grid, nthreads, args, 0, stream);
  cudaEventRecord(b, stream); cudaEventSynchronize(b);
  float ms = 0; cudaEventElapsedTime(&ms, a, b);
  cudaFree(counter); cudaFree(sense);
  return (double)ms * 1e6 / (double)iters;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("barrier_ns", &barrier_ns, "per-grid-barrier cost in ns");
#if defined(HANDROLL) && defined(PTX)
  m.attr("kind") = "handroll-ptx";
#elif defined(HANDROLL)
  m.attr("kind") = "handroll-c++";
#else
  m.attr("kind") = "grid.sync";
#endif
}
