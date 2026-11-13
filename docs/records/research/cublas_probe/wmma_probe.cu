// Decisive micro-benchmark for the dormouse 1B question: on this exact GPU
// (RTX 5060 Ti), how much faster is a hand-written WMMA GEMM (f16 inputs,
// f32 accumulator, nvcc 13.4) than the f32 path our stack is limited to?
//
// Context: dormouse measures 3.5-7.6 TFLOP/s for fp32 GEMMs; LLMQ (2512.15306)
// reports 39 TFLOP/s for 0.5B bf16 on this card. If WMMA here lands anywhere
// near 40+ TFLOP/s, hand-writing the GEMM is the 1B enabler. If it lands at
// ~8, the whole tensor-core project is pointless and the fp32 path is the
// floor.
//
// Build: nvcc -O3 -arch=sm_120 -o wmma_probe wmma_probe.cu
#include <cstdio>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <algorithm>

using namespace nvcuda;

#define CHECK(x)                                                               \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__);        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// ── fp32 baseline: the classic tiled SGEMM, 128x128 tile, 8x8 per thread ──
#define TS 64
#define TN 4
__global__ void sgemm(const float *A, const float *B, float *C, int M, int N, int K) {
  __shared__ float As[TS][TS + 1];
  __shared__ float Bs[TS + 1][TS];
  const int tx = threadIdx.x, ty = threadIdx.y;  // 16x16 threads, 4x4 each
  const int row = blockIdx.y * TS + ty * TN;
  const int col = blockIdx.x * TS + tx * TN;
  float acc[TN][TN] = {};
  for (int k0 = 0; k0 < K; k0 += TS) {
#pragma unroll
    for (int i = 0; i < TN; ++i)
      As[ty * TN + i][tx] = (row + i < M && k0 + tx < K) ? A[(size_t)(row + i) * K + k0 + tx] : 0.f;
#pragma unroll
    for (int i = 0; i < TN; ++i)
      Bs[tx * TN + i][ty] = (k0 + ty * TN + i < K && col + i < N) ? B[(size_t)(k0 + ty * TN + i) * N + col + i] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TS; ++k)
#pragma unroll
      for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += As[ty * TN + i][k] * Bs[k][tx * TN + j];
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TN; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      if (row + i < M && col + j < N) C[(size_t)(row + i) * N + col + j] = acc[i][j];
}

// ── WMMA: 16x16x16 f16 fragments, f32 accumulator (the AMP path) ──
__global__ void wmma_gemm(const half *A, const half *B, float *C, int M, int N, int K) {
  // Row-major A [M,K], col-major B [K,N] so both feed wmma directly.
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  int warp = (threadIdx.x + threadIdx.y * blockDim.x) / 32;
  int lane = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
  int warp_m = blockIdx.y * (blockDim.y * 2) + warp * 16;   // 2 warps tall
  int warp_n = blockIdx.x * (blockDim.x * 2) + (warp % 2) * 16;  // 2 warps wide
  wmma::fill_fragment(acc, 0.0f);
  for (int k0 = 0; k0 < K; k0 += 16) {
    __shared__ half As[32][16];  // 2 warps tall x 16 k
    __shared__ half Bs[16][32];  // 16 k x 2 warps wide
    int t = threadIdx.x + threadIdx.y * blockDim.x;
    int nthreads = blockDim.x * blockDim.y;
    for (int i = t; i < 32 * 16; i += nthreads) {
      int r = i / 16, c = i % 16;
      int gm = warp_m * 0 + r + (warp / 2) * 0;  // filled below
      (void)gm;
      int m_row = warp_m - (warp / 2) * 0 + r;
      // As rows follow the block's 2-warp-tall tile
      int tile_m = blockIdx.y * (blockDim.y * 2) * 0;  // unused
      (void)tile_m;
      int rowA = blockIdx.y * (blockDim.y * 2) + r;
      As[i / 16][i % 16] = (rowA < M && k0 + i % 16 < K) ? A[(size_t)rowA * K + k0 + i % 16] : __float2half(0.f);
      int colB = blockIdx.x * (blockDim.x * 2) + r;
      Bs[i % 16][i / 16] = (k0 + i % 16 < K && colB < N) ? B[(size_t)(k0 + i % 16) * N + colB] : __float2half(0.f);
    }
    __syncthreads();
    wmma::load_matrix_sync(a_frag, &As[(warp / 2) * 16][0], 16);
    wmma::load_matrix_sync(b_frag, &Bs[0][(warp % 2) * 16], 16);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
  }
  int m_row = blockIdx.y * (blockDim.y * 2) + (warp / 2) * 16 + (lane / 4) * 0;
  // Store through the accumulator layout.
  float *cptr = C + (size_t)(blockIdx.y * (blockDim.y * 2) + (warp / 2) * 16) * N + blockIdx.x * (blockDim.x * 2) + (warp % 2) * 16;
  wmma::store_matrix_sync(cptr, acc, N, wmma::mem_row_major);
  (void)m_row;
}

int main() {
  const int M = 5120, K = 2048, N = 8192;  // dormouse's 1B-scale FFN shape
  const int iters = 20;
  double flops = 2.0 * M * K * N;

  float *A32, *B32, *C32;
  half *A16, *B16;
  CHECK(cudaMalloc(&A32, sizeof(float) * M * K));
  CHECK(cudaMalloc(&B32, sizeof(float) * K * N));
  CHECK(cudaMalloc(&C32, sizeof(float) * M * N));
  CHECK(cudaMalloc(&A16, sizeof(half) * M * K));
  CHECK(cudaMalloc(&B16, sizeof(half) * K * N));
  std::vector<float> h((size_t)std::max(M * K, K * N), 0.01f);
  CHECK(cudaMemcpy(A32, h.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B32, h.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(A16, h.data(), sizeof(half) * M * K, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B16, h.data(), sizeof(half) * K * N, cudaMemcpyHostToDevice));

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  // sgemm
  dim3 tb(16, 16), gb((N + TS - 1) / TS, (M + TS - 1) / TS);
  sgemm<<<gb, tb>>>(A32, B32, C32, M, N, K);
  CHECK(cudaDeviceSynchronize());
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) sgemm<<<gb, tb>>>(A32, B32, C32, M, N, K);
  cudaEventRecord(e1);
  CHECK(cudaEventSynchronize(e1));
  float ms_f32;
  cudaEventElapsedTime(&ms_f32, e0, e1);
  ms_f32 /= iters;

  // wmma
  dim3 wb(32, 2), wg((N + 63) / 64, (M + 31) / 32);
  wmma_gemm<<<wg, wb>>>(A16, B16, C32, M, N, K);
  CHECK(cudaDeviceSynchronize());
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) wmma_gemm<<<wg, wb>>>(A16, B16, C32, M, N, K);
  cudaEventRecord(e1);
  CHECK(cudaEventSynchronize(e1));
  float ms_f16;
  cudaEventElapsedTime(&ms_f16, e0, e1);
  ms_f16 /= iters;

  // correctness spot-check on the wmma path
  std::vector<float> c(M * N);
  CHECK(cudaMemcpy(c.data(), C32, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
  double ref = 0.01 * 0.01 * 16;  // K-block contribution, rough sanity only
  printf("wmma C[0..3] = %.4f %.4f %.4f %.4f (rough ref %.4f)\n", c[0], c[1], c[2], c[3], ref);

  printf("shape %dx%dx%d (%.1f GFLOP)\n", M, K, N, flops / 1e9);
  printf("  fp32 sgemm : %8.3f ms  %6.1f TFLOP/s\n", ms_f32, flops / (ms_f32 * 1e-3) / 1e12);
  printf("  wmma f16   : %8.3f ms  %6.1f TFLOP/s   (%.2fx)\n", ms_f16, flops / (ms_f16 * 1e-3) / 1e12,
         ms_f32 / ms_f16);
  return 0;
}
