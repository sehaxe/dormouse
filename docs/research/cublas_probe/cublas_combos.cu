// cuBLAS combo sweep: which (A/B, C, computeType) combinations does this
// driver accept, and what do they actually run at?
//
// Why this matters for dormouse: the stack's own tensor-core paths are dead
// (bf16 fails burn-spectral's tests on pre.4; f16 panics in cubecl-ir), and a
// hand-rolled WMMA kernel lost to a tiled fp32 SGEMM (8.4 vs 10.2 TFLOP/s).
// cuBLAS is the tuned path every framework actually uses. The interesting
// rows:
//   - TF32 (COMPUTE_32F_FAST_TF32): tensor cores on our EXISTING fp32 data,
//     zero dtype risk. If this is fast, it is the cheapest win available.
//   - bf16/f16 in, fp32 accumulate: the AMP path, what 1B needs.
//
// Build: nvcc -O3 -arch=sm_120 -o cublas_combos cublas_combos.cu -lcublas
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const char *dt_name(cudaDataType_t t) {
  switch (t) {
    case CUDA_R_32F: return "f32";
    case CUDA_R_16F: return "f16";
    case CUDA_R_16BF: return "bf16";
    default: return "?";
  }
}
static const char *ct_name(cublasComputeType_t c) {
  switch (c) {
    case CUBLAS_COMPUTE_32F: return "C32";
    case CUBLAS_COMPUTE_32F_FAST_TF32: return "C32_TF32";
    case CUBLAS_COMPUTE_32F_FAST_16F: return "C32_F16";
    case CUBLAS_COMPUTE_32F_FAST_16BF: return "C32_BF16";
    default: return "?";
  }
}

int main() {
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { printf("cublasCreate failed\n"); return 1; }
  const int M = 5120, K = 2048, N = 8192;  // dormouse 1B-scale FFN
  const double flops = 2.0 * M * K * N;
  const int iters = 20;

  void *A32, *B32, *C32, *A16, *B16;
  cudaMalloc(&A32, (size_t)M * K * 4);
  cudaMalloc(&B32, (size_t)K * N * 4);
  cudaMalloc(&C32, (size_t)M * N * 4);
  cudaMalloc(&A16, (size_t)M * K * 2);
  cudaMalloc(&B16, (size_t)K * N * 2);
  size_t hn = (size_t)M * K > (size_t)K * N ? (size_t)M * K : (size_t)K * N;
  std::vector<float> h16(hn, 0.01f);
  std::vector<__half> h16a((size_t)M * K), h16b((size_t)K * N);
  for (size_t i = 0; i < h16a.size(); ++i) h16a[i] = __float2half(h16[i]);
  for (size_t i = 0; i < h16b.size(); ++i) h16b[i] = __float2half(h16[i]);
  cudaMemcpy(A32, h16.data(), (size_t)M * K * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(B32, h16.data(), (size_t)K * N * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(A16, h16a.data(), (size_t)M * K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(B16, h16b.data(), (size_t)K * N * 2, cudaMemcpyHostToDevice);

  cudaDataType_t abts[] = {CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF};
  cudaDataType_t cts[] = {CUDA_R_32F, CUDA_R_16F, CUDA_R_16BF};
  cublasComputeType_t cts2[] = {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
                                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_COMPUTE_32F_FAST_16BF};

  printf("shape %dx%dx%d (%.1f GFLOP)\n", M, K, N, flops / 1e9);
  printf("%-6s %-6s %-10s %10s %10s %8s\n", "A/B", "C", "compute", "ms", "TFLOP/s", "status");
  for (auto abt : abts) {
    for (auto ct : cts) {
      for (auto comp : cts2) {
        // Reduced-precision inputs only make sense with the matching fast mode.
        if (abt == CUDA_R_16BF && comp != CUBLAS_COMPUTE_32F &&
            comp != CUBLAS_COMPUTE_32F_FAST_16BF) continue;
        if (abt == CUDA_R_16F && comp != CUBLAS_COMPUTE_32F &&
            comp != CUBLAS_COMPUTE_32F_FAST_16F) continue;
        const void *A = abt == CUDA_R_32F ? A32 : A16;
        const void *B = abt == CUDA_R_32F ? B32 : B16;
        const float alpha = 1.0f, beta = 0.0f;
        auto call = [&] {
          return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, abt, K, A,
                              abt, K, &beta, C32, ct, N, comp, CUBLAS_GEMM_DEFAULT);
        };
        cublasStatus_t st = call();
        if (st != CUBLAS_STATUS_SUCCESS) {
          printf("%-6s %-6s %-10s %10s %10s %8d\n", dt_name(abt), dt_name(ct), ct_name(comp), "-",
                 "-", (int)st);
          continue;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) { printf("sync fail\n"); continue; }
        // accuracy vs the fp32 reference
        std::vector<float> ref((size_t)M * N), got((size_t)M * N);
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B32, CUDA_R_32F, K, A32,
                     CUDA_R_32F, K, &beta, C32, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(ref.data(), C32, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
        // timed loop
        double maxrel = 0;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) call();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= iters;
        call();
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), C32, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < ref.size(); i += 101) {
          double den = std::fmax(1e-6, std::fabs((double)ref[i]));
          double d = std::fabs((double)got[i] - (double)ref[i]) / den;
          if (d > maxrel) maxrel = d;
        }
        printf("%-6s %-6s %-10s %10.3f %10.1f %8.2e\n", dt_name(abt), dt_name(ct), ct_name(comp),
               ms, flops / (ms * 1e-3) / 1e12, maxrel);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
      }
    }
  }
  cublasDestroy(h);
  return 0;
}
