#include "MathFunctions.h"
#include "Cuda.h"

template <typename Dtype>
void soooa_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void soooa_set<int>(const int N, const int alpha, int* Y);
template void soooa_set<float>(const int N, const float alpha, float* Y);
template void soooa_set<double>(const int N, const double alpha, double* Y);


template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void soooa_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void soooa_gpu_set<int>(const int N, const int alpha, int* Y);
template void soooa_gpu_set<float>(const int N, const float alpha, float* Y);
template void soooa_gpu_set<double>(const int N, const double alpha, double* Y);





template <typename Dtype>
void soooa_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    //if (Caffe::mode() == Caffe::GPU) {
//#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
//#else
//      NO_GPU;
//#endif
    //} else {
     // memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    //}
  }
}

template void soooa_copy<int>(const int N, const int* X, int* Y);
template void soooa_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void soooa_copy<float>(const int N, const float* X, float* Y);
template void soooa_copy<double>(const int N, const double* X, double* Y);




template <typename Dtype>
__global__ void sub_kernel(const uint32_t n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void soooa_gpu_sub<float>(const uint32_t N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void soooa_gpu_sub<double>(const uint32_t N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}



template <typename Dtype>
__global__ void mul_kernel(const uint32_t n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void soooa_gpu_mul<float>(const uint32_t N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void soooa_gpu_mul<double>(const uint32_t N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}


template <>
void soooa_gpu_dot<float>(const uint32_t n, const float* x, const float* y,
    float* out) {
  //CUBLAS_CHECK(cublasSdot(Cuda::cublasHandle, n, x, 1, y, 1, out));
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, n, x, 1, y, 1, out));
}

template <>
void soooa_gpu_dot<double>(const uint32_t n, const double* x, const double* y,
    double * out) {
	checkCudaErrors(cublasDdot(Cuda::cublasHandle, n, x, 1, y, 1, out));
}

template <>
void soooa_gpu_scal<float>(const uint32_t N, const float alpha, float *X) {
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, N, &alpha, X, 1));
}

template <>
void soooa_gpu_scal<double>(const uint32_t N, const double alpha, double *X) {
	checkCudaErrors(cublasDscal(Cuda::cublasHandle, N, &alpha, X, 1));
}

template <>
void soooa_gpu_axpy<float>(const uint32_t N, const float alpha, const float* X,
    float* Y) {
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, N, &alpha, X, 1, Y, 1));
}

template <>
void soooa_gpu_axpy<double>(const uint32_t N, const double alpha, const double* X,
    double* Y) {
	checkCudaErrors(cublasDaxpy(Cuda::cublasHandle, N, &alpha, X, 1, Y, 1));
}


template <>
void soooa_gpu_axpby<float>(const uint32_t N, const float alpha, const float* X,
    const float beta, float* Y) {
	soooa_gpu_scal<float>(N, beta, Y);
	soooa_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void soooa_gpu_axpby<double>(const uint32_t N, const double alpha, const double* X,
    const double beta, double* Y) {
	soooa_gpu_scal<double>(N, beta, Y);
	soooa_gpu_axpy<double>(N, alpha, X, Y);
}


template <>
void soooa_gpu_asum<float>(const int n, const float* x, float* y) {
	checkCudaErrors(cublasSasum(Cuda::cublasHandle, n, x, 1, y));
}

template <>
void soooa_gpu_asum<double>(const int n, const double* x, double* y) {
	checkCudaErrors(cublasDasum(Cuda::cublasHandle, n, x, 1, y));
}

void soooa_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
	  checkCudaErrors(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template<>
void soooa_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, const float* A, const float* B, const float beta, float* C) {
	// Note that cublas follows fortran order.
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;

	cublasOperation_t cuTransA =
	(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB =
	(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, cuTransB, cuTransA, N, M, K,
			&alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void soooa_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N,
		const float alpha, const float* A, const float* x, const float beta, float* y) {
	cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	checkCudaErrors(cublasSgemv(Cuda::cublasHandle, cuTransA, N, M, &alpha, A, N, x,
			1, &beta, y, 1));
}





