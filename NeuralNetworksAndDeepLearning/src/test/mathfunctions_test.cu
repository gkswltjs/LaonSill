#if 0

#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "Cuda.h"
#include "SyncMem.h"

using namespace std;

enum CBLAS_TRANSPOSE {
	CblasNoTrans,
	CblasTrans,
};

void cublas_test();
void cudnn_test();

void setupCuda();
void cleanupCuda();

void fill_array(float* array, size_t size);
void print_array(const float* array, uint32_t rows, uint32_t cols);
void print_array_consecutive(const float* array, size_t size);



void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, const float* A, const float* B, const float beta,
		float* C) {
	// Note that cublas follows fortran order.
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cublasOperation_t cuTransA =
			(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB =
			(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, cuTransB, cuTransA,
			N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

int main(void) {
	//cublas_test();
	cudnn_test();
}

void cublas_test() {
	setupCuda();

	const uint32_t M = 3;		// rows of a
	const uint32_t N = 2;		// cols of a, rows of b
	const uint32_t K = 4;		// cols of b

	const size_t a_size = M*N;
	const size_t b_size = N*K;
	const size_t c_size = M*K;

	SyncMem<float> a, b, c;
	a.reshape(a_size);
	b.reshape(b_size);
	c.reshape(c_size);

	fill_array(a.mutable_host_mem(), a_size);
	fill_array(b.mutable_host_mem(), b_size);

	print_array_consecutive(a.host_mem(), a_size);
	print_array(a.host_mem(), M, N);
	cout << "---------" << endl;
	print_array_consecutive(b.host_mem(), b_size);
	print_array(b.host_mem(), N, K);

	caffe_gpu_gemm(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
			M, K, N, 1.0f, a.device_mem(), b.device_mem(), 0.0f, c.mutable_device_mem());

	print_array_consecutive(c.host_mem(), c_size);
	print_array(c.host_mem(), M, K);

	cleanupCuda();
}

void cudnn_test() {

}

void fill_array(float* array, size_t size) {
	//srand(time(0));

	for (size_t i = 0; i < size; i++) {
		array[i] = (float)rand() / RAND_MAX;
	}
}

void print_array(const float* array, uint32_t rows, uint32_t cols) {
	for (uint32_t i = 0; i < rows; i++) {
		for (uint32_t j = 0; j < cols; j++) {
			cout << array[i*cols + j] << ", ";
		}
		cout << endl;
	}
}

void print_array_consecutive(const float* array, size_t size) {
	for (size_t i = 0; i < size; i++) {
		cout << array[i] << ",";
	}
	cout << endl;
}


void setupCuda() {
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
}

void cleanupCuda() {
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
}





#endif
