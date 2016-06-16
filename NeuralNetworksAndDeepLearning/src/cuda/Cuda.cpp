/*
 * Cuda.cpp
 *
 *  Created on: 2016. 6. 16.
 *      Author: jhkim
 */

#include "Cuda.h"

int Cuda::gpuid = 0;
cudnnHandle_t Cuda::cudnnHandle;
cublasHandle_t Cuda::cublasHandle;

Cuda::Cuda() {}
Cuda::~Cuda() {}

void Cuda::create(int gpuid) {
	int num_gpus;
	checkCudaErrors(cudaGetDeviceCount(&num_gpus));
	if(gpuid < 0 || gpuid >= num_gpus) {
		printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n", gpuid, num_gpus);
	  exit(1);
	}

	Cuda::gpuid = gpuid;
	checkCudaErrors(cudaSetDevice(Cuda::gpuid));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));
}

void Cuda::destroy() {
	checkCudaErrors(cudaSetDevice(Cuda::gpuid));
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));
}
