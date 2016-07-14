/*
 * Cuda.cpp
 *
 *  Created on: 2016. 6. 16.
 *      Author: jhkim
 */

#include "Cuda.h"
#include "../Timer.h"

int Cuda::gpuid = 0;
cudnnHandle_t Cuda::cudnnHandle;
cublasHandle_t Cuda::cublasHandle;

Cuda::Cuda() {}
Cuda::~Cuda() {}

void Cuda::create(int gpuid) {
	int num_gpus;

	Timer timer;
	timer.start();
	checkCudaErrors(cudaGetDeviceCount(&num_gpus));
	if(gpuid < 0 || gpuid >= num_gpus) {
		printf("ERROR: Invalid GPU ID %d (There are %d GPUs on this machine)\n", gpuid, num_gpus);
	  exit(1);
	}
	//cout << "cudaGetDeviceCount: " << timer.stop(false) << endl;
	timer.start();
	Cuda::gpuid = gpuid;
	checkCudaErrors(cudaSetDevice(Cuda::gpuid));
	//cout << "cudaSetDevice: " << timer.stop(false) << endl;

	timer.start();
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	//cout << "cublasCreate: " << timer.stop(false) << endl;

	timer.start();
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));
	//cout << "cudnnCreate: " << timer.stop(false) << endl;
}

void Cuda::destroy() {
	checkCudaErrors(cudaSetDevice(Cuda::gpuid));
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));
}

void Cuda::refresh() {
	checkCudaErrors(cudaSetDevice(Cuda::gpuid));
}

















