/*
 * Cuda.h
 *
 *  Created on: 2016. 6. 16.
 *      Author: jhkim
 */

#ifndef CUDA_H_
#define CUDA_H_

#include <iostream>
#include <sstream>
#include <cudnn.h>
#include <cublas_v2.h>

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)







#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		i < (n); \
		i += blockDim.x * gridDim.x)











// Block width for CUDA kernels
#define BW 128

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
	return (nominator + denominator - 1) / denominator;
}



class Cuda {
public:
	Cuda();
	virtual ~Cuda();

	static int gpuid;
	static cudnnHandle_t cudnnHandle;
	static cublasHandle_t cublasHandle;

	static void create(int gpuid);
	static void destroy();
	static void refresh();

};

#endif /* CUDA_H_ */
