/**
 * @file	Cuda.h
 * @date	2016/6/16
 * @author	jhkim
 * @brief	Cuda관련 매크로 및 상수, 클래스 선언.
 * @details
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






/**
 * @brief Cuda 라이브러리 사용에 필요한 핸들을 생성하여 전역으로 관리하는 클래스.
 * @detail Cuda 장치를 설정하고 cudnn, cublas 관련 핸들을 전역으로 생성, 삭제, 리프레시 하는 역할을 함.
 */
class Cuda {
public:
	Cuda();
	virtual ~Cuda();

	/**
	 * @details 지정된 id이 Cuda 장치를 설정하고 cudnn, cublas 핸들을 생성.
	 * @param gpuid Cuda를 사용할 장치의 id
	 */
	static void create(int gpuid);
	/**
	 * @details create()를 통해 생성한 Cuda관련 리소스를 정리.
	 */
	static void destroy();
	/**
	 * @details Cuda를 실행할 장치 재설정.
	 */
	static void refresh();

	static int gpuid;							///< Cuda를 사용할 장치의 id
	static cudnnHandle_t cudnnHandle;			///< cudnn 라이브러리 컨텍스트 핸들.
	static cublasHandle_t cublasHandle;			///< cublas 라이브러리 컨텍스트 핸들.

	static const float alpha;
	static const float beta;

};

#endif /* CUDA_H_ */
