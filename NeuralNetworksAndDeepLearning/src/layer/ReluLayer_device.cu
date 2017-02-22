/**
 * @file ReluLayer_device.cu
 * @date 2017-02-15
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "ReluLayer.h"
#include "Exception.h"
#include "NetworkConfig.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "ColdLog.h"
#include "Perf.h"

using namespace std;

#ifdef GPU_MODE

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels
//

template <typename Dtype>
__global__ void ApplyLeakyForward(Dtype* output, int size, Dtype leaky)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    if (output[idx] < 0)
        output[idx] = leaky * output[idx];
}

template <typename Dtype>
__global__ void ApplyLeakyBackward(Dtype* inputGrad, int size, Dtype leaky)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    if (inputGrad[idx] <= 0)
        inputGrad[idx] = leaky;
}

template <typename Dtype>
void ReluLayer<Dtype>::applyLeakyForward() {
	int size = this->_outputData[0]->getCountByAxis(0);
    Dtype* outputData = this->_outputData[0]->mutable_device_data();

    ApplyLeakyForward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        outputData, size, (Dtype)this->leaky);
}

template <typename Dtype>
void ReluLayer<Dtype>::applyLeakyBackward() {
	int size = this->_outputData[0]->getCountByAxis(0);
    Dtype* inputGrad = this->_outputData[0]->mutable_device_grad();

    ApplyLeakyBackward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputGrad, size, (Dtype)this->leaky);
}

template void ReluLayer<float>::applyLeakyForward();
template void ReluLayer<float>::applyLeakyBackward();

#endif
