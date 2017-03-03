/**
 * @file HyperTangentLayer_device.cu
 * @date 2017-03-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "HyperTangentLayer.h"
#include "Exception.h"
#include "NetworkConfig.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "ColdLog.h"
#include "Perf.h"

#define HYPERTANGENT_LOG   1

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

template <typename Dtype>
__global__ void Forward(const Dtype *input, int size, Dtype *output)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	output[idx] = tanh(input[idx]);
}

template <typename Dtype>
__global__ void Backward(const Dtype *outputGrad, const Dtype *output, int size,
    Dtype *inputGrad)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	inputGrad[idx] = outputGrad[idx] * (1.0 - output[idx] * output[idx]);
}

template <typename Dtype>
HyperTangentLayer<Dtype>::HyperTangentLayer() {
	this->type = Layer<Dtype>::HyperTangent;
}

template <typename Dtype>
HyperTangentLayer<Dtype>::HyperTangentLayer(Builder* builder)
	: HiddenLayer<Dtype>(builder) {
	initialize();
}

template <typename Dtype>
HyperTangentLayer<Dtype>::HyperTangentLayer(const string name) : HiddenLayer<Dtype>(name) {
	initialize();
}

template <typename Dtype>
void HyperTangentLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::HyperTangent;
}

template <typename Dtype>
void HyperTangentLayer<Dtype>::feedforward() {
    const Dtype* inputData = this->_inputData[0]->device_data();
    Dtype* outputData = this->_outputData[0]->mutable_device_data();
    int size = this->_inputData[0]->getCountByAxis(0);

    Forward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, size, outputData);
}

template <typename Dtype>
void HyperTangentLayer<Dtype>::backpropagation() {
	const Dtype* outputGrads = this->_outputData[0]->device_grad();
    const Dtype* outputData = this->_outputData[0]->device_data();
	Dtype* inputGrads = this->_inputData[0]->mutable_device_grad();
    int size = this->_inputData[0]->getCountByAxis(0);

    Backward<<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        outputGrads, outputData, size, inputGrads);
}

template <typename Dtype>
void HyperTangentLayer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
		const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
		const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
		assert(count == inputDataCount);
	}

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();

    // XXX: 현재 FC에 대해서만 생각하였음
    // TODO: Conv Layer에 대한 구현 필요
	uint32_t batches = inputShape[0];
	uint32_t channels = inputShape[1];
	uint32_t rows = inputShape[2];
	uint32_t cols = inputShape[3];

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({batches, channels, rows, cols});

	STDOUT_COND_LOG(HYPERTANGENT_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        this->name.c_str(), batches, channels, rows, cols);
	STDOUT_COND_LOG(HYPERTANGENT_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        this->name.c_str(), batches, channels, rows, cols);
}

template class HyperTangentLayer<float>;
