/**
 * @file CrossEntropyWithLossLayer_device.cu
 * @date 2017-02-06
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "CrossEntropyWithLossLayer.h"
#include "Exception.h"
#include "NetworkConfig.h"
#include "SysLog.h"
#include "StdOutLog.h"

#define CROSSENTROPYWITHLOSSLAYER_LOG   1

using namespace std;

// Cross Entropy : 
//  z * -log(x) + (1 - z) * -log(1 - x)
//  x : input
template <typename Dtype>
__global__ void Forward(const Dtype* input, Dtype z, int depth, int batchCount,
    Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth)
		return;

    output[idx] = 0;
    for (int i = 0; i < batchCount; i++) {
        int index = i * depth + idx;
        Dtype x;
        if (input[index] < 0.00001)
            x = 0.00001;
        else
            x = input[index];

        output[idx] += z * log(x) + (1 - z) * log(1 - x);
    }

    output[idx] = (-1.0) * output[idx] / (Dtype)batchCount;
}

// gradient = x - z
// x : input
template <typename Dtype>
__global__ void Backward(const Dtype* input, const Dtype z, int depth, int batchCount,
    Dtype* gradient) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= depth)
		return;

    gradient[idx] = 0;

    for (int i = 0; i < batchCount; i++) {
        int index = i * depth + idx;
        gradient[idx] += (input[index] - z);
    }
   
    gradient[idx] = gradient[idx] / (Dtype)batchCount;
}

template <typename Dtype>
CrossEntropyWithLossLayer<Dtype>::CrossEntropyWithLossLayer()
	: LossLayer<Dtype>() {
	initialize(0);
}

template <typename Dtype>
CrossEntropyWithLossLayer<Dtype>::CrossEntropyWithLossLayer(Builder* builder)
	: LossLayer<Dtype>(builder) {
	initialize(builder->_targetValue);
}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::initialize(Dtype targetValue) {
	this->type = Layer<Dtype>::CrossEntropyWithLoss;
    this->targetValue = targetValue;
    this->depth = 0;
}

template<typename Dtype>
CrossEntropyWithLossLayer<Dtype>::~CrossEntropyWithLossLayer() {

}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
        const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
        const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
        assert(count == inputDataCount);
    }

    if (!Layer<Dtype>::_isInputShapeChanged(0))
        return;

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();

	uint32_t batches = inputShape[0];
	uint32_t channels = inputShape[1];
	uint32_t rows = inputShape[2];
	uint32_t cols = inputShape[3];
    uint32_t depth = this->_inputData[0]->getCountByAxis(1);

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({1, 1, depth, 1});

	STDOUT_COND_LOG(CROSSENTROPYWITHLOSSLAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        this->name.c_str(), batches, channels, rows, cols);
	STDOUT_COND_LOG(CROSSENTROPYWITHLOSSLAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        this->name.c_str(), batches, channels, rows, cols);

    if (this->depth == 0) {
        this->depth = depth;
    } else {
        SASSERT(this->depth == depth, "old depth=%d, depth=%d", this->depth, depth);
    }
}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::feedforward() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];

    const Dtype *inputData = this->_inputData[0]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();

	Forward<Dtype><<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, (Dtype)this->targetValue, this->depth, batchCount, outputData);
}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];

    const Dtype *inputData = this->_inputData[0]->device_data();
	Dtype* inputGrads = this->_inputData[0]->mutable_device_grad();

	Backward<Dtype><<<SOOOA_GET_BLOCKS(this->depth), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, (Dtype)this->targetValue, this->depth, batchCount, inputGrads);
}

template <typename Dtype>
Dtype CrossEntropyWithLossLayer<Dtype>::cost() {
	return this->_outputData[0]->host_data()[0];
}

template class CrossEntropyWithLossLayer<float>;
