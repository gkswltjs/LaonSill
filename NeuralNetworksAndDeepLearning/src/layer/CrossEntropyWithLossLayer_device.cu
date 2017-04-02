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
__global__ void CEForward(const Dtype* input, Dtype z, int size, Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x;
    if (input[idx] < 0.00001)
        x = 0.00001;
    else if (input[idx] > 0.99999)
        x = 0.99999;
    else
        x = input[idx];

    output[idx] += z * logf(x) + (1 - z) * logf(1 - x);
}

template <typename Dtype>
__global__ void CEForward2(const Dtype* input, const Dtype* input2, int size, Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x;
    Dtype z = input2[idx];
    if (input[idx] < 0.00001)
        x = 0.00001;
    else if (input[idx] > 0.99999)
        x = 0.99999;
    else
        x = input[idx];

    output[idx] = (-1.0) * (z * logf(x) + (1 - z) * logf(1 - x));
}

// Cross Entropy with logit(sigmoid): 
//  Loss : x - x * z + log (1 + exp(-x))        ....    x >= 0
//         -x * z + log(1 + exp(x))             ....    x < 0
//  x : input, z : target
template <typename Dtype>
__global__ void CEForwardWithSigmoid(const Dtype* input, Dtype z, int size, Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x;

    x = input[idx];
    if (x < 0) { 
        output[idx] = ((-1.0) * x * z + logf(1 + expf(x)));
    } else {
        output[idx] = (x - x * z + logf(1 + expf( (-1.0) * x)));
    }
}

// Cross Entropy with logit(sigmoid): 
//  Loss : x - x * z + log (1 + exp(-x))        ....    x >= 0
//         -x * z + log(1 + exp(x))             ....    x < 0
//  x : input, z : target
template <typename Dtype>
__global__ void CEForwardWithSigmoid2(const Dtype* input, const Dtype* input2, int size,
    Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x;
    Dtype z;

    x = input[idx];
    z = input2[idx];

    if (x < 0) { 
        output[idx] = ((-1.0) * x * z + logf(1 + expf(x)));
    } else {
        output[idx] = (x - x * z + logf(1 + expf( (-1.0) * x)));
    }
}

// gradient = x - z
// x : input
template <typename Dtype>
__global__ void CEBackward(const Dtype* input, const Dtype z, int size, Dtype* gradient) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    gradient[idx] = input[idx] - z;
}

// gradient = x - z
// x : input
template <typename Dtype>
__global__ void CEBackward2(const Dtype* input, const Dtype* input2, int size,
    Dtype* gradient) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    gradient[idx] = input[idx] - input2[idx];
}

// gradient : 1 - z - exp(-x) / (1 + exp(-x))   ....   x >= 0
//            -z + exp(x) / (1 + exp(x))        ....   x < 0 
// x : input
template <typename Dtype>
__global__ void CEBackwardWithSigmoid(const Dtype* input, const Dtype z, int size,
    Dtype* gradient) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x = input[idx];

    if (input[idx] < 0) {
        gradient[idx] = ((-1.0) * z + expf(x) / (1 + expf(x)));
    } else {
        gradient[idx] = (1 - z - expf((-1.0) * x) / (1 + expf((-1.0) * x)));
    }
}

// gradient : 1 - z - exp(-x) / (1 + exp(-x))   ....   x >= 0
//            -z + exp(x) / (1 + exp(x))        ....   x < 0 
// x : input
template <typename Dtype>
__global__ void CEBackwardWithSigmoid2(const Dtype* input, const Dtype* input2, int size,
    Dtype* gradient) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype x = input[idx];
    Dtype z = input2[idx];

    if (input[idx] < 0) {
        gradient[idx] = ((-1.0) * z + expf(x) / (1 + expf(x)));
    } else {
        gradient[idx] = (1 - z - expf((-1.0) * x) / (1 + expf((-1.0) * x)));
    }
}

template <typename Dtype>
CrossEntropyWithLossLayer<Dtype>::CrossEntropyWithLossLayer()
	: LossLayer<Dtype>() {
	initialize(0, false);
}

template <typename Dtype>
CrossEntropyWithLossLayer<Dtype>::CrossEntropyWithLossLayer(Builder* builder)
	: LossLayer<Dtype>(builder) {
	initialize(builder->_targetValue, builder->_withSigmoid);
}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::initialize(Dtype targetValue, bool withSigmoid) {
	this->type = Layer<Dtype>::CrossEntropyWithLoss;
    this->targetValue = targetValue;
    this->depth = 0;
    this->withSigmoid = withSigmoid;
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
	this->_outputData[0]->reshape({batches, 1, depth, 1});

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

    SASSERT0(this->_inputData.size() <= 2);
    if (this->_inputData.size() == 2) {
        // target value가 아닌 target values가 지정이 된 경우 
        const vector<uint32_t>& inputShape2 = this->_inputData[1]->getShape();
        SASSERT0(inputShape2[0] == inputShape[0]);
        SASSERT0(this->depth == this->_inputData[0]->getCountByAxis(1));
    }
}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::feedforward() {
	reshape();

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];

    const Dtype *inputData = this->_inputData[0]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();

    int count = this->depth * batchCount;

    if (this->withSigmoid) {
        if (this->_inputData.size() == 2) {
            const Dtype *inputData2 = this->_inputData[1]->device_data();
            CEForwardWithSigmoid2<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, inputData2, count, outputData);
        } else {
            CEForwardWithSigmoid<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, (Dtype)this->targetValue, count, outputData);
        }
    } else {
        if (this->_inputData.size() == 2) {
            const Dtype *inputData2 = this->_inputData[1]->device_data();
            CEForward2<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, inputData2, count, outputData);
        } else {
            CEForward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, (Dtype)this->targetValue, count, outputData);
        }
    }
}

template <typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];

    const Dtype *inputData = this->_inputData[0]->device_data();
	Dtype* inputGrads = this->_inputData[0]->mutable_device_grad();

    int count = batchCount * this->depth;

    if (this->withSigmoid) {
        if (this->_inputData.size() == 2) {
            const Dtype *inputData2 = this->_inputData[1]->device_data();
            CEBackwardWithSigmoid2<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, inputData2, count, inputGrads);
        } else {
            CEBackwardWithSigmoid<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, (Dtype)this->targetValue, count, inputGrads);
        }
    } else {
        if (this->_inputData.size() == 2) {
            const Dtype *inputData2 = this->_inputData[1]->device_data();
            CEBackward2<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, inputData2, count, inputGrads);
        } else {
            CEBackward<Dtype><<<SOOOA_GET_BLOCKS(count), SOOOA_CUDA_NUM_THREADS>>>(
                inputData, (Dtype)this->targetValue, count, inputGrads);
        }

    }
}

template <typename Dtype>
Dtype CrossEntropyWithLossLayer<Dtype>::cost() {
    const Dtype* outputData = this->_outputData[0]->host_data();
    Dtype avg = 0.0;
    for (int i = 0; i < this->depth; i++) {
        avg += outputData[i];
    }
	return avg / (Dtype)this->depth;
}

template<typename Dtype>
void CrossEntropyWithLossLayer<Dtype>::setTargetValue(Dtype value) {
    this->targetValue = value;
}

template class CrossEntropyWithLossLayer<float>;
