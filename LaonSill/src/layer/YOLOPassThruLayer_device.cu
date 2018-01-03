/**
 * @file YOLOPassThruLayer_device.cu
 * @date 2018-01-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "YOLOPassThruLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

#define EPSILON                 0.000001

template <typename Dtype>
__global__ void YoloPassThruForward(const Dtype* input, int size, int channels, int rows, 
        int cols, Dtype* output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    int curBatch = idx / channels;
    int curChannel = idx % channels;

    int halfRows = rows / 2;
    int halfCols = cols / 2;

    // 1st quarter 
    int targetBaseIndex = curBatch * channels * rows * cols + 
        (curChannel + 0 * channels) * halfRows * halfCols;
    int sourceBaseIndex = idx * rows * cols;

    int curIndex = 0;
    for (int i = 0; i < halfRows; i++) {
        for (int j = 0; j < halfCols; j++) {
            output[targetBaseIndex + curIndex] = input[sourceBaseIndex + i * cols + j];
            curIndex = curIndex + 1;
        }
    }

    // 2nd quarter
    targetBaseIndex = curBatch * channels * rows * cols + 
        (curChannel + 1 * channels) * halfRows * halfCols;
    sourceBaseIndex = idx * rows * cols;

    curIndex = 0;
    for (int i = 0; i < halfRows; i++) {
        for (int j = 0; j < halfCols; j++) {
            output[targetBaseIndex + curIndex] = 
                input[sourceBaseIndex + i * cols + halfCols + j];
            curIndex = curIndex + 1;
        }
    }

    // 3rd quarter
    targetBaseIndex = curBatch * channels * rows * cols + 
        (curChannel + 2 * channels) * halfRows * halfCols;
    sourceBaseIndex = idx * rows * cols;

    curIndex = 0;
    for (int i = 0; i < halfRows; i++) {
        for (int j = 0; j < halfCols; j++) {
            output[targetBaseIndex + curIndex] = 
                input[sourceBaseIndex + i * cols + j + cols * halfRows];
            curIndex = curIndex + 1;
        }
    }

    // 4th quarter
    targetBaseIndex = curBatch * channels * rows * cols + 
        (curChannel + 3 * channels) * halfRows * halfCols;
    sourceBaseIndex = idx * rows * cols;

    curIndex = 0;
    for (int i = 0; i < halfRows; i++) {
        for (int j = 0; j < halfCols; j++) {
            output[targetBaseIndex + curIndex] = 
                input[sourceBaseIndex + i * cols + j + cols * halfRows + halfCols];
            curIndex = curIndex + 1;
        }
    }
}

template <typename Dtype>
__global__ void YoloPassThruBackward(const Dtype* outputGrad, int size, int channels, 
        int rows, int cols, Dtype* inputGrad) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    int curBatch = idx / channels;
    int curChannel = idx % channels;

    int halfRows = rows / 2;
    int halfCols = cols / 2;

    // 1st quarter 
    int targetBaseIndex = curBatch * channels * rows * cols + 
        (curChannel + 0 * channels) * halfRows * halfCols;
    int sourceBaseIndex = idx * rows * cols;

    int curIndex = 0;
    for (int i = 0; i < halfRows; i++) {
        for (int j = 0; j < halfCols; j++) {
            inputGrad[sourceBaseIndex + i * cols + j] = outputGrad[targetBaseIndex + curIndex];
            curIndex = curIndex + 1;
        }
    }

    // 2nd quarter
    targetBaseIndex = curBatch * channels * rows * cols + 
        (curChannel + 1 * channels) * halfRows * halfCols;
    sourceBaseIndex = idx * rows * cols;

    curIndex = 0;
    for (int i = 0; i < halfRows; i++) {
        for (int j = 0; j < halfCols; j++) {
            inputGrad[sourceBaseIndex + i * cols + halfCols + j] = 
                outputGrad[targetBaseIndex + curIndex];
            curIndex = curIndex + 1;
        }
    }

    // 3rd quarter
    targetBaseIndex = curBatch * channels * rows * cols + 
        (curChannel + 2 * channels) * halfRows * halfCols;
    sourceBaseIndex = idx * rows * cols;

    curIndex = 0;
    for (int i = 0; i < halfRows; i++) {
        for (int j = 0; j < halfCols; j++) {
            inputGrad[sourceBaseIndex + i * cols + j + cols * halfRows] =
                outputGrad[targetBaseIndex + curIndex];
            curIndex = curIndex + 1;
        }
    }

    // 4th quarter
    targetBaseIndex = curBatch * channels * rows * cols + 
        (curChannel + 3 * channels) * halfRows * halfCols;
    sourceBaseIndex = idx * rows * cols;

    curIndex = 0;
    for (int i = 0; i < halfRows; i++) {
        for (int j = 0; j < halfCols; j++) {
            inputGrad[sourceBaseIndex + i * cols + j + cols * halfRows + halfCols] =
                outputGrad[targetBaseIndex + curIndex];
            curIndex = curIndex + 1;
        }
    }
}


template <typename Dtype>
void YOLOPassThruLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({batches, channels * 4, rows / 2, cols / 2});
}

template <typename Dtype>
void YOLOPassThruLayer<Dtype>::feedforward() {
	reshape();

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int channelCount = inputShape[1];
    int rowCount = inputShape[2];
    int colCount = inputShape[3];
    int size = batchCount * channelCount;

    const Dtype *inputData = this->_inputData[0]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();

    YoloPassThruForward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, size, channelCount, rowCount, colCount, outputData);
}

template <typename Dtype>
void YOLOPassThruLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int channelCount = inputShape[1];
    int rowCount = inputShape[2];
    int colCount = inputShape[3];

    int size = batchCount * channelCount;

    const Dtype *outputGrad = this->_outputData[0]->device_grad();
    Dtype *inputGrad = this->_inputData[0]->mutable_device_grad();

    YoloPassThruBackward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        outputGrad, size, channelCount, rowCount, colCount, inputGrad);
}

template void YOLOPassThruLayer<float>::reshape();
template void YOLOPassThruLayer<float>::feedforward();
template void YOLOPassThruLayer<float>::backpropagation();

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* YOLOPassThruLayer<Dtype>::initLayer() {
	YOLOPassThruLayer* layer = NULL;
	SNEW(layer, YOLOPassThruLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool YOLOPassThruLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::backwardTensor(void* instancePtr) {
	YOLOPassThruLayer<Dtype>* layer = (YOLOPassThruLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void YOLOPassThruLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template void* YOLOPassThruLayer<float>::initLayer();
template void YOLOPassThruLayer<float>::destroyLayer(void* instancePtr);
template void YOLOPassThruLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool YOLOPassThruLayer<float>::allocLayerTensors(void* instancePtr);
template void YOLOPassThruLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void YOLOPassThruLayer<float>::backwardTensor(void* instancePtr);
template void YOLOPassThruLayer<float>::learnTensor(void* instancePtr);


