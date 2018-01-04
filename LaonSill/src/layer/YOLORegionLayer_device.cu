/**
 * @file YOLORegionLayer_device.cu
 * @date 2018-01-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "cuda_runtime.h"

#include "YOLOLossLayer.h"
#include "YOLORegionLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

#define EPSILON                 0.000001

template <typename Dtype>
__global__ void YoloRegionForward(const Dtype* input, int size, const Dtype* anchorVals,
        Dtype* output) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= size)
		return;

    for (int i = 0; i < YOLO_ANCHOR_BOX_COUNT; i++) {
        int boxBaseIndex = idx * YOLO_GRID_ELEM_COUNT + i * YOLO_ELEM_COUNT_PER_ANCHORBOX;

        Dtype x1 = input[boxBaseIndex + 0];
        Dtype y1 = input[boxBaseIndex + 1];
        Dtype w1 = input[boxBaseIndex + 2];
        Dtype h1 = input[boxBaseIndex + 3];
        Dtype c1 = input[boxBaseIndex + 4];

        output[boxBaseIndex + 0] = 1.0 / (1.0 + expf((-1.0) * x1));
        output[boxBaseIndex + 1] = 1.0 / (1.0 + expf((-1.0) * y1));

        output[boxBaseIndex + 2] = 
            anchorVals[i * 2 + 0]  * expf(w1) / (Dtype)(YOLO_GRID_ONE_AXIS_COUNT);
        output[boxBaseIndex + 3] = 
            anchorVals[i * 2 + 1] * expf(h1) / (Dtype)(YOLO_GRID_ONE_AXIS_COUNT);

        output[boxBaseIndex + 4] = 1.0 / (1.0 + expf((-1.0) * c1));
     
        // exponential 함수에서 매우 큰값이 나오는 것을 막기 위해서..
        Dtype sum = 0.0;
        Dtype maxVal = input[boxBaseIndex + 5 + 0];
        for (int j = 1; j < YOLO_CLASS_COUNT; j++) {
            if (input[boxBaseIndex + 5 + j] > maxVal)
                maxVal = input[boxBaseIndex + 5 + j];
        }

        for (int j = 0; j < YOLO_CLASS_COUNT; j++) {
            Dtype class1 = input[boxBaseIndex + 5 + j] - maxVal;

            output[boxBaseIndex + 5 + j] = expf(class1);
            sum += output[boxBaseIndex + 5 + j];
        }

        for (int j = 0; j < YOLO_CLASS_COUNT; j++) {
            output[boxBaseIndex + 5 + j] = output[boxBaseIndex + 5 + j] / (sum + EPSILON);
        }
    }
}

template <typename Dtype>
__global__ void YoloRegionBackward(const Dtype* output, int size, Dtype* inputGrad) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    for (int i = 0; i < YOLO_ANCHOR_BOX_COUNT; i++) {
        int boxBaseIndex = idx * YOLO_GRID_ELEM_COUNT + i * YOLO_ELEM_COUNT_PER_ANCHORBOX;

        Dtype x1 = output[boxBaseIndex + 0];
        Dtype y1 = output[boxBaseIndex + 1];
        Dtype w1 = output[boxBaseIndex + 2];
        Dtype h1 = output[boxBaseIndex + 3];
        Dtype c1 = output[boxBaseIndex + 4];

        inputGrad[boxBaseIndex + 0] = x1 * (1.0 - x1);
        inputGrad[boxBaseIndex + 1] = y1 * (1.0 - y1);
        inputGrad[boxBaseIndex + 2] = w1;
        inputGrad[boxBaseIndex + 3] = h1;
        inputGrad[boxBaseIndex + 4] = c1 * (1.0 - c1);

        for (int j = 0; j < YOLO_CLASS_COUNT; j++) {
            inputGrad[boxBaseIndex + 5 + j] = 
                output[boxBaseIndex + 5 + j] * (1.0 - output[boxBaseIndex + 5 + j]);
        }
    }
}


template <typename Dtype>
void YOLORegionLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	this->_inputShape[0] = {batches, channels, rows, cols};
	this->_outputData[0]->reshape({batches, channels, rows, cols});
    this->anchorSet->reshape({1, 1, 1, 10});

    Dtype* anchorData = (Dtype*)this->anchorSet->mutable_host_data();
    for (int i = 0; i < SLPROP(YOLORegion, anchors).size(); i++) {
        anchorData[i] = SLPROP(YOLORegion, anchors)[i];
    }

    for (int i = SLPROP(YOLORegion, anchors).size(); i < 10; i++) {
        anchorData[i] = 0.5;
    }
}

template <typename Dtype>
void YOLORegionLayer<Dtype>::feedforward() {
	reshape();

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int size = batchCount * YOLO_GRID_COUNT;

    const Dtype* anchorVals = this->anchorSet->device_data();
    const Dtype *inputData = this->_inputData[0]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();

    YoloRegionForward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, size, anchorVals, outputData);
}

template <typename Dtype>
void YOLORegionLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int size = batchCount * YOLO_GRID_COUNT;

    const Dtype *outputData = this->_outputData[0]->device_data();
    Dtype *inputGrad = this->_inputData[0]->mutable_device_grad();

    YoloRegionBackward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        outputData, size, inputGrad);
}

template void YOLORegionLayer<float>::reshape();
template void YOLORegionLayer<float>::feedforward();
template void YOLORegionLayer<float>::backpropagation();

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* YOLORegionLayer<Dtype>::initLayer() {
	YOLORegionLayer* layer = NULL;
	SNEW(layer, YOLORegionLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool YOLORegionLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::backwardTensor(void* instancePtr) {
	YOLORegionLayer<Dtype>* layer = (YOLORegionLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void YOLORegionLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template void* YOLORegionLayer<float>::initLayer();
template void YOLORegionLayer<float>::destroyLayer(void* instancePtr);
template void YOLORegionLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool YOLORegionLayer<float>::allocLayerTensors(void* instancePtr);
template void YOLORegionLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void YOLORegionLayer<float>::backwardTensor(void* instancePtr);
template void YOLORegionLayer<float>::learnTensor(void* instancePtr);


