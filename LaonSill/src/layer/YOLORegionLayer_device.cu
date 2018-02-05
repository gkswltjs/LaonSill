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
        Dtype* output, bool softmax, int classNum) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= size)
		return;

    int elemPerAnchorBox = classNum + 4;
    int gridElemCount = YOLO_ANCHOR_BOX_COUNT * elemPerAnchorBox;

    for (int i = 0; i < YOLO_ANCHOR_BOX_COUNT; i++) {
        int outBoxIndex = idx * gridElemCount + i * elemPerAnchorBox;
        int inBoxIndex = idx + i * YOLO_GRID_COUNT * elemPerAnchorBox;

        Dtype x1 = input[inBoxIndex + 0 * YOLO_GRID_COUNT];
        Dtype y1 = input[inBoxIndex + 1 * YOLO_GRID_COUNT];
        Dtype w1 = input[inBoxIndex + 2 * YOLO_GRID_COUNT];
        Dtype h1 = input[inBoxIndex + 3 * YOLO_GRID_COUNT];
        Dtype c1 = input[inBoxIndex + 4 * YOLO_GRID_COUNT];

        output[outBoxIndex + 0] = 1.0 / (1.0 + expf((-1.0) * x1));
        output[outBoxIndex + 1] = 1.0 / (1.0 + expf((-1.0) * y1));

        output[outBoxIndex + 2] = 
            anchorVals[i * 2 + 0]  * expf(w1) / (Dtype)(YOLO_GRID_ONE_AXIS_COUNT);
        output[outBoxIndex + 3] = 
            anchorVals[i * 2 + 1] * expf(h1) / (Dtype)(YOLO_GRID_ONE_AXIS_COUNT);

        output[outBoxIndex + 4] = 1.0 / (1.0 + expf((-1.0) * c1));
    
        if (softmax) {
            // exponential 함수에서 매우 큰값이 나오는 것을 막기 위해서..
            Dtype sum = 0.0;
            Dtype maxVal = input[outBoxIndex + 5 + 0];
            for (int j = 1; j < classNum; j++) {
                if (input[inBoxIndex + (5 + j) * YOLO_GRID_COUNT] > maxVal)
                    maxVal = input[inBoxIndex + 5 + j];
            }

            for (int j = 0; j < classNum; j++) {
                Dtype class1 = input[inBoxIndex + (5 + j) * YOLO_GRID_COUNT] - maxVal;

                output[outBoxIndex + 5 + j] = expf(class1);
                sum += output[outBoxIndex + 5 + j];
            }

            for (int j = 0; j < classNum; j++) {
                output[outBoxIndex + 5 + j] = output[outBoxIndex + 5 + j] / (sum + EPSILON);
            }
        } else {
            for (int j = 0; j < classNum; j++) {
                output[outBoxIndex + 5 + j] = 1.0 / 
                    (1.0 + expf((-1.0) * input[inBoxIndex + (5 + j) * YOLO_GRID_COUNT]));
            }
        }
    }
}

template <typename Dtype>
__global__ void YoloRegionBackward(const Dtype* outputGrad, const Dtype* output, int size,
        Dtype* inputGrad, int classNum) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    int elemPerAnchorBox = classNum + 4;
    int gridElemCount = YOLO_ANCHOR_BOX_COUNT + elemPerAnchorBox;

    for (int i = 0; i < YOLO_ANCHOR_BOX_COUNT; i++) {
        int outBoxIndex = idx * gridElemCount + i * elemPerAnchorBox;
        int inBoxIndex = idx + i * YOLO_GRID_COUNT * elemPerAnchorBox;

        Dtype x1 = output[outBoxIndex + 0];
        Dtype y1 = output[outBoxIndex + 1];
        Dtype w1 = output[outBoxIndex + 2];
        Dtype h1 = output[outBoxIndex + 3];
        Dtype c1 = output[outBoxIndex + 4];

        inputGrad[inBoxIndex + 0 * YOLO_GRID_COUNT] = x1 * (1.0 - x1) *
            outputGrad[outBoxIndex + 0];
        inputGrad[inBoxIndex + 1 * YOLO_GRID_COUNT] = y1 * (1.0 - y1) *
            outputGrad[outBoxIndex + 1];
        inputGrad[inBoxIndex + 2 * YOLO_GRID_COUNT] = w1 * outputGrad[outBoxIndex + 2];
        inputGrad[inBoxIndex + 3 * YOLO_GRID_COUNT] = h1 * outputGrad[outBoxIndex + 3];
        inputGrad[inBoxIndex + 4 * YOLO_GRID_COUNT] = c1 * (1.0 - c1) * outputGrad[outBoxIndex + 4];

        for (int j = 0; j < classNum; j++) {
            inputGrad[inBoxIndex + (5 + j) * YOLO_GRID_COUNT] = 
                output[outBoxIndex + 5 + j] * (1.0 - output[outBoxIndex + 5 + j]) *
                outputGrad[outBoxIndex + 5 + j];
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
	this->_outputData[0]->reshape({batches, rows, cols, channels});
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
        inputData, size, anchorVals, outputData, SLPROP(YOLORegion, softmax),
        SLPROP(YOLORegion, numClasses));
}

template <typename Dtype>
void YOLORegionLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int size = batchCount * YOLO_GRID_COUNT;

    const Dtype *outputGrad = this->_outputData[0]->device_grad();
    const Dtype *outputData = this->_outputData[0]->device_data();
    Dtype *inputGrad = this->_inputData[0]->mutable_device_grad();

    YoloRegionBackward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        outputGrad, outputData, size, inputGrad, SLPROP(YOLORegion, numClasses));
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


