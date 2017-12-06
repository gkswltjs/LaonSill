/**
 * @file YOLOLossLayer_device.cu
 * @date 2017-04-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "YOLOLossLayer.h"
#include "Network.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;

#define YOLOLOSSLAYER_LOG         1

#define YOLO_GRID_COUNT             49
#define YOLO_GRID_ONE_AXIS_COUNT    7
#define YOLO_GRID_ELEM_COUNT        30

#define YOLO_CLASS_COUNT            20
#define YOLO_GROUND_TRUTH_ELEM_COUNT    (YOLO_CLASS_COUNT + 6)


#define EPSILON                 0.000001

template <typename Dtype>
__global__ void YoloBackward(const Dtype* input, const Dtype* input2, const Dtype* output,
    int size, Dtype noobj, Dtype coord, Dtype* grad) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype gridX = input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 0];
    Dtype gridY = input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 1];

    int gridXInt = (int)(gridX + EPSILON);
    int gridYInt = (int)(gridY + EPSILON);

    int gridIdx = gridXInt * YOLO_GRID_ONE_AXIS_COUNT + gridYInt;
    if (gridIdx != (idx % YOLO_GRID_COUNT)) {
        for (int i = 0; i < YOLO_GRID_ELEM_COUNT; i++) {
            grad[idx * YOLO_GRID_ELEM_COUNT + 0] = 0.0;
        }

        grad[idx * YOLO_GRID_ELEM_COUNT + 4] = 2.0 * output[idx + YOLO_GRID_ELEM_COUNT + 4];
        grad[idx * YOLO_GRID_ELEM_COUNT + 9] = 2.0 * output[idx + YOLO_GRID_ELEM_COUNT + 9];

        return;
    }

    // backward 1st box
    grad[idx * YOLO_GRID_ELEM_COUNT + 0] = 2.0 * output[idx + YOLO_GRID_ELEM_COUNT + 0];
    grad[idx * YOLO_GRID_ELEM_COUNT + 1] = 2.0 * output[idx + YOLO_GRID_ELEM_COUNT + 1];
    grad[idx * YOLO_GRID_ELEM_COUNT + 2] = output[idx + YOLO_GRID_ELEM_COUNT + 2] /
        sqrtf(input[idx + YOLO_GRID_ELEM_COUNT + 2] + EPSILON);
    grad[idx * YOLO_GRID_ELEM_COUNT + 3] = output[idx + YOLO_GRID_ELEM_COUNT + 3] /
        sqrtf(input[idx + YOLO_GRID_ELEM_COUNT + 3] + EPSILON);
    grad[idx * YOLO_GRID_ELEM_COUNT + 4] = 2.0 * output[idx + YOLO_GRID_ELEM_COUNT + 4];

    // backward 2nd box
    grad[idx * YOLO_GRID_ELEM_COUNT + 5] = 2.0 * output[idx + YOLO_GRID_ELEM_COUNT + 5];
    grad[idx * YOLO_GRID_ELEM_COUNT + 6] = 2.0 * output[idx + YOLO_GRID_ELEM_COUNT + 6];
    grad[idx * YOLO_GRID_ELEM_COUNT + 7] = output[idx + YOLO_GRID_ELEM_COUNT + 7] /
        sqrtf(input[idx + YOLO_GRID_ELEM_COUNT + 7] + EPSILON);
    grad[idx * YOLO_GRID_ELEM_COUNT + 8] = output[idx + YOLO_GRID_ELEM_COUNT + 8] /
        sqrtf(input[idx + YOLO_GRID_ELEM_COUNT + 8] + EPSILON);
    grad[idx * YOLO_GRID_ELEM_COUNT + 9] = 2.0 * output[idx + YOLO_GRID_ELEM_COUNT + 9];

    // backward class
    for (int i = 0; i < YOLO_CLASS_COUNT; i++) {
        grad[idx * YOLO_GRID_ELEM_COUNT + 10 + i] = 
            2.0 * output[idx + YOLO_GRID_ELEM_COUNT + 10 + i];
    }
}

// YOLO forward
// 소스가 너무 길다.. 정리해야 할꺼 같다.. 그런데 지금하기는 귀찮다.. ㅜㅜ
template <typename Dtype>
__global__ void YoloForward(const Dtype* input, const Dtype* input2, int size, 
    Dtype noobj, Dtype coord, Dtype* output) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype gridX = input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 0];
    Dtype gridY = input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 1];

    int gridXInt = (int)(gridX + EPSILON);
    int gridYInt = (int)(gridY + EPSILON);

    int gridIdx = gridXInt * YOLO_GRID_ONE_AXIS_COUNT + gridYInt;
    if (gridIdx != (idx % YOLO_GRID_COUNT)) {
        for (int i = 0; i < YOLO_GRID_ELEM_COUNT; i++) {
            output[idx * YOLO_GRID_ELEM_COUNT + 0] = 0.0;
        }

        Dtype c1 = input[idx * YOLO_GRID_ELEM_COUNT + 4];
        Dtype c2 = input[idx * YOLO_GRID_ELEM_COUNT + 9];
        output[idx * YOLO_GRID_ELEM_COUNT + 4] = noobj * (c1 - 1.0) * (c1 - 1.0);
        output[idx * YOLO_GRID_ELEM_COUNT + 9] = noobj * (c2 - 1.0) * (c2 - 1.0);
        return;
    }

    // 1st Box
    Dtype x1 = input[idx * YOLO_GRID_ELEM_COUNT + 0];
    Dtype y1 = input[idx * YOLO_GRID_ELEM_COUNT + 1];
    Dtype w1 = input[idx * YOLO_GRID_ELEM_COUNT + 2];
    Dtype h1 = input[idx * YOLO_GRID_ELEM_COUNT + 3];
    Dtype c1 = input[idx * YOLO_GRID_ELEM_COUNT + 4];

    // 2nd Box
    Dtype x2 = input[idx * YOLO_GRID_ELEM_COUNT + 5];
    Dtype y2 = input[idx * YOLO_GRID_ELEM_COUNT + 6];
    Dtype w2 = input[idx * YOLO_GRID_ELEM_COUNT + 7];
    Dtype h2 = input[idx * YOLO_GRID_ELEM_COUNT + 8];
    Dtype c2 = input[idx * YOLO_GRID_ELEM_COUNT + 9];

    // ground truth Box
    Dtype x = input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 2];
    Dtype y = input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 3];
    Dtype w = input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 4];
    Dtype h = input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 5];
    
    // calc 1st box iou
    Dtype left = max(x1 - w1 / 2.0, x - w / 2.0);
    Dtype right = min(x1 + w1 / 2.0, x + w / 2.0);
    Dtype top = max(y1 - h1 / 2.0, y - h / 2.0);
    Dtype bottom = min(y1 + h1 / 2.0, y + h / 2.0);
    Dtype ov_w = right - left;
    Dtype ov_h = bottom - top;

    Dtype b_inter;
    if (ov_w <= 0 || ov_h <= 0)
        b_inter = 0.0;
    else
        b_inter = ov_w * ov_h;
   
    Dtype b_union;
    b_union = w1 * h1 + w * h - b_inter;
    Dtype box1_iou = b_inter / b_union;

    // calc 2nd box iou
    left = max(x2 - w2 / 2.0, x - w / 2.0);
    right = min(x2 + w2 / 2.0, x + w / 2.0);
    top = max(y2 - h2 / 2.0, y - h / 2.0);
    bottom = min(y2 + h2 / 2.0, y + h / 2.0);
    ov_w = right - left;
    ov_h = bottom - top;

    if (ov_w <= 0 || ov_h <= 0)
        b_inter = 0.0;
    else
        b_inter = ov_w * ov_h;
   
    b_union = w2 * h2 + w * h - b_inter;
    Dtype box2_iou = b_inter / b_union;

    // forward 1st box
    if (box1_iou > 0 && box1_iou > box2_iou) {
        output[idx * YOLO_GRID_ELEM_COUNT + 0] = coord * (x1 - x) * (x1 - x);
        output[idx * YOLO_GRID_ELEM_COUNT + 1] = coord * (y1 - y) * (y1 - y);
        output[idx * YOLO_GRID_ELEM_COUNT + 2] = coord *
            (sqrtf(w1 + EPSILON) - sqrtf(w + EPSILON)) *
            (sqrtf(w1 + EPSILON) - sqrtf(w + EPSILON));
        output[idx * YOLO_GRID_ELEM_COUNT + 3] = coord *
            (sqrtf(h1 + EPSILON) - sqrtf(h + EPSILON)) *
            (sqrtf(h1 + EPSILON) - sqrtf(h + EPSILON));
        output[idx * YOLO_GRID_ELEM_COUNT + 4] = (c1 - 1.0) * (c1 - 1.0);
    } else {
        output[idx * YOLO_GRID_ELEM_COUNT + 0] = 0.0;
        output[idx * YOLO_GRID_ELEM_COUNT + 1] = 0.0;
        output[idx * YOLO_GRID_ELEM_COUNT + 2] = 0.0;
        output[idx * YOLO_GRID_ELEM_COUNT + 3] = 0.0;
        output[idx * YOLO_GRID_ELEM_COUNT + 4] = noobj * (c1 - 1.0) * (c1 - 1.0);
    }

    // forward 2nd box
    if (box2_iou > 0 && box2_iou > box1_iou) {
        output[idx * YOLO_GRID_ELEM_COUNT + 5] = coord * (x2 - x) * (x2 - x);
        output[idx * YOLO_GRID_ELEM_COUNT + 6] = coord * (y2 - y) * (y2 - y);
        output[idx * YOLO_GRID_ELEM_COUNT + 7] = coord *
            (sqrtf(w2 + EPSILON) - sqrtf(w + EPSILON)) *
            (sqrtf(w2 + EPSILON) - sqrtf(w + EPSILON));
        output[idx * YOLO_GRID_ELEM_COUNT + 8] = coord *
            (sqrtf(h2 + EPSILON) - sqrtf(h + EPSILON)) *
            (sqrtf(h2 + EPSILON) - sqrtf(h + EPSILON));
        output[idx * YOLO_GRID_ELEM_COUNT + 9] = (c2 - 1.0) * (c2 - 1.0);
    } else {
        output[idx * YOLO_GRID_ELEM_COUNT + 5] = 0.0;
        output[idx * YOLO_GRID_ELEM_COUNT + 6] = 0.0;
        output[idx * YOLO_GRID_ELEM_COUNT + 7] = 0.0;
        output[idx * YOLO_GRID_ELEM_COUNT + 8] = 0.0;
        output[idx * YOLO_GRID_ELEM_COUNT + 9] = noobj * (c2 - 1.0) * (c2 - 1.0);
    }

    // forward class
    for (int i = 0; i < 20; i++) {
        output[idx * YOLO_GRID_ELEM_COUNT + 10 + i] =
            (input[idx * YOLO_GRID_ELEM_COUNT + 10 + i] - 
            input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 6 + i]) * 
            (input[idx * YOLO_GRID_ELEM_COUNT + 10 + i] - 
            input2[idx * YOLO_GROUND_TRUTH_ELEM_COUNT + 6 + i]);
    }
}

template <typename Dtype>
YOLOLossLayer<Dtype>::YOLOLossLayer() : LossLayer<Dtype>() {
	this->type = Layer<Dtype>::YOLOLoss;
}

template<typename Dtype>
YOLOLossLayer<Dtype>::~YOLOLossLayer() {

}

template <typename Dtype>
void YOLOLossLayer<Dtype>::reshape() {
	if (!Layer<Dtype>::_adjustInputShape()) {
        const uint32_t count = Util::vecCountByAxis(this->_inputShape[0], 1);
        const uint32_t inputDataCount = this->_inputData[0]->getCountByAxis(1);
        assert(count == inputDataCount);
    }

    if (!Layer<Dtype>::_isInputShapeChanged(0))
        return;

    SASSERT0(this->_inputData.size() == 2);

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputShape;
	this->_outputData[0]->reshape(this->_inputShape[0]);

    const vector<uint32_t>& inputShape2 = this->_inputData[1]->getShape();
	this->_inputShape[1] = inputShape2;

	STDOUT_COND_LOG(YOLOLOSSLAYER_LOG, 
        "<%s> layer' input-0 has reshaped as: %dx%dx%dx%d\n",
        SLPROP_BASE(name).c_str(), inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
	STDOUT_COND_LOG(YOLOLOSSLAYER_LOG,
	    "<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n", 
        SLPROP_BASE(name).c_str(), inputShape[0], inputShape[1], inputShape[2], inputShape[3]);
}

template <typename Dtype>
void YOLOLossLayer<Dtype>::feedforward() {
	reshape();

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int size = batchCount * YOLO_GRID_COUNT;

    const Dtype *inputData = this->_inputData[0]->device_data();
    const Dtype *inputData2 = this->_inputData[1]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();

    YoloForward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, inputData2, size, (Dtype)SLPROP(YOLOLoss, noobj),
        (Dtype)SLPROP(YOLOLoss, coord), outputData);
}

template <typename Dtype>
void YOLOLossLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int size = batchCount * YOLO_GRID_COUNT;

    const Dtype *inputData = this->_inputData[0]->device_data();
    const Dtype *inputData2 = this->_inputData[1]->device_data();
    const Dtype *outputData = this->_outputData[0]->device_data();
    Dtype *outputGrad = this->_outputData[0]->mutable_device_grad();

    YoloBackward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, inputData2, outputData, size, (Dtype)SLPROP(YOLOLoss, noobj),
        (Dtype)SLPROP(YOLOLoss, coord), outputGrad);
}

template <typename Dtype>
Dtype YOLOLossLayer<Dtype>::cost() {
    const Dtype* outputData = this->_outputData[0]->host_data();
    Dtype avg = 0.0;

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int count = this->_outputData[0]->getCount();

    for (int i = 0; i < count; i++) {
        avg += outputData[i];
    }
	return avg / (Dtype)batchCount;
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* YOLOLossLayer<Dtype>::initLayer() {
	YOLOLossLayer* layer = NULL;
	SNEW(layer, YOLOLossLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(index < 2);
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(index == 0);
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool YOLOLossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::backwardTensor(void* instancePtr) {
    YOLOLossLayer<Dtype>* layer = (YOLOLossLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void YOLOLossLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class YOLOLossLayer<float>;
