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

#define EPSILON                 0.000001

template <typename Dtype>
__global__ void YoloBackward(const Dtype* input, const Dtype* input2,
    int size, Dtype noobjVal, Dtype coordVal, Dtype objVal, Dtype classVal, Dtype* grad) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= size)
		return;

    // NOTE: 아래 코드는 필요 없습니다. forward에서 backward까지 진행하기 때문입니다.
#if 0
    Dtype labelClass = input2[idx * YOLOINPUT_ELEMCOUNT_PER_GRID + 6];

    int labelClassInt = (int)(labelClass + EPSILON);

    if (labelClassInt == 0) {
        for (int i = 0; i < YOLO_GRID_ELEM_COUNT; i++) {
            grad[idx * YOLO_GRID_ELEM_COUNT + i] = 0.0;
        }

        for (int j = 0; j < YOLO_ANCHOR_BOX_COUNT; j++) {
            int confidenceIndex = 
                idx * YOLO_GRID_ELEM_COUNT + j * YOLO_ELEM_COUNT_PER_ANCHORBOX + 4;
            grad[confidenceIndex] = noobjVal * (2.0) * input[confidenceIndex];
        }

        return;
    }

    Dtype x = input2[idx * YOLOINPUT_ELEMCOUNT_PER_GRID + 2];
    Dtype y = input2[idx * YOLOINPUT_ELEMCOUNT_PER_GRID + 3];
    Dtype w = input2[idx * YOLOINPUT_ELEMCOUNT_PER_GRID + 4];
    Dtype h = input2[idx * YOLOINPUT_ELEMCOUNT_PER_GRID + 5];

    // backward boxes & classes
    for (int i = 0; i < YOLO_ANCHOR_BOX_COUNT; i++) {
        int boxBaseIndex = idx * YOLO_GRID_ELEM_COUNT + i * YOLO_ELEM_COUNT_PER_ANCHORBOX;

        grad[boxBaseIndex + 0] = coordVal * (2.0) * (input[boxBaseIndex + 0] - x);
        grad[boxBaseIndex + 1] = coordVal * (2.0) * (input[boxBaseIndex + 1] - y);
        grad[boxBaseIndex + 2] = 
            coordVal * (sqrtf(input[boxBaseIndex + 2] + EPSILON) - sqrtf(w + EPSILON))
                    / sqrtf(input[boxBaseIndex + 2] + EPSILON);
        grad[boxBaseIndex + 3] = 
            coordVal * (sqrtf(input[boxBaseIndex + 3] + EPSILON) - sqrtf(h + EPSILON))
                    / sqrtf(input[boxBaseIndex + 3] + EPSILON);
        grad[boxBaseIndex + 4] = objVal * (2.0) * (input[boxBaseIndex + 4] - 1.0);

        for (int j = 0; j < YOLO_CLASS_COUNT; j++) {
            if (j == labelClassInt - 1) {
                grad[boxBaseIndex + 5 + j] =
                    classVal * (2.0) * (input[boxBaseIndex + 5 + j] - 1.0);
            } else {
                grad[boxBaseIndex + 5 + j] = 
                    classVal * (2.0) * (input[boxBaseIndex + 5 + j]);
            }
        }
    }
#endif
}

// YOLO forward
// 소스가 너무 길다.. 정리해야 할꺼 같다.. 그런데 지금하기는 귀찮다.. ㅜㅜ
template <typename Dtype>
__global__ void YoloForward(const Dtype* input, const Dtype* input2, int size, 
    Dtype noobjVal, Dtype coordVal, Dtype objVal, Dtype classVal, Dtype* output, 
    Dtype* inputGrad) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype labelClass = input2[idx * YOLOINPUT_ELEMCOUNT_PER_GRID + 6];  // 첫번째 label

    int labelClassInt = (int)(labelClass + EPSILON);

    if (labelClassInt == 0) {

        output[idx] = 0.0;

        for (int i = 0; i < YOLO_ANCHOR_BOX_COUNT; i++) {
            int confidenceIndex = 
                idx * YOLO_GRID_ELEM_COUNT + i * YOLO_ELEM_COUNT_PER_ANCHORBOX + 4;
            int boxBaseIndex = idx * YOLO_GRID_ELEM_COUNT + i * 
                YOLO_ELEM_COUNT_PER_ANCHORBOX;
            Dtype c1 = input[confidenceIndex];
            output[idx] = output[idx] + noobjVal * (c1 - 0.0) * (c1 - 0.0);

            for (int j = 0 ; j < YOLO_ELEM_COUNT_PER_ANCHORBOX; j++) {
                inputGrad[boxBaseIndex + j] = 0.0;
            }

            inputGrad[confidenceIndex] = 2 * noobjVal * (c1 - 0.0);
        }

        return;
    }

    for (int t = 0; t < YOLOINPUT_GTCOUNT_PER_GRID; t++) {
        // ground truth Box
        int labelBaseIndex = idx * YOLOINPUT_ELEMCOUNT_PER_GRID + 
            t * YOLOINPUT_ELEMCOUNT_PER_GT;
        labelClass = input2[labelBaseIndex + 6];
        labelClassInt = (int)(labelClass + EPSILON);

        if (labelClassInt == 0)
            break;

        Dtype x = input2[labelBaseIndex + 2];
        Dtype y = input2[labelBaseIndex + 3];
        Dtype w = input2[labelBaseIndex + 4];
        Dtype h = input2[labelBaseIndex + 5];

        // anchor boxes
        int bestBoxIndex = 0;
        Dtype bestBoxIOU = 0.0; 

        Dtype box_iou[5];

        for (int i = 0; i < YOLO_ANCHOR_BOX_COUNT; i++) {
            int boxBaseIndex = idx * YOLO_GRID_ELEM_COUNT + i * YOLO_ELEM_COUNT_PER_ANCHORBOX;

            Dtype x1 = input[boxBaseIndex + 0];
            Dtype y1 = input[boxBaseIndex + 1];
            Dtype w1 = input[boxBaseIndex + 2];
            Dtype h1 = input[boxBaseIndex + 3];

            // calc box iou
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
            box_iou[i] = b_inter / b_union;

            if (i == 0) {
                bestBoxIndex = 0;
                bestBoxIOU = box_iou[i];
            } else {
                if (bestBoxIOU < box_iou[i]) {
                    bestBoxIndex = i;
                    bestBoxIOU = box_iou[i];
                }
            }
        }

        // forward boxes & classes
        output[idx] = 0.0;
        for (int i = 0; i < YOLO_ANCHOR_BOX_COUNT; i++) {
            if (bestBoxIndex != i)
                continue;

            int boxBaseIndex = idx * YOLO_GRID_ELEM_COUNT + i * YOLO_ELEM_COUNT_PER_ANCHORBOX;
            Dtype x1 = input[boxBaseIndex + 0];
            Dtype y1 = input[boxBaseIndex + 1];
            Dtype w1 = input[boxBaseIndex + 2];
            Dtype h1 = input[boxBaseIndex + 3];
            Dtype c1 = input[boxBaseIndex + 4];

            inputGrad[boxBaseIndex + 0] = coordVal * (x1 - x) * 2.0;
            inputGrad[boxBaseIndex + 1] = coordVal * (y1 - y) * 2.0;
            inputGrad[boxBaseIndex + 2] = coordVal * 
                (sqrtf(w1 + EPSILON) - sqrt(w + EPSILON)) / sqrtf(w1 + EPSILON);
            inputGrad[boxBaseIndex + 3] = coordVal * 
                (sqrtf(h1 + EPSILON) - sqrt(h + EPSILON)) / sqrtf(h1 + EPSILON);
            inputGrad[boxBaseIndex + 4] = coordVal * (c1 - box_iou[i]) * 2.0;

            for (int j = 0; j < YOLO_CLASS_COUNT; j++) {
                if (j == labelClassInt - 1) {
                    inputGrad[boxBaseIndex + 5 + j] = 
                        classVal * (input[boxBaseIndex + 5 + j] - 1.0) * 2.0;
                } else {
                    inputGrad[boxBaseIndex + 5 + j] = 
                        classVal * (input[boxBaseIndex + 5 + j] - 0.0) * 2.0;
                }
            }

            output[idx] = output[idx] + coordVal * (x1 - x) * (x1 - x);
            output[idx] = output[idx] + coordVal * (y1 - y) * (y1 - y);

            output[idx] = output[idx] + coordVal *
                (sqrtf(w1 + EPSILON) - sqrtf(w + EPSILON)) *
                (sqrtf(w1 + EPSILON) - sqrtf(w + EPSILON));

            output[idx] = output[idx] + coordVal *
                (sqrtf(h1 + EPSILON) - sqrtf(h + EPSILON)) *
                (sqrtf(h1 + EPSILON) - sqrtf(h + EPSILON));

            output[idx] = output[idx] + objVal * (c1 - box_iou[i]) * (c1 - box_iou[i]);

            for (int j = 0; j < YOLO_CLASS_COUNT; j++) {
                if (j == labelClassInt - 1) {
                    output[idx] = output[idx] + classVal * 
                        (input[boxBaseIndex + 5 + j] - 1.0) * 
                        (input[boxBaseIndex + 5 + j] - 1.0);
                } else {
                    output[idx] = output[idx] + classVal *
                        (input[boxBaseIndex + 5 + j] - 0.0) * 
                        (input[boxBaseIndex + 5 + j] - 0.0);
                }
            }
        }
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
    Dtype *inputGrad = this->_inputData[0]->mutable_device_grad();

    YoloForward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, inputData2, size, (Dtype)SLPROP(YOLOLoss, noobj),
        (Dtype)SLPROP(YOLOLoss, coord), (Dtype)SLPROP(YOLOLoss, obj), 
        (Dtype)SLPROP(YOLOLoss, class), outputData, inputGrad);
}

template <typename Dtype>
void YOLOLossLayer<Dtype>::backpropagation() {
    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int size = batchCount * YOLO_GRID_COUNT;

    const Dtype *inputData = this->_inputData[0]->device_data();
    const Dtype *inputData2 = this->_inputData[1]->device_data();
    Dtype *inputGrad = this->_inputData[0]->mutable_device_grad();

    YoloBackward<Dtype><<<SOOOA_GET_BLOCKS(size), SOOOA_CUDA_NUM_THREADS>>>(
        inputData, inputData2, size, (Dtype)SLPROP(YOLOLoss, noobj),
        (Dtype)SLPROP(YOLOLoss, coord), (Dtype)SLPROP(YOLOLoss, obj), 
        (Dtype)SLPROP(YOLOLoss, class), inputGrad);
}

template <typename Dtype>
Dtype YOLOLossLayer<Dtype>::cost() {
    const Dtype* outputData = this->_outputData[0]->host_data();
    Dtype avg = 0.0;

    const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
    int batchCount = inputShape[0];
    int count = YOLO_GRID_COUNT * batchCount;


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
