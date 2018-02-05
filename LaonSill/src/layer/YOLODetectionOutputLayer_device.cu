/**
 * @file YOLODetectionOutputLayer_device.cu
 * @date 2018-01-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "YOLODetectionOutputLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"
#include "YOLOLossLayer.h"

using namespace std;

#define EPSILON                 0.000001
#define YOLODETOUT_DET_ELEM_COUNT   (7)
#define YOLODETOUT_GT_ELEM_COUNT    (8)

// FIXME: YOLO Detection Output 레이어는 DetectionEvaluatedLayer의 입력을 맞추기 위한
//        레이어이다. 만약 DetectionEvaluatedLayer input tensor의 포맷이 변경이 된다면
//        그에따라서 YOLO Detection Output Layer도 변경 되어야 한다.
/**************************************************************************************
 * YOLO detection layer's output tensor
 *
 *  (1) output[0] => detection tensor
 *  +------------------+-------+-------+------+------+------+------+
 *  | itemID(batchIdx) | label | score | xmin | ymin | xmax | ymax | -> batch0, grid(0,0)
 *  +------------------+-------+-------+------+------+------+------+    anchor#0
 *  | itemID(batchIdx) | label | score | xmin | ymin | xmax | ymax | -> batch0, grid(0,0)
 *  +------------------+-------+-------+------+------+------+------+    anchor#1
 *  |                          ...                                 |
 *  +------------------+-------+-------+------+------+------+------+
 *  | itemID(batchIdx) | label | score | xmin | ymin | xmax | ymax | -> batch7, grid(12,12)
 *  +------------------+-------+-------+------+------+------+------+    anchor#4
 *
 *  (2) output[1] => ground truth tensor
 *  +--------+-------+-------+------+------+------+------+-----------+
 *  | itemID | label | score | xmin | ymin | xmax | ymax | difficult | -> batch0, grid(0,0)
 *  +--------+-------+-------+------+------+------+------+-----------+    anchor#0
 *  | itemID | label | score | xmin | ymin | xmax | ymax | difficult | -> batch0, grid(0,0)
 *  +--------+-------+-------+------+------+------+------+-----------+    anchor#1
 *  |                ...                                             |
 *  +--------+-------+-------+------+------+------+------+-----------+
 *  | itemID | label | score | xmin | ymin | xmax | ymax | difficult | -> batch7, grid(12,12)
 *  +--------+-------+-------+------+------+------+------+-----------+    anchor#4
 *
 * */

template <typename Dtype>
__global__ void YOLODetectionForwardDetectionTensor(const Dtype* input, int size, 
        Dtype scoreThres, int classNum, Dtype* output) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    int elemPerAnchorBox = classNum + 4;

    Dtype c = input[idx * elemPerAnchorBox + 4];

    if (c < scoreThres) {
        output[idx * YOLODETOUT_DET_ELEM_COUNT + 0] = -1.0 - EPSILON;
        output[idx * YOLODETOUT_DET_ELEM_COUNT + 1] = -1.0 - EPSILON;
        return;
    }

    Dtype maxValue = input[idx * elemPerAnchorBox + 5 + 0];
    int maxValueIdx = 0;

    for (int i = 1; i < classNum; i++) {
        if (maxValue < input[idx * elemPerAnchorBox + 5 + i]) {
            maxValue = input[idx * elemPerAnchorBox + 5 + i];
            maxValueIdx = i;
        }
    }

    Dtype label = (Dtype)maxValueIdx + EPSILON;
    Dtype score = c * maxValue;
    if (score < scoreThres) {
        output[idx * YOLODETOUT_DET_ELEM_COUNT + 0] = -1.0 - EPSILON;
        output[idx * YOLODETOUT_DET_ELEM_COUNT + 1] = -1.0 - EPSILON;
        return;
    }

    int curBatch = idx / (YOLO_GRID_COUNT * YOLO_ANCHOR_BOX_COUNT);
    Dtype itemID = (Dtype)curBatch + EPSILON;
    int gridIdx = (idx % (YOLO_GRID_COUNT * YOLO_ANCHOR_BOX_COUNT)) / YOLO_ANCHOR_BOX_COUNT;
    int gridX = gridIdx % YOLO_GRID_ONE_AXIS_COUNT;
    int gridY = gridIdx / YOLO_GRID_ONE_AXIS_COUNT;

    Dtype x = input[idx * elemPerAnchorBox + 0];
    Dtype y = input[idx * elemPerAnchorBox + 1];
    Dtype w = input[idx * elemPerAnchorBox + 2];
    Dtype h = input[idx * elemPerAnchorBox + 3];

    Dtype minX = (x + (Dtype)gridX) / (Dtype)YOLO_GRID_ONE_AXIS_COUNT - w / 2.0;
    Dtype maxX = (x + (Dtype)gridX) / (Dtype)YOLO_GRID_ONE_AXIS_COUNT + w / 2.0;
    Dtype minY = (y + (Dtype)gridY) / (Dtype)YOLO_GRID_ONE_AXIS_COUNT - h / 2.0;
    Dtype maxY = (y + (Dtype)gridY) / (Dtype)YOLO_GRID_ONE_AXIS_COUNT + h / 2.0;

    output[idx * YOLODETOUT_DET_ELEM_COUNT + 0] = itemID;
    output[idx * YOLODETOUT_DET_ELEM_COUNT + 1] = label + 1.0;  // background : 0
    output[idx * YOLODETOUT_DET_ELEM_COUNT + 2] = score;
    output[idx * YOLODETOUT_DET_ELEM_COUNT + 3] = minX;
    output[idx * YOLODETOUT_DET_ELEM_COUNT + 4] = minY;
    output[idx * YOLODETOUT_DET_ELEM_COUNT + 5] = maxX;
    output[idx * YOLODETOUT_DET_ELEM_COUNT + 6] = maxY;
}

template <typename Dtype>
__global__ void YOLODetectionForwardGroundTruthTensor(const Dtype* input, int size, 
        Dtype* output) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

    Dtype gridX = input[idx * YOLOINPUT_ELEMCOUNT_PER_GT + 0];
    Dtype gridY = input[idx * YOLOINPUT_ELEMCOUNT_PER_GT + 1];
    Dtype x = input[idx * YOLOINPUT_ELEMCOUNT_PER_GT + 2];
    Dtype y = input[idx * YOLOINPUT_ELEMCOUNT_PER_GT + 3];
    Dtype w = input[idx * YOLOINPUT_ELEMCOUNT_PER_GT + 4];
    Dtype h = input[idx * YOLOINPUT_ELEMCOUNT_PER_GT + 5];
    Dtype class_id = input[idx * YOLOINPUT_ELEMCOUNT_PER_GT + 6];

    int label = (int)(class_id + EPSILON);
    if (label == 0) {
        output[idx * YOLODETOUT_GT_ELEM_COUNT + 0] = -1.0 - EPSILON;
        output[idx * YOLODETOUT_GT_ELEM_COUNT + 1] = -1.0 - EPSILON;
        return;
    }
    
    Dtype minX = (x + gridX) / (Dtype)YOLO_GRID_ONE_AXIS_COUNT - w / 2.0;
    Dtype maxX = (x + gridX) / (Dtype)YOLO_GRID_ONE_AXIS_COUNT + w / 2.0;
    Dtype minY = (y + gridY) / (Dtype)YOLO_GRID_ONE_AXIS_COUNT - h / 2.0;
    Dtype maxY = (y + gridY) / (Dtype)YOLO_GRID_ONE_AXIS_COUNT + h / 2.0;

    int curBatch = idx / (YOLO_GRID_COUNT * YOLOINPUT_GTCOUNT_PER_GRID);
    Dtype itemID = (Dtype)curBatch + EPSILON;

    output[idx * YOLODETOUT_GT_ELEM_COUNT + 0] = itemID;
    output[idx * YOLODETOUT_GT_ELEM_COUNT + 1] = class_id + EPSILON;
    output[idx * YOLODETOUT_GT_ELEM_COUNT + 2] = 1.0;       // score.. meaningless
    output[idx * YOLODETOUT_GT_ELEM_COUNT + 3] = minX;
    output[idx * YOLODETOUT_GT_ELEM_COUNT + 4] = minY;
    output[idx * YOLODETOUT_GT_ELEM_COUNT + 5] = maxX;
    output[idx * YOLODETOUT_GT_ELEM_COUNT + 6] = maxY;
    output[idx * YOLODETOUT_GT_ELEM_COUNT + 7] = 0.0 + EPSILON;     // difficult.. meaningless
}

template <typename Dtype>
void YOLODetectionOutputLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

    // FIXME: 필요한 데이터만 넣게되면 shape을 줄일 수 있다. 
    //        하지만, 병렬로 돌리기 위해서는 shape를 고정시킬 필요가 있다.
    //        줄였을때에 큰 이득이 있는지 생각해보고 합리적이면 추후에 줄이자.
    //
	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];

    uint32_t detectionCount = (uint32_t)(YOLO_ANCHOR_BOX_COUNT * YOLO_GRID_COUNT * batches);
    this->_outputData[0]->reshape({1U, 1U, detectionCount, 
            (uint32_t)YOLODETOUT_DET_ELEM_COUNT});
    int totalElemCount = this->_inputData[0]->getCount();
    int elemPerAnchorBox = (int)(SLPROP(YOLODetectionOutput, numClasses) + 4);
    SASSUME0(totalElemCount == detectionCount * elemPerAnchorBox);

	const vector<uint32_t>& inputShape2 = this->_inputData[1]->getShape();
	uint32_t batches2 	= inputShape2[0];

    uint32_t gtCount = (uint32_t)(YOLOINPUT_GTCOUNT_PER_GRID * YOLO_GRID_COUNT * batches2);
    this->_outputData[1]->reshape({1U, 1U, gtCount, (uint32_t)YOLODETOUT_GT_ELEM_COUNT});
    totalElemCount = this->_inputData[1]->getCount();
    SASSUME0(totalElemCount == gtCount * YOLOINPUT_ELEMCOUNT_PER_GT);

    SASSUME0(batches == batches2);
}

template <typename Dtype>
void YOLODetectionOutputLayer<Dtype>::feedforward() {
	reshape();

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];

    int detectionCount = YOLO_ANCHOR_BOX_COUNT * YOLO_GRID_COUNT * batches;
    int gtCount = YOLOINPUT_GTCOUNT_PER_GRID * YOLO_GRID_COUNT * batches;

    // (1) fill detection tensor
    const Dtype *inputData = this->_inputData[0]->device_data();
    Dtype *outputData = this->_outputData[0]->mutable_device_data();

    YOLODetectionForwardDetectionTensor<Dtype><<<SOOOA_GET_BLOCKS(detectionCount), 
        SOOOA_CUDA_NUM_THREADS>>>(inputData, detectionCount,
                (Dtype)SLPROP(YOLODetectionOutput, scoreThres),
                (int)SLPROP(YOLODetectionOutput, numClasses), outputData);

    // (2) fill ground truth tensor
    const Dtype *inputData2 = this->_inputData[1]->device_data();
    Dtype *outputData2 = this->_outputData[1]->mutable_device_data();

    YOLODetectionForwardGroundTruthTensor<Dtype><<<SOOOA_GET_BLOCKS(gtCount), 
        SOOOA_CUDA_NUM_THREADS>>>(inputData2, gtCount, outputData2);
}

template <typename Dtype>
void YOLODetectionOutputLayer<Dtype>::backpropagation() {
}

template void YOLODetectionOutputLayer<float>::reshape();
template void YOLODetectionOutputLayer<float>::feedforward();
template void YOLODetectionOutputLayer<float>::backpropagation();

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* YOLODetectionOutputLayer<Dtype>::initLayer() {
	YOLODetectionOutputLayer* layer = NULL;
	SNEW(layer, YOLODetectionOutputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void YOLODetectionOutputLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLODetectionOutputLayer<Dtype>* layer = (YOLODetectionOutputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLODetectionOutputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

    YOLODetectionOutputLayer<Dtype>* layer = (YOLODetectionOutputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() <= 1);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() <= 1);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool YOLODetectionOutputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLODetectionOutputLayer<Dtype>* layer = (YOLODetectionOutputLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void YOLODetectionOutputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	YOLODetectionOutputLayer<Dtype>* layer = (YOLODetectionOutputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void YOLODetectionOutputLayer<Dtype>::backwardTensor(void* instancePtr) {
	YOLODetectionOutputLayer<Dtype>* layer = (YOLODetectionOutputLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void YOLODetectionOutputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template void* YOLODetectionOutputLayer<float>::initLayer();
template void YOLODetectionOutputLayer<float>::destroyLayer(void* instancePtr);
template void YOLODetectionOutputLayer<float>::setInOutTensor(void* instancePtr,
        void* tensorPtr, bool isInput, int index);
template bool YOLODetectionOutputLayer<float>::allocLayerTensors(void* instancePtr);
template void YOLODetectionOutputLayer<float>::forwardTensor(void* instancePtr,
        int miniBatchIdx);
template void YOLODetectionOutputLayer<float>::backwardTensor(void* instancePtr);
template void YOLODetectionOutputLayer<float>::learnTensor(void* instancePtr);
