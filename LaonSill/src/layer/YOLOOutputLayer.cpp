/*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 | @ file      YOLOOutputLayer.cpp
 * @ date      2018-02-06
 | @ author    SUN
 * @ brief     
 | @ details   
 *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/

#include <vector>
#include <array>

#include "YOLOOutputLayer.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"
#include "ImageUtil.h"
#include "YOLOLossLayer.h"

#include "frcnn_common.h"

using namespace std;

/*******************************************************************************************
 * YOLO Output layer : YOLO 커스텀 데이터 inference 지원을 위한 레이어.
 *
 *  << output tensor format >>
 *  +---------+-------+-------+------+------+------+------+
 *  | padding | label | score | xmin | ymin | xmax | ymax |
 *  +---------+-------+-------+------+------+------+------+
 *  | padding | label | score | xmin | ymin | xmax | ymax |
 *  +---------+-------+-------+------+------+------+------+
 *  |                        ...                          |
 *  +---------+-------+-------+------+------+------+------+
 *  | padding | label | score | xmin | ymin | xmax | ymax |
 *  +---------+-------+-------+------+------+------+------+
 *
 *******************************************************************************************/

template <typename Dtype>
YOLOOutputLayer<Dtype>::YOLOOutputLayer()
: Layer<Dtype>() {
    this->type = Layer<Dtype>::YOLOOutput;
}

template <typename Dtype>
YOLOOutputLayer<Dtype>::~YOLOOutputLayer() {
}

template class YOLOOutputLayer<float>;

template <typename Dtype>
void YOLOOutputLayer<Dtype>::YOLOOutputForward(const Dtype* inputData,
        const int classNum) {

    int side = YOLO_GRID_ONE_AXIS_COUNT;
    int gridCell = YOLO_GRID_COUNT;

    int anchorBox = YOLO_ANCHOR_BOX_COUNT;
    int elemPerAnchorBox = classNum + 4;
    int gridElemCount = anchorBox * elemPerAnchorBox;
    float confThresh = YOLO_DEFAULT_CONFIDENCE_THRES;

    int imageWidth = YOLO_IMAGE_DEFAULT_WIDTH;
    int imageHeight = YOLO_IMAGE_DEFAULT_HEIGHT;

    int resultCount = 0;
    float left, top, right, bottom;

    vector<vector<Dtype>> output;

    for (int i = 0; i < gridCell; i++){
        int gridX = i % side;
        int gridY = i / side;

        for (int j = 0; j < anchorBox; j++){
            int bboxesIdx = i * gridElemCount + j * elemPerAnchorBox;
            float x = inputData[bboxesIdx + 0];
            float y = inputData[bboxesIdx + 1];
            float w = inputData[bboxesIdx + 2];
            float h = inputData[bboxesIdx + 3];
            float c = inputData[bboxesIdx + 4];

            float maxConfidence = inputData[bboxesIdx + 5];
            int labelIdx = 0;

            for (int k = 1; k < classNum - 1; k++){
                if (maxConfidence < inputData[bboxesIdx + 5 + k]){
                    labelIdx = k;
                    maxConfidence = inputData[bboxesIdx + 5 + k];
                }
            }

            float score = c * maxConfidence;

            if (score <= confThresh) {
                continue;
            }

            resultCount++;

            top = (float)((((float)gridY + y) / (float)side - 0.5 * h) *
                (float)imageHeight);
            bottom = (float)((((float)gridY + y) / (float)side + 0.5 * h) *
                (float)imageHeight);
            left = (float)((((float)gridX + x) / (float)side - 0.5 * w) *
                (float)imageWidth);
            right = (float)((((float)gridX + x) / (float)side + 0.5 * w) *
                (float)imageWidth);

            vector<Dtype> bbox(6);
            bbox[0] = score;
            bbox[1] = labelIdx;
            bbox[2] = left;
            bbox[3] = top;
            bbox[4] = right;
            bbox[5] = bottom;

            output.push_back(bbox);
        }

        // NMS ***
        float nmsThresh = YOLO_DEFAULT_NMS_THRES;
        vector<vector<Dtype>> result;

        for (int i = 0; i < classNum - 1; i++) {
            vector<uint32_t> keep;
            vector<vector<float>> bboxes;
            vector<float> scores;

            for (int j = 0; j < resultCount; j++) {
                if (output[j][1] != i) {
                    continue;
                }
                vector<float> coord = {output[j][2], output[j][3],
                   output[j][4], output[j][5]};
                bboxes.push_back(coord);
                scores.push_back(output[j][0]);
            }

            if (bboxes.size() == 0)
                continue;

            nms(bboxes, scores, nmsThresh, keep);

            for (int k = 0; k < keep.size(); k++) {
                vector<Dtype> pred(7);
                pred[0] = 0.f; // frcnn output과 동일하게 맞추기 위함.
                pred[1] = float(i);
                pred[2] = scores[keep[k]];
                pred[3] = bboxes[keep[k]][0];
                pred[4] = bboxes[keep[k]][1];
                pred[5] = bboxes[keep[k]][2];
                pred[6] = bboxes[keep[k]][3];

                result.push_back(pred);
            }

        }

        if (result.size() > 0) {
            fillDataWith2dVec(result, this->_outputData[0]);
        } else {
            this->_outputData[0]->reshape({1, 1, 1, 7});
            this->_outputData[0]->mutable_host_data()[1] = -1;
        }
    }
}

template <typename Dtype>
void YOLOOutputLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

}

template <typename Dtype>
void YOLOOutputLayer<Dtype>::feedforward(){
    reshape();

    const Dtype* inputData = this->_inputData[0]->host_data();

    YOLOOutputForward(inputData, (int)SLPROP(YOLOOutput, numClasses));

}

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* YOLOOutputLayer<Dtype>::initLayer() {
	YOLOOutputLayer* layer = NULL;
	SNEW(layer, YOLOOutputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::destroyLayer(void* instancePtr) {
    YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool YOLOOutputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::backwardTensor(void* instancePtr) {
	YOLOOutputLayer<Dtype>* layer = (YOLOOutputLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void YOLOOutputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template void* YOLOOutputLayer<float>::initLayer();
template void YOLOOutputLayer<float>::destroyLayer(void* instancePtr);
template void YOLOOutputLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool YOLOOutputLayer<float>::allocLayerTensors(void* instancePtr);
template void YOLOOutputLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void YOLOOutputLayer<float>::backwardTensor(void* instancePtr);
template void YOLOOutputLayer<float>::learnTensor(void* instancePtr);
