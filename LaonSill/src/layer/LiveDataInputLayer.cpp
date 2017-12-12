/**
 * @file LiveDataInputLayer.cpp
 * @date 2017-12-11
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <unistd.h>

#include <vector>

#include "LiveDataInputLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "WorkContext.h"
#include "Param.h"
#include "Perf.h"
#include "MemoryMgmt.h"

using namespace std;


template <typename Dtype>
LiveDataInputLayer<Dtype>::LiveDataInputLayer()
: InputLayer<Dtype>() {
	this->type = Layer<Dtype>::LiveDataInput;
}

template <typename Dtype>
LiveDataInputLayer<Dtype>::~LiveDataInputLayer() {
	// TODO Auto-generated destructor stub
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < SLPROP_BASE(output).size(); i++) {
			SLPROP_BASE(input).push_back(SLPROP_BASE(output)[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	Layer<Dtype>::_adjustInputShape();

    this->_inputShape[0][0] = 1;
    this->_inputShape[0][1] = 3;
    this->_inputShape[0][2] = SLPROP(LiveDataInput, rows);
    this->_inputShape[0][3] = SLPROP(LiveDataInput, cols);
	this->_inputData[0]->reshape(this->_inputShape[0]);

//  int inputImageSize = 3 * SLPROP(LiveDataInput, rows) * SLPROP(LiveDataInput, cols);
//  this->_inputData[0]->set_device_with_host_data(this->images, 0, inputImageSize);
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::feedImage(const int channels, const int height,
		const int width, float* image) {
	SASSERT0(channels == 3);

    SASSERT0(height == SLPROP(LiveDataInput, rows));
    SASSERT0(width == SLPROP(LiveDataInput, cols));
	SASSERT0(image != NULL);

	cv::Mat img(height, width, CV_32FC3, image);
	float* imPtr = (float*)img.data;

    SASSERT0(img.rows == height);
    SASSERT0(img.cols == width);
    SASSERT0(img.channels() == 3);

	int n = img.rows * img.cols * img.channels();
	for (int i = 0; i < n; i+=3) {
		imPtr[i+0] -= SLPROP(LiveDataInput, pixelMeans)[0];
		imPtr[i+1] -= SLPROP(LiveDataInput, pixelMeans)[1];
		imPtr[i+2] -= SLPROP(LiveDataInput, pixelMeans)[2];
	}

	// 'data'
	const vector<uint32_t> inputShape = {1, (uint32_t)img.rows, (uint32_t)img.cols, 3};
	this->_inputData[0]->reshape(inputShape);
	this->_inputData[0]->set_host_data((Dtype*)img.data);

	// Move channels (axis 3) to axis 1
	// Axis order will become: (batch elem, channel, height, width)
	const vector<uint32_t> channelSwap = {0, 3, 1, 2};
	this->_inputData[0]->transpose(channelSwap);
	this->_inputShape[0] = this->_inputData[0]->getShape();
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::feedforward() {
	reshape();
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::feedforward(unsigned int baseIndex, const char* end) {
	reshape();
}

template <typename Dtype>
int LiveDataInputLayer<Dtype>::getNumTrainData() {
    return 1;
}

template <typename Dtype>
int LiveDataInputLayer<Dtype>::getNumTestData() {
	return 0;
}

template <typename Dtype>
void LiveDataInputLayer<Dtype>::shuffleTrainDataSet() {

}

/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* LiveDataInputLayer<Dtype>::initLayer() {
	LiveDataInputLayer* layer = NULL;
	SNEW(layer, LiveDataInputLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void LiveDataInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    LiveDataInputLayer<Dtype>* layer = (LiveDataInputLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void LiveDataInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	// XXX
	if (isInput) {
		SASSERT0(false);
	} else {
		SASSERT0(index < 2);
	}

    LiveDataInputLayer<Dtype>* layer = (LiveDataInputLayer<Dtype>*)instancePtr;
    if (!isInput) {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool LiveDataInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    LiveDataInputLayer<Dtype>* layer = (LiveDataInputLayer<Dtype>*)instancePtr;
    layer->reshape();

    if (SNPROP(miniBatch) == 0) {
		int trainDataNum = layer->getNumTrainData();
		if (trainDataNum % SNPROP(batchSize) == 0) {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize);
		} else {
			SNPROP(miniBatch) = trainDataNum / SNPROP(batchSize) + 1;
		}
		WorkContext::curPlanInfo->miniBatchCount = SNPROP(miniBatch);
	}

    return true;
}

template<typename Dtype>
void LiveDataInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	LiveDataInputLayer<Dtype>* layer = (LiveDataInputLayer<Dtype>*)instancePtr;
	layer->feedforward();

}

template<typename Dtype>
void LiveDataInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    // do nothing
}

template<typename Dtype>
void LiveDataInputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class LiveDataInputLayer<float>;
