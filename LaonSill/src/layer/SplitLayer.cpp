/*
 * SplitLayer.cpp
 *
 *  Created on: Nov 8, 2016
 *      Author: jkim
 */

#include <vector>

#include "SplitLayer.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MathFunctions.h"
#include "MemoryMgmt.h"

#define SPLITLAYER_LOG 0

using namespace std;

template <typename Dtype>
SplitLayer<Dtype>::SplitLayer()
	: Layer<Dtype>() {
	this->type = Layer<Dtype>::Split;
}


template <typename Dtype>
SplitLayer<Dtype>::~SplitLayer() {

}


template <typename Dtype>
void SplitLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	this->_inputShape[0] = inputShape;

	for (uint32_t i = 0; i < this->_outputData.size(); i++) {
		this->_outputData[i]->reshape(inputShape);
	}
}

template <typename Dtype>
void SplitLayer<Dtype>::feedforward() {
	reshape();

	for (uint32_t i = 0; i < this->_outputData.size(); i++) {
		this->_outputData[i]->set_device_data(this->_inputData[0]);
	}
}

template <typename Dtype>
void SplitLayer<Dtype>::backpropagation() {
	const int count = this->_inputData[0]->getCount();
	if (this->_outputData.size() == 1) {
		soooa_copy(count, this->_outputData[0]->device_grad(),
				this->_inputData[0]->mutable_device_grad());
		return;
	}
	soooa_gpu_add(count, this->_outputData[0]->device_grad(),
			this->_outputData[1]->device_grad(), this->_inputData[0]->mutable_device_grad());

	// Add remaining top blobs diffs.
	for (int i = 2; i < this->_outputData.size(); i++) {
		const Dtype* outputGrad = this->_outputData[i]->device_grad();
		Dtype* inputGrad = this->_inputData[0]->mutable_device_grad();
		soooa_gpu_axpy(count, Dtype(1.), outputGrad, inputGrad);
	}
}


/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* SplitLayer<Dtype>::initLayer() {
	SplitLayer* layer = NULL;
	SNEW(layer, SplitLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void SplitLayer<Dtype>::destroyLayer(void* instancePtr) {
    SplitLayer<Dtype>* layer = (SplitLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void SplitLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

    SplitLayer<Dtype>* layer = (SplitLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        SASSERT0(index == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool SplitLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    SplitLayer<Dtype>* layer = (SplitLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void SplitLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	SplitLayer<Dtype>* layer = (SplitLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void SplitLayer<Dtype>::backwardTensor(void* instancePtr) {
	SplitLayer<Dtype>* layer = (SplitLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void SplitLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class SplitLayer<float>;
