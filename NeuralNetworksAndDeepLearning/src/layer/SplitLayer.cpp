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

	/*
	this->setInDimension(this->_inputData[0]->getShape());
	this->out_dim = this->in_dim;

	// 일단 SplitLayer는 forward pass에서
	// 하나의 output이 여러 input으로 전달되는 경우를 담당한다고 전제
	// 입력은 항상 1개로 상정

	//for (uint32_t i = 1; i < this->_inputs.size(); i++) {
	//	if (this->_inputData[i]->getCount() == 0)
	//		this->_inputData[i]->reshape({this->in_dim.batches, this->in_dim.channels,
	//		this->in_dim.rows, this->in_dim.cols});
	//}

	// 모든 output data에 대해 shape처리한다.
	//	ㄴLayer의 _shape()가 호출되어도 이미지 shape처리되었기 때문에 무시된다.
	for (uint32_t i = 0; i < this->_outputs.size(); i++) {
		if (this->_outputData[i]->getCount() == 0)
			this->_outputData[i]->reshape({this->out_dim.batches, this->out_dim.channels,
			this->out_dim.rows, this->out_dim.cols});
	}

	if(recursive) {
		Layer<Dtype>::_shape();
	}
	*/
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

#if SPLITLAYER_LOG
	const string targetLayer = "rpn/output-split";
#endif

	this->_inputData[0]->reset_device_grad();
	for (uint32_t i = 0; i < this->_outputData.size(); i++) {

#if SPLITLAYER_LOG
		if (this->name == targetLayer) {
			Data<Dtype>::printConfig = true;
			this->_outputData[i]->print_grad({}, false);
			Data<Dtype>::printConfig = false;
		}
#endif
		this->_inputData[0]->add_device_grad(this->_outputData[i]);
#if SPLITLAYER_LOG
		if (this->name == targetLayer) {
			Data<Dtype>::printConfig = true;
			this->_inputData[0]->print_grad({}, false);
			Data<Dtype>::printConfig = false;
		}
#endif
	}
#if SPLITLAYER_LOG
	if (this->name == targetLayer) {
		exit(1);
	}
#endif
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* SplitLayer<Dtype>::initLayer() {
    SplitLayer* layer = new SplitLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void SplitLayer<Dtype>::destroyLayer(void* instancePtr) {
    SplitLayer<Dtype>* layer = (SplitLayer<Dtype>*)instancePtr;
    delete layer;
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
