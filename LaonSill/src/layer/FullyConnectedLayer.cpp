/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"
#include "Util.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "MemoryMgmt.h"

using namespace std;


template<typename Dtype>
FullyConnectedLayer<Dtype>::FullyConnectedLayer() : LearnableLayer<Dtype>() {
	this->type = Layer<Dtype>::FullyConnected;

	const string& name = SLPROP_BASE(name);
	this->_params.resize(2);
	this->_params[ParamType::Weight] = NULL;
	SNEW(this->_params[ParamType::Weight], Data<Dtype>, name + "_weight");
	SASSUME0(this->_params[ParamType::Weight] != NULL);

	this->_params[ParamType::Bias] = NULL;
	SNEW(this->_params[ParamType::Bias], Data<Dtype>, name + "_bias");
	SASSUME0(this->_params[ParamType::Bias] != NULL);

	this->_paramsInitialized.resize(2);
	this->_paramsInitialized[ParamType::Weight] = false;
	this->_paramsInitialized[ParamType::Bias] = false;

	this->_paramsHistory.resize(2);
	this->_paramsHistory[ParamType::Weight] = NULL;
	SNEW(this->_paramsHistory[ParamType::Weight], Data<Dtype>, name + "_weight_history");
	SASSUME0(this->_paramsHistory[ParamType::Weight] != NULL);

	this->_paramsHistory[ParamType::Bias] = NULL;
	SNEW(this->_paramsHistory[ParamType::Bias], Data<Dtype>, name + "_bias_history");
	SASSUME0(this->_paramsHistory[ParamType::Bias] != NULL);

	this->_paramsHistory2.resize(2);
	this->_paramsHistory2[ParamType::Weight] = NULL;
	SNEW(this->_paramsHistory2[ParamType::Weight], Data<Dtype>, name + "_weight_history2");
	SASSUME0(this->_paramsHistory2[ParamType::Weight] != NULL);

	this->_paramsHistory2[ParamType::Bias] = NULL;
	SNEW(this->_paramsHistory2[ParamType::Bias], Data<Dtype>, name + "_bias_history2");
	SASSUME0(this->_paramsHistory2[ParamType::Bias] != NULL);
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* FullyConnectedLayer<Dtype>::initLayer() {
	FullyConnectedLayer* layer = NULL;
	SNEW(layer, FullyConnectedLayer<Dtype>);
	SASSUME0(layer != NULL);
    return (void*)layer;
}

template<typename Dtype>
void FullyConnectedLayer<Dtype>::destroyLayer(void* instancePtr) {
    FullyConnectedLayer<Dtype>* layer = (FullyConnectedLayer<Dtype>*)instancePtr;
    SDELETE(layer);
}

template<typename Dtype>
void FullyConnectedLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 3);
	} else {
		SASSERT0(index < 1);
	}

    FullyConnectedLayer<Dtype>* layer = (FullyConnectedLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool FullyConnectedLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    FullyConnectedLayer<Dtype>* layer = (FullyConnectedLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void FullyConnectedLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    FullyConnectedLayer<Dtype>* layer = (FullyConnectedLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void FullyConnectedLayer<Dtype>::backwardTensor(void* instancePtr) {
    FullyConnectedLayer<Dtype>* layer = (FullyConnectedLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void FullyConnectedLayer<Dtype>::learnTensor(void* instancePtr) {
    FullyConnectedLayer<Dtype>* layer = (FullyConnectedLayer<Dtype>*)instancePtr;
    layer->update();
}

template class FullyConnectedLayer<float>;
