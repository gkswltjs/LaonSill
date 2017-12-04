/*
 * DummyLossLayer.cpp
 *
 *  Created on: Sep 7, 2017
 *      Author: jkim
 */

#include "DummyLossLayer.h"

template <typename Dtype>
DummyLossLayer<Dtype>::DummyLossLayer()
: LossLayer<Dtype>() {
	this->type = Layer<Dtype>::DummyLoss;
}

template <typename Dtype>
DummyLossLayer<Dtype>::~DummyLossLayer() {}

template <typename Dtype>
void DummyLossLayer<Dtype>::reshape() {}

template <typename Dtype>
void DummyLossLayer<Dtype>::feedforward() {}

template <typename Dtype>
void DummyLossLayer<Dtype>::backpropagation() {}

template <typename Dtype>
Dtype DummyLossLayer<Dtype>::cost() {
	return Dtype(1.0);
}



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* DummyLossLayer<Dtype>::initLayer() {
    DummyLossLayer* layer = new DummyLossLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void DummyLossLayer<Dtype>::destroyLayer(void* instancePtr) {
    DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void DummyLossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DummyLossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void DummyLossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void DummyLossLayer<Dtype>::backwardTensor(void* instancePtr) {
	DummyLossLayer<Dtype>* layer = (DummyLossLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void DummyLossLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}



template class DummyLossLayer<float>;


