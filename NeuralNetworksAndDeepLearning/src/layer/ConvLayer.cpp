/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "Util.h"
#include "SysLog.h"
#include "PropMgmt.h"

using namespace std;

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* ConvLayer<Dtype>::initLayer() {
    ConvLayer* layer = new ConvLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void ConvLayer<Dtype>::destroyLayer(void* instancePtr) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void ConvLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ConvLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void ConvLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void ConvLayer<Dtype>::backwardTensor(void* instancePtr) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void ConvLayer<Dtype>::learnTensor(void* instancePtr) {
    ConvLayer<Dtype>* layer = (ConvLayer<Dtype>*)instancePtr;
    layer->update();
}

template class ConvLayer<float>;
