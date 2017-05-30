/**
 * @file AAALayer.cpp
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "AAALayer.h"
#include "Util.h"
#include "Exception.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
AAALayer<Dtype>::AAALayer() : Layer<Dtype>() {
    this->type = Layer<float>::AAA;
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* AAALayer<Dtype>::initLayer() {
    AAALayer* layer = new AAALayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void AAALayer<Dtype>::destroyLayer(void* instancePtr) {
    AAALayer<Dtype>* layer = (AAALayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void AAALayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    AAALayer<Dtype>* layer = (AAALayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool AAALayer<Dtype>::allocLayerTensors(void* instancePtr) {
    AAALayer<Dtype>* layer = (AAALayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void AAALayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    AAALayer<Dtype>* layer = (AAALayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void AAALayer<Dtype>::backwardTensor(void* instancePtr) {
    AAALayer<Dtype>* layer = (AAALayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void AAALayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class AAALayer<float>;
