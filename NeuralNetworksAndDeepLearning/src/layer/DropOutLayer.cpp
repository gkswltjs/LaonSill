/**
 * @file DropOutLayer.cpp
 * @date 2017-04-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "DropOutLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
DropOutLayer<Dtype>::DropOutLayer(Builder* builder) : Layer<Dtype>(builder) {
	initialize(builder->_scale, builder->_probability);
}

template<typename Dtype>
DropOutLayer<Dtype>::DropOutLayer(const string& name) 
: Layer<Dtype>(name) {
	initialize(SLPROP(DropOut, scale), SLPROP(DropOut, probability));
}

template <typename Dtype>
DropOutLayer<Dtype>::~DropOutLayer() {
}

template <typename Dtype>
void DropOutLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputShape;

	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	this->_outputData[0]->reshape(inputShape);
    this->mask->reshape(batches * channels * rows * cols);
}

template <typename Dtype>
void DropOutLayer<Dtype>::feedforward() {
	reshape();
    doDropOutForward();
}

template <typename Dtype>
void DropOutLayer<Dtype>::backpropagation() {
    doDropOutBackward();
}

template <typename Dtype>
void DropOutLayer<Dtype>::initialize(double scale, double probability) {
	this->type = Layer<Dtype>::DropOut;
    this->scale = scale;
    this->probability = probability;

    shared_ptr<SyncMem<Dtype>> tempMask(new SyncMem<Dtype>());
    this->mask = tempMask;
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* DropOutLayer<Dtype>::initLayer() {
    DropOutLayer* layer = new DropOutLayer<Dtype>(SLPROP_BASE(name));
    return (void*)layer;
}

template<typename Dtype>
void DropOutLayer<Dtype>::destroyLayer(void* instancePtr) {
    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void DropOutLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DropOutLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void DropOutLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void DropOutLayer<Dtype>::backwardTensor(void* instancePtr) {
    DropOutLayer<Dtype>* layer = (DropOutLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void DropOutLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class DropOutLayer<float>;
