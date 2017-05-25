/*
 * DummyInputLayer.cpp
 *
 *  Created on: Jan 21, 2017
 *      Author: jkim
 */

#include "DummyInputLayer.h"
#include "PropMgmt.h"

using namespace std;

template <typename Dtype>
DummyInputLayer<Dtype>::DummyInputLayer(Builder* builder)
: InputLayer<Dtype>(builder) {
	initialize();
}

template<typename Dtype>
DummyInputLayer<Dtype>::DummyInputLayer(const string& name)
: InputLayer<Dtype>(name) {

}

template <typename Dtype>
DummyInputLayer<Dtype>::~DummyInputLayer() {}


template <typename Dtype>
void DummyInputLayer<Dtype>::feedforward() {
	reshape();
}

template <typename Dtype>
void DummyInputLayer<Dtype>::reshape() {
	const vector<string>& outputs = SLPROP(Input, output);
	vector<string>& inputs = SLPROP(Input, input);
	if (inputs.size() < 1) {
		for (uint32_t i = 0; i < outputs.size(); i++) {
			inputs.push_back(outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	Layer<Dtype>::_adjustInputShape();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		this->_inputShape[i] = this->_inputData[i]->getShape();
	}

}

template <typename Dtype>
void DummyInputLayer<Dtype>::initialize() {

}

template<typename Dtype>
int DummyInputLayer<Dtype>::getNumTrainData() {
    return 1;
}

template<typename Dtype>
int DummyInputLayer<Dtype>::getNumTestData() {
    return 1;
}

template<typename Dtype>
void DummyInputLayer<Dtype>::shuffleTrainDataSet() {
}

template class DummyInputLayer<float>;




/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* DummyInputLayer<Dtype>::initLayer() {
    DummyInputLayer* layer = new DummyInputLayer<Dtype>(SLPROP_BASE(name));
    return (void*)layer;
}

template<typename Dtype>
void DummyInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    DummyInputLayer<Dtype>* layer = (DummyInputLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void DummyInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    DummyInputLayer<Dtype>* layer = (DummyInputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool DummyInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    DummyInputLayer<Dtype>* layer = (DummyInputLayer<Dtype>*)instancePtr;
    //layer->reshape();
    return true;
}

template<typename Dtype>
void DummyInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    cout << "DummyInputLayer.. forward(). miniBatchIndex : " << miniBatchIdx << endl;
}

template<typename Dtype>
void DummyInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    cout << "DummyInputLayer.. backward()" << endl;
}

template<typename Dtype>
void DummyInputLayer<Dtype>::learnTensor(void* instancePtr) {
    cout << "DummyInputLayer.. learn()" << endl;
}

template void* DummyInputLayer<float>::initLayer();
template void DummyInputLayer<float>::destroyLayer(void* instancePtr);
template void DummyInputLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool DummyInputLayer<float>::allocLayerTensors(void* instancePtr);
template void DummyInputLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void DummyInputLayer<float>::backwardTensor(void* instancePtr);
template void DummyInputLayer<float>::learnTensor(void* instancePtr);
