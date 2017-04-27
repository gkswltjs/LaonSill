/*
 * DummyInputLayer.cpp
 *
 *  Created on: Jan 21, 2017
 *      Author: jkim
 */

#include "DummyInputLayer.h"


template <typename Dtype>
DummyInputLayer<Dtype>::DummyInputLayer(Builder* builder)
: InputLayer<Dtype>(builder) {
	initialize();
}

template <typename Dtype>
DummyInputLayer<Dtype>::~DummyInputLayer() {}


template <typename Dtype>
void DummyInputLayer<Dtype>::feedforward() {
	reshape();
}

template <typename Dtype>
void DummyInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();
}

template <typename Dtype>
void DummyInputLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < this->_outputs.size(); i++) {
			this->_inputs.push_back(this->_outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (!adjusted)
		return;

	/*
	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		this->_inputShape[i] = this->_inputData[i]->getShape();
	}
	*/
	this->_inputData[0]->reshape({16, 3, 224, 224});
	this->_inputData[1]->reshape({16, 1, 1, 1});

	this->_inputShape[0] = this->_inputData[0]->getShape();
	this->_inputShape[1] = this->_inputData[1]->getShape();

}

template <typename Dtype>
void DummyInputLayer<Dtype>::initialize() {

}

template<typename Dtype>
int DummyInputLayer<Dtype>::getNumTrainData() {
    return 100000;
}

template<typename Dtype>
int DummyInputLayer<Dtype>::getNumTestData() {
    return 1;
}

template<typename Dtype>
void DummyInputLayer<Dtype>::shuffleTrainDataSet() {
}

template class DummyInputLayer<float>;
