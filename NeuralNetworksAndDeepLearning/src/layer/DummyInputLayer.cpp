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
void DummyInputLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < this->_outputs.size(); i++) {
			this->_inputs.push_back(this->_outputs[i]);
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

template class DummyInputLayer<float>;
