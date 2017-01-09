/*
 * HiddenLayer.cpp
 *
 *  Created on: 2016. 9. 6.
 *      Author: jhkim
 */


#include "HiddenLayer.h"
#include "NetworkConfig.h"

using namespace std;

template <typename Dtype>
HiddenLayer<Dtype>::HiddenLayer() {}

template <typename Dtype>
HiddenLayer<Dtype>::HiddenLayer(Builder* builder)
: Layer<Dtype>(builder) {}

template <typename Dtype>
HiddenLayer<Dtype>::HiddenLayer(const string& name)
: Layer<Dtype>(name) {}


template <typename Dtype>
HiddenLayer<Dtype>::~HiddenLayer() {}


template <typename Dtype>
void HiddenLayer<Dtype>::backpropagation() {
	this->_inputData[0]->set_device_grad(this->_outputData[0]);
}

template <typename Dtype>
void HiddenLayer<Dtype>::reshape() {
	Layer<Dtype>::reshape();
}

template class HiddenLayer<float>;
