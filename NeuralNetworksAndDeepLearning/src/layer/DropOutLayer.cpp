/**
 * @file DropOutLayer.cpp
 * @date 2017-04-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "DropOutLayer.h"

using namespace std;

template <typename Dtype>
DropOutLayer<Dtype>::DropOutLayer(Builder* builder) : Layer<Dtype>(builder) {
	initialize(builder->_scale, builder->_probability);
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

template class DropOutLayer<float>;
