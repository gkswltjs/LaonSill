/*
 * ReshapeLayer.cpp
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */


#if 0
#include "ReshapeLayer.h"

template <typename Dtype>
ReshapeLayer<Dtype>::ReshapeLayer() {
	// TODO Auto-generated constructor stub
}

template <typename Dtype>
ReshapeLayer<Dtype>::ReshapeLayer(Builder* builder) {
	this->shape = builder->_shape;

	initialize();
}

template <typename Dtype>
ReshapeLayer<Dtype>::~ReshapeLayer() {

}





template <typename Dtype>
void ReshapeLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	// Layer does not allow in-place computation.
	assert(this->_inputData[0] != this->_outputData[0]);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::feedforward() {
	reshape();

	this->_inputShape[0] = this->_inputData[0]->getShape();
	vector<uint32_t> outputShape(this->shape.size());

	int inferredAxis = -1;
	for (uint32_t i = 0; i < this->shape.size(); i++) {
		if (this->shape[i] == 0) {
			outputShape[i] = this->_inputShape[0][i];
		} else if (this->shape[i] == -1) {
			inferredAxis = i;
		} else {
			outputShape[i] = this->shape[i];
		}
	}

	// there is! inferred axis ...
	if (inferredAxis >= 0) {
		const uint32_t inputShapeCount = Util::vecCountByAxis(this->_inputShape[0], 0);
		const uint32_t outputShapeCount = Util::vecCountByAxis(outputShape);
		assert(inputShapeCount % outputShapeCount == 0);

		const uint32_t inferredAxisShape = inputShapeCount / outputShapeCount;
		outputShape[inferredAxis] = inferredAxisShape;
	}

	this->_outputData[0]->shape(outputShape);

	this->_outputData[0]->set_device_data(this->_inputData[0]);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::backpropagation() {

}



template <typename Dtype>
void ReshapeLayer<Dtype>::initialize() {
	assert(this->shape.size() == 4);

	int inferredAxis = -1;
	uint32_t numInferredAxis = 0;

	const uint32_t topNumAxis = this->shape.size();

	for (uint32_t i = 0; i < topNumAxis; i++) {
		if (this->shape[0] == -1) {
			numInferredAxis++;
			if (numInferredAxis > 1) {
				cout << "more than 1 inferred axes exist ... " << endl;
				exit(1);
			}
		}
	}
}




template class ReshapeLayer<float>;




#endif


































