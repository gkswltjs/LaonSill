/*
 * ReshapeLayer.cpp
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */


#if 1

#include <vector>

#include "ReshapeLayer.h"
#include "Util.h"

#define RESHAPELAYER_LOG 0

using namespace std;

template <typename Dtype>
ReshapeLayer<Dtype>::ReshapeLayer(Builder* builder)
	: Layer<Dtype>(builder) {
	this->shape = builder->_shape;
	this->axis = builder->_axis;
	this->numAxes = builder->_numAxes;
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


	const vector<uint32_t>& inputDataShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputDataShape;

	const uint32_t dim = inputDataShape.size();
	vector<uint32_t> outputDataShape(dim);

	for (uint32_t i = 0; i < dim; i++) {
		if (this->shape[i] > 0)
			outputDataShape[i] = this->shape[i];
	}
	for (uint32_t i = 0; i < copyAxes.size(); i++) {
		outputDataShape[copyAxes[i]] = inputDataShape[copyAxes[i]];
	}

	if (this->inferredAxis >= 0) {
		const uint32_t inputDataSize = this->_inputData[0]->getCount();
		uint32_t fixedSize = 1;
		for (uint32_t i = 0; i < dim; i++) {
			if (outputDataShape[i] > 0)
				fixedSize *= outputDataShape[i];
		}
		assert(inputDataSize % fixedSize == 0 &&
				"input count must be divisible by the product");
		outputDataShape[inferredAxis] = inputDataSize / fixedSize;
	}

	this->_outputData[0]->reshape(outputDataShape);

#if RESHAPELAYER_LOG
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			this->name.c_str(), outputDataShape[0], outputDataShape[1],
			outputDataShape[2], outputDataShape[3]);
#endif


	assert(this->_inputData[0]->getCount() == this->_outputData[0]->getCount() &&
			"output count must match input count");

	this->_outputData[0]->share_data(this->_inputData[0]);
	this->_outputData[0]->share_grad(this->_inputData[0]);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::feedforward() {
	reshape();

#if RESHAPELAYER_LOG
	Data<Dtype>::printConfig = true;
	const vector<uint32_t> shape;
	this->_inputData[0]->print_data(shape, false);
	this->_inputData[0]->print_grad(shape, false);
	this->_outputData[0]->print_data(shape, false);
	this->_outputData[0]->print_grad(shape, false);
	Data<Dtype>::printConfig = false;
#endif

}

template <typename Dtype>
void ReshapeLayer<Dtype>::backpropagation() {
	// do nothing ...
}



template <typename Dtype>
void ReshapeLayer<Dtype>::initialize() {
	assert(this->shape.size() == 4);

	this->inferredAxis = -1;
	this->copyAxes.clear();
	const uint32_t topNumAxis = this->shape.size();
	this->constantCount = 1;

	for (uint32_t i = 0; i < topNumAxis; i++) {
		const int topDim = this->shape[i];
		if (topDim == 0) {
			copyAxes.push_back(i);
		} else if (topDim == -1) {
			assert(inferredAxis == -1 &&
					"new shape contains multiple -1 dims ... ");
			inferredAxis = i;
		} else {
			constantCount *= topDim;
		}
	}
}




template class ReshapeLayer<float>;




#endif
