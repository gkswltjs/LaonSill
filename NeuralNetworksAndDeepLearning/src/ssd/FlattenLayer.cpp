/*
 * FlattenLayer.cpp
 *
 *  Created on: Apr 22, 2017
 *      Author: jkim
 */

#include "FlattenLayer.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
FlattenLayer<Dtype>::FlattenLayer(Builder* builder)
: Layer<Dtype>(builder),
  axis(builder->_axis) {

	initialize();
}

template <typename Dtype>
FlattenLayer<Dtype>::~FlattenLayer() {

}

template <typename Dtype>
void FlattenLayer<Dtype>::reshape() {
	SASSERT0(this->_inputs.size() == 1 && this->_outputs.size() == 1);
	SASSERT(this->_inputs[0] != this->_outputs[0],
			"this layer does not allow in-place computation.");

	Layer<Dtype>::_adjustInputShape();
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	// TODO: Data에 CanonicalAxis 구현 필요
	const int startAxis = this->axis;
	const int endAxis = this->_inputData[0]->getShape().size();

	vector<uint32_t> outputShape;
	for (int i = 0; i < startAxis; i++) {
		outputShape.push_back(this->_inputData[0]->getShape(i));
	}
	const int flattenedDim = this->_inputData[0]->getCountByAxis(startAxis, endAxis + 1);
	outputShape.push_back(flattenedDim);
	for (int i = endAxis + 1; i < this->_inputData[0]->getShape().size(); i++) {
		outputShape.push_back(this->_inputData[0]->getShape(i));
	}
	this->_outputData[0]->reshape(outputShape);
	SASSERT0(this->_outputData[0]->getCount() == this->_inputData[0]->getCount());
}

template <typename Dtype>
void FlattenLayer<Dtype>::feedforward() {
	reshape();

	this->_outputData[0]->_data = this->_inputData[0]->_data;
}

template <typename Dtype>
void FlattenLayer<Dtype>::backpropagation() {
	this->_inputData[0]->_grad = this->_outputData[0]->_grad;
}

template <typename Dtype>
void FlattenLayer<Dtype>::initialize() {

}



template class FlattenLayer<float>;
