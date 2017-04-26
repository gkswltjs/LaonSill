/*
 * AccuracyLayer.cpp
 *
 *  Created on: Apr 25, 2017
 *      Author: jkim
 */

#include "AccuracyLayer.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
AccuracyLayer<Dtype>::AccuracyLayer(Builder* builder)
: Layer<Dtype>(builder) {
	initialize(builder);
}

template <typename Dtype>
AccuracyLayer<Dtype>::~AccuracyLayer() {

}

template <typename Dtype>
void AccuracyLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	// label shape는 변하지 않음.
	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	// XXX: SO!!!! TEMPORAL
	//this->_inputData[0]->reshape({10, 1, 1000, 1});

	this->_inputShape[0] = this->_inputData[0]->getShape();
	this->_inputShape[1] = this->_inputData[1]->getShape();


	this->_printOn();
	this->_inputData[0]->print_shape();
	this->_inputData[1]->print_shape();
	//exit(1);
	this->_printOff();





	SASSERT(this->topK <= this->_inputData[0]->getCount() / this->_inputData[1]->getCount(),
			"topK must be less than or equal to the number of classes.");
	this->outerNum = this->_inputData[0]->getCountByAxis(0, this->labelAxis);
	this->innerNum = this->_inputData[0]->getCountByAxis(this->labelAxis + 1);
	SASSERT(this->outerNum * this->innerNum == this->_inputData[1]->getCount(),
			"Number of labels must match number of predictions.");

	vector<uint32_t> outputShape({1, 1, 1, 1});
	this->_outputData[0]->reshape(outputShape);
}

template <typename Dtype>
void AccuracyLayer<Dtype>::feedforward() {
	reshape();

	Dtype accuracy = 0;
	const Dtype* inputData = this->_inputData[0]->host_data();
	const Dtype* inputLabel = this->_inputData[1]->host_data();
	const int dim = this->_inputData[0]->getCount() / this->outerNum;
	const int numLabels = this->_inputData[0]->getShape(this->labelAxis);
	vector<Dtype> maxVal(this->topK + 1);
	vector<int> maxId(this->topK + 1);

	int count = 0;
	for (int i = 0; i < this->outerNum; i++) {
		for (int j = 0; j < this->innerNum; j++) {
			const int labelValue = static_cast<int>(inputLabel[i * this->innerNum + j]);
			if (this->hasIgnoreLabel && labelValue == this->ignoreLabel) {
				continue;
			}
			SASSERT0(labelValue >= 0);
			SASSERT0(labelValue < numLabels);
			// Tok-k accuracy
			vector<pair<Dtype, int>> inputDataVector;
			for (int k = 0; k < numLabels; k++) {
				inputDataVector.push_back(make_pair(
						inputData[i * dim + k * this->innerNum + j], k));
			}
			std::partial_sort(
					inputDataVector.begin(), inputDataVector.begin() + this->topK,
					inputDataVector.end(), std::greater<pair<Dtype, int>>());
			// check if true label is in top k predictions
			for (int k = 0; k < this->topK; k++) {
				if (inputDataVector[k].second == labelValue) {
					accuracy++;
					break;
				}
			}
			count++;
		}
	}

	this->_outputData[0]->mutable_host_data()[0] = accuracy / count;
	// Accuracy layer should not be used as a loss function.
}

template <typename Dtype>
void AccuracyLayer<Dtype>::backpropagation() {
	//SASSERT(false, "Not implemented yet.");
}


template <typename Dtype>
Dtype AccuracyLayer<Dtype>::getAccuracy() {
	Dtype accuracy = this->_outputData[0]->host_data()[0];
	return accuracy;
}





template <typename Dtype>
void AccuracyLayer<Dtype>::initialize(Builder* builder) {
	SASSERT0(this->_inputs.size() == 2 && this->_outputs.size() == 1);
	SASSERT((this->_inputs[0] != this->_outputs[0]) &&
			(this->_inputs[1] != this->_outputs[0]),
			"this layer does not allow in-place computation.");

	this->topK = builder->_topK;
	this->labelAxis = builder->_axis;

	this->hasIgnoreLabel = (builder->_ignoreLabel >= 0);
	if (this->hasIgnoreLabel) {
		this->ignoreLabel = builder->_ignoreLabel;
	}
}



template class AccuracyLayer<float>;
