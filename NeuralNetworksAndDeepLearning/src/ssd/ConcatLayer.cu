/*
 * ConcatLayer.cpp
 *
 *  Created on: Apr 26, 2017
 *      Author: jkim
 */

#include "ConcatLayer.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
__global__ void Concat(const int nthreads, const Dtype* in_data,
		const bool forward, const int num_concats, const int concat_size,
		const int top_concat_axis, const int bottom_concat_axis,
		const int offset_concat_axis, Dtype* out_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int total_concat_size = concat_size * bottom_concat_axis;
		const int concat_num = index / total_concat_size;
		const int concat_index = index % total_concat_size;
		const int top_index = concat_index +
				(concat_num * top_concat_axis + offset_concat_axis) * concat_size;
		if (forward) {
			out_data[top_index] = in_data[index];
		} else {
			out_data[index] = in_data[top_index];
		}
	}
}


template <typename Dtype>
ConcatLayer<Dtype>::ConcatLayer(Builder* builder)
: Layer<Dtype>(builder) {
	initialize(builder);
}

template <typename Dtype>
ConcatLayer<Dtype>::~ConcatLayer() {
}

template <typename Dtype>
void ConcatLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	bool inputShapeChanged = false;
	for (int i = 0; i < this->_inputData.size(); i++) {
		if (Layer<Dtype>::_isInputShapeChanged(0)) {
			inputShapeChanged = true;
			break;
		}
	}

	if (!inputShapeChanged)
		return;

	const int numAxes = this->_inputData[0]->numAxes();
	SASSERT0(this->concatAxis < numAxes);

	// Initialize with the first Data.
	vector<uint32_t> outputShape = this->_inputData[0]->getShape();
	this->numConcat = this->_inputData[0]->getCountByAxis(0, this->concatAxis);
	this->concatInputSize = this->_inputData[0]->getCountByAxis(this->concatAxis + 1);

	int inputCountSum = this->_inputData[0]->getCount();
	for (int i = 1; i < this->_inputData.size(); i++) {
		SASSERT(numAxes == this->_inputData[i]->numAxes(),
				"All inputs must have the same #axes.");
		for (int j = 0; j < numAxes; j++) {
			if (j == this->concatAxis)
				continue;
			SASSERT(outputShape[j] == this->_inputData[i]->getShape(j),
					"All inputs must have the same shape, except at concatAxis.");
		}
		inputCountSum += this->_inputData[i]->getCount();
		outputShape[this->concatAxis] += this->_inputData[i]->getShape(this->concatAxis);
	}
	this->_outputData[0]->reshape(outputShape);
	SASSERT0(inputCountSum == this->_outputData[0]->getCount());

	if (this->_inputData.size() == 1) {
		this->_outputData[0]->share_data(this->_inputData[0]);
		this->_outputData[0]->share_grad(this->_inputData[0]);
	}
}

template <typename Dtype>
void ConcatLayer<Dtype>::feedforward() {
	reshape();

	if (this->_inputData.size() == 1)
		return;

	Dtype* outputData = this->_outputData[0]->mutable_device_data();
	int offsetConcatAxis = 0;
	const int outputConcatAxis = this->_outputData[0]->getShape(this->concatAxis);
	const bool kForward = true;
	for (int i = 0; i < this->_inputData.size(); i++) {
		const Dtype* inputData = this->_inputData[i]->device_data();
		const int inputConcatAxis = this->_inputData[i]->getShape(this->concatAxis);
		const int inputConcatSize = inputConcatAxis * this->concatInputSize;
		const int nthreads = inputConcatSize * this->numConcat;
		Concat<Dtype><<<SOOOA_GET_BLOCKS(nthreads), SOOOA_CUDA_NUM_THREADS>>>(
				nthreads, inputData, kForward, this->numConcat, this->concatInputSize,
				outputConcatAxis, inputConcatAxis, offsetConcatAxis, outputData);
		offsetConcatAxis += inputConcatAxis;
	}
}

template <typename Dtype>
void ConcatLayer<Dtype>::backpropagation() {
	if (this->_inputData.size() == 1)
		return;

	const Dtype* outputGrad = this->_outputData[0]->device_grad();
	int offsetConcatAxis = 0;
	const int outputConcatAxis = this->_outputData[0]->getShape(this->concatAxis);
	const bool kForward = false;
	for (int i = 0; i < this->_inputData.size(); i++) {
		const int inputConcatAxis = this->_inputData[i]->getShape(this->concatAxis);
		if (this->_propDown[i]) {
			Dtype* inputGrad = this->_inputData[i]->mutable_device_grad();
			const int inputConcatSize = inputConcatAxis * this->concatInputSize;
			const int nthreads = inputConcatSize * this->numConcat;
			Concat<Dtype><<<SOOOA_GET_BLOCKS(nthreads), SOOOA_CUDA_NUM_THREADS>>>(
					nthreads, outputGrad, kForward, this->numConcat, this->concatInputSize,
					outputConcatAxis, inputConcatAxis, offsetConcatAxis, inputGrad);
		}
		offsetConcatAxis += inputConcatAxis;
	}
}

template <typename Dtype>
void ConcatLayer<Dtype>::initialize(Builder* builder) {
	SASSERT0(this->_inputs.size() > 1 && this->_outputs.size() == 1);
	SASSERT(builder->_axis >= 0, "axis should be specified ... ");
	this->concatAxis = builder->_axis;
}

template class ConcatLayer<float>;
