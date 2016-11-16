/*
 * DepthConcatLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "DepthConcatLayer.h"

using namespace std;

//#define DEPTHCONCAT_LOG

template <typename Dtype>
DepthConcatLayer<Dtype>::DepthConcatLayer() {
	this->type = Layer<Dtype>::DepthConcat;
}

template <typename Dtype>
DepthConcatLayer<Dtype>::DepthConcatLayer(Builder* builder)
	: HiddenLayer<Dtype>(builder) {
	initialize();
}

template <typename Dtype>
DepthConcatLayer<Dtype>::DepthConcatLayer(const string name)
	: HiddenLayer<Dtype>(name) {
	initialize();
}

template <typename Dtype>
DepthConcatLayer<Dtype>::~DepthConcatLayer() {}

template <typename Dtype>
void DepthConcatLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::DepthConcat;
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::shape() {
	Layer<Dtype>::_adjustInputShape();

	// 입력 데이터의 shape가 변경된 것이 있는 지 확인
	bool inputShapeReshaped = false;
	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (Layer<Dtype>::_isInputShapeChanged(i)) {
			inputShapeReshaped = true;
			this->_inputShape[i] = this->_inputData[i]->getShape();
		}
	}

	if (!inputShapeReshaped) {
		return;
	}

	uint32_t batches 	= this->_inputShape[0][0];
	uint32_t channels 	= 0;
	uint32_t rows 		= this->_inputShape[0][2];
	uint32_t cols 		= this->_inputShape[0][3];

	for (uint32_t i = 0; i < this->_inputData.size(); i++) {
		channels += this->_inputData[i]->getShape()[1];
	}

	this->_outputData[0]->shape({batches, channels, rows, cols});

	/*
	this->out_dim.channels = 0;
	for (uint32_t i = 0; i < this->_inputs.size(); i++) {
		this->out_dim.channels += this->_inputData[i]->getShape()[1];
	}
	this->out_dim.batches = this->_inputData[0]->getShape()[0];
	this->out_dim.rows = this->_inputData[0]->getShape()[2];
	this->out_dim.cols = this->_inputData[0]->getShape()[3];

	HiddenLayer<Dtype>::shape();

	this->in_dim = this->out_dim;

	if (recursive) {
		HiddenLayer<Dtype>::_shape();
	}
	*/

#ifdef DEPTHCONCAT_LOG
	cout << "shape depthConcatLayer in_dim: " << this->in_dim.batches << "x" << this->in_dim.channels << "x" << this->in_dim.rows << "x" << this->in_dim.cols << endl;
	cout << "shape depthConcatLayer out_dim: " << this->out_dim.batches << "x" << this->out_dim.channels << "x" << this->out_dim.rows << "x" << this->out_dim.cols << endl;
#endif
}

/*
template <typename Dtype>
void DepthConcatLayer<Dtype>::reshape(uint32_t idx, io_dim in_dim) {
	//if (this->isFirstPrevLayerRequest(idx)) this->out_dim.channels = 0;
	this->out_dim.channels += in_dim.channels;
	HiddenLayer<Dtype>::reshape(idx, in_dim);

#ifdef DEPTHCONCAT_LOG
	cout << "reshape depthConcatLayer in_dim: " << this->in_dim.batches << "x" << this->in_dim.channels << "x" << this->in_dim.rows << "x" << this->in_dim.cols << endl;
	cout << "reshape depthConcatLayer out_dim: " << this->out_dim.batches << "x" << this->out_dim.channels << "x" << this->out_dim.rows << "x" << this->out_dim.cols << endl;
#endif
}
*/


template <typename Dtype>
void DepthConcatLayer<Dtype>::_feedforward() {
	uint32_t batchOffset = 0;
	for (uint32_t i = 0; i < this->_inputs.size(); i++) {
		batchOffset += this->_inputData[i]->getCountByAxis(1);
	}

	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();
	const uint32_t batchSize = this->_inputData[0]->getShape()[0];
	uint32_t inBatchOffset = 0;
	for (uint32_t i = 0; i < this->_inputs.size(); i++) {
		const Dtype* d_inputData = this->_inputData[i]->device_data();
		const uint32_t inputCountByChannel = this->_inputData[i]->getCountByAxis(1);
		if (i > 0) {
			inBatchOffset += this->_inputData[i-1]->getCountByAxis(1);
		}
		for (uint32_t j = 0; j < batchSize; j++) {
			checkCudaErrors(cudaMemcpyAsync(
					d_outputData+batchOffset*j+inBatchOffset,
					d_inputData+inputCountByChannel*j,
					inputCountByChannel,
					cudaMemcpyDeviceToDevice));
		}
	}
}


template <typename Dtype>
void DepthConcatLayer<Dtype>::_backpropagation() {
	uint32_t batchOffset = 0;
	for (uint32_t i = 0; i < this->_inputs.size(); i++) {
		batchOffset += this->_inputData[i]->getCountByAxis(1);
	}

	const Dtype* d_outputData = this->_outputData[0]->device_data();
	const uint32_t batchSize = this->_inputData[0]->getShape()[0];
	uint32_t inBatchOffset = 0;
	for (uint32_t i = 0; i < this->_inputs.size(); i++) {
		Dtype* d_inputData = this->_inputData[i]->mutable_device_data();
		const uint32_t inputCountByChannel = this->_inputData[i]->getCountByAxis(1);
		if (i > 0) {
			inBatchOffset += this->_inputData[i-1]->getCountByAxis(1);
		}
		for (uint32_t j = 0; j < batchSize; j++) {
			checkCudaErrors(cudaMemcpyAsync(
					d_inputData+inputCountByChannel*j,
					d_outputData+batchOffset*j+inBatchOffset,
					inputCountByChannel,
					cudaMemcpyDeviceToDevice));
		}
	}
}


template <typename Dtype>
void DepthConcatLayer<Dtype>::_clearShape() {
	HiddenLayer<Dtype>::_clearShape();
}





#ifndef GPU_MODE
template <typename Dtype>
void DepthConcatLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::DepthConcat;

	this->offsetIndex = 0;
	this->input.reset();
	this->delta_input.set_size(size(output));
	this->delta_input.zeros();
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::feedforward(uint32_t idx, const rcube &input, const char *end=0) {
	this->input = join_slices(this->input, input);
	Util::printCube(this->input, "input:");

	this->offsets.push_back(this->input.n_slices);

	if(!isLastPrevLayerRequest(idx)) return;

	this->output = this->input;

	propFeedforward(this->output, end);

	// backward pass에서 input을 사용하지 않으므로 여기서 reset할 수 있음
	this->input.reset();
	this->offsetIndex = 0;
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::backpropagation(uint32_t idx, HiddenLayer<Dtype>* next_layer) {
	Util::printCube(delta_input, "delta_input:");
	rcube w_next_delta(size(delta_input));
	Util::convertCube(next_layer->getDeltaInput(), delta_input);
	delta_input += w_next_delta;
	// delta_input = join_slices(this->delta_input, next_layer->getDeltaInput());
	if(!isLastNextLayerRequest(idx)) return;

	propBackpropagation();
	this->delta_input.zeros();
}
#endif



template class DepthConcatLayer<float>;







