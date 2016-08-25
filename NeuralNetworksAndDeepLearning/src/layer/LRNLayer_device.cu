/*
 * LRNLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#ifdef GPU_MODE

#include "LRNLayer.h"
#include "../Util.h"

LRNLayer::~LRNLayer() {
	checkCUDNN(cudnnDestroyLRNDescriptor(lrnDesc));
}

void LRNLayer::initialize(lrn_dim lrn_d) {
	this->type = Layer::LRN;
	this->lrn_d = lrn_d;

	checkCUDNN(cudnnCreateLRNDescriptor(&lrnDesc));
	checkCUDNN(cudnnSetLRNDescriptor(lrnDesc, lrn_d.local_size, lrn_d.alpha, lrn_d.beta, lrn_d.k));
}


// (1 + alpha/n * sigma(i)(xi^2))^beta
void LRNLayer::_feedforward() {
	//this->d_input = input;

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	_input->print_data("d_input:");

	const DATATYPE* d_input = _input->device_data();
	DATATYPE* d_output = _output->mutable_device_data();
	checkCUDNN(cudnnLRNCrossChannelForward(Cuda::cudnnHandle,
			lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&Cuda::alpha, inputTensorDesc, d_input,
			&Cuda::beta, outputTensorDesc, d_output));

	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, 1, 1, this->name+string("/d_output:"));
	_output->print_data(this->name+string("/d_output:"));
}

void LRNLayer::_backpropagation() {
	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	_output->print_grad("d_delta_output:");
	const DATATYPE* d_output = _output->device_data();
	const DATATYPE* d_delta_output = _output->device_grad();
	const DATATYPE* d_input = _input->device_data();
	DATATYPE* d_delta_input = _input->mutable_device_grad();
	checkCUDNN(cudnnLRNCrossChannelBackward(Cuda::cudnnHandle,
			lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&Cuda::alpha, outputTensorDesc, d_output, outputTensorDesc, d_delta_output, inputTensorDesc, d_input,
			&Cuda::beta, inputTensorDesc, d_delta_input));

	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_delta_input:");
	_input->print_grad("d_delta_input:");
}

#endif










