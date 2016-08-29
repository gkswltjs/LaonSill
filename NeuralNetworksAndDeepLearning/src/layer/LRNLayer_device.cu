/*
 * LRNLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#ifdef GPU_MODE

#include "LRNLayer.h"
#include "../Util.h"

template <typename Dtype>
LRNLayer<Dtype>::~LRNLayer() {
	checkCUDNN(cudnnDestroyLRNDescriptor(lrnDesc));
}

template <typename Dtype>
void LRNLayer<Dtype>::initialize(lrn_dim lrn_d) {
	this->type = Layer<Dtype>::LRN;
	this->lrn_d = lrn_d;

	checkCUDNN(cudnnCreateLRNDescriptor(&lrnDesc));
	checkCUDNN(cudnnSetLRNDescriptor(lrnDesc, lrn_d.local_size, lrn_d.alpha, lrn_d.beta, lrn_d.k));
}


// (1 + alpha/n * sigma(i)(xi^2))^beta
template <typename Dtype>
void LRNLayer<Dtype>::_feedforward() {
	//this->d_input = input;

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	this->_input->print_data("d_input:");

	const Dtype* d_input = this->_input->device_data();
	Dtype* d_output = this->_output->mutable_device_data();
	checkCUDNN(cudnnLRNCrossChannelForward(Cuda::cudnnHandle,
			lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&Cuda::alpha, this->inputTensorDesc, d_input,
			&Cuda::beta, this->outputTensorDesc, d_output));

	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, 1, 1, this->name+string("/d_output:"));
	this->_output->print_data(this->name+string("/d_output:"));
}

template <typename Dtype>
void LRNLayer<Dtype>::_backpropagation() {
	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	this->_output->print_grad("d_delta_output:");
	const Dtype* d_output = this->_output->device_data();
	const Dtype* d_delta_output = this->_output->device_grad();
	const Dtype* d_input = this->_input->device_data();
	Dtype* d_delta_input = this->_input->mutable_device_grad();
	checkCUDNN(cudnnLRNCrossChannelBackward(Cuda::cudnnHandle,
			lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&Cuda::alpha, this->outputTensorDesc, d_output, this->outputTensorDesc, d_delta_output, this->inputTensorDesc, d_input,
			&Cuda::beta, this->inputTensorDesc, d_delta_input));

	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_delta_input:");
	this->_input->print_grad("d_delta_input:");
}


template LRNLayer<float>::~LRNLayer();
template void LRNLayer<float>::initialize(lrn_dim lrn_d);
template void LRNLayer<float>::_feedforward();
template void LRNLayer<float>::_backpropagation();

#endif










