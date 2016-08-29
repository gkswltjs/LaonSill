/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#ifdef GPU_MODE

#include "PoolingLayer.h"

template <typename Dtype>
void PoolingLayer<Dtype>::_shape(bool recursive) {
	cudnnTensorDescriptor_t tempInputTensorDesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&tempInputTensorDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(tempInputTensorDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				this->in_dim.batches, this->in_dim.channels, this->in_dim.rows, this->in_dim.cols));

	int n, c, h, w;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(pooling_fn->getPoolDesc(),
			tempInputTensorDesc,
			&n, &c, &h, &w));

	this->out_dim.batches = n;
	this->out_dim.channels = c;
	this->out_dim.rows = h;
	this->out_dim.cols = w;

	checkCUDNN(cudnnDestroyTensorDescriptor(tempInputTensorDesc));

	if(recursive) {
		HiddenLayer<Dtype>::_shape();
	}

	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta, sizeof(Dtype)*out_dim.batchsize()));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(Dtype)*in_dim.batchsize()));
}

template <typename Dtype>
void PoolingLayer<Dtype>::_feedforward() {
	//this->d_input = input;

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	this->_input->print_data("d_input:");
	const Dtype* d_input = this->_input->device_data();
	Dtype* d_output = this->_output->mutable_device_data();
	pooling_fn->forward(this->inputTensorDesc, d_input, this->outputTensorDesc, d_output);

	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, 1, 1, this->name+string("/d_output:"));
	this->_output->print_data(this->name+string("/d_output:"));
}

template <typename Dtype>
void PoolingLayer<Dtype>::_backpropagation() {
	// backpropagate delta to delta_input
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_output:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	this->_output->print_data("d_output:");
	this->_input->print_data("d_input:");

	const Dtype* d_output = this->_output->device_data();
	const Dtype* d_delta_output = this->_output->device_grad();
	const Dtype* d_input = this->_input->device_data();
	Dtype* d_delta_input = this->_input->mutable_device_grad();
	pooling_fn->backward(this->outputTensorDesc, d_output, d_delta_output, this->inputTensorDesc, d_input, d_delta_input);

	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_delta_input:");
	this->_input->print_grad("d_delta_input:");
}


template void PoolingLayer<float>::_shape(bool recursive);
template void PoolingLayer<float>::_feedforward();
template void PoolingLayer<float>::_backpropagation();

#endif




