/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#ifdef GPU_MODE

#include "PoolingLayer.h"

void PoolingLayer::_shape(bool recursive) {
	cudnnTensorDescriptor_t tempInputTensorDesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&tempInputTensorDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(tempInputTensorDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols));

	int n, c, h, w;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(pooling_fn->getPoolDesc(),
			tempInputTensorDesc,
			&n, &c, &h, &w));

	out_dim.batches = n;
	out_dim.channels = c;
	out_dim.rows = h;
	out_dim.cols = w;

	checkCUDNN(cudnnDestroyTensorDescriptor(tempInputTensorDesc));

	if(recursive) {
		HiddenLayer::_shape();
	}

	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta, sizeof(DATATYPE)*out_dim.batchsize()));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*in_dim.batchsize()));
}

void PoolingLayer::_feedforward() {
	//this->d_input = input;

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	_input->print_data("d_input:");
	const DATATYPE* d_input = _input->device_data();
	DATATYPE* d_output = _output->mutable_device_data();
	pooling_fn->pool(inputTensorDesc, d_input, outputTensorDesc, d_output);

	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, 1, 1, this->name+string("/d_output:"));
	_output->print_data(this->name+string("/d_output:"));
}

void PoolingLayer::_backpropagation() {
	// backpropagate delta to delta_input
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_output:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	_output->print_data("d_output:");
	_input->print_data("d_input:");

	const DATATYPE* d_output = _output->device_data();
	const DATATYPE* d_delta_output = _output->device_grad();
	const DATATYPE* d_input = _input->device_data();
	DATATYPE* d_delta_input = _input->mutable_device_grad();
	pooling_fn->d_pool(outputTensorDesc, d_output, d_delta_output, inputTensorDesc, d_input, d_delta_input);

	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_delta_input:");
	_input->print_grad("d_delta_input:");
}

#endif




