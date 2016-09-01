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
}

template <typename Dtype>
void PoolingLayer<Dtype>::_feedforward() {
	this->_input->print_data("inputData:");
	const Dtype* d_inputData = this->_input->device_data();
	Dtype* d_outputData = this->_output->mutable_device_data();
	pooling_fn->forward(this->inputTensorDesc, d_inputData,
			this->outputTensorDesc, d_outputData);
	this->_output->print_data(this->name+string("/outputData:"));
}

template <typename Dtype>
void PoolingLayer<Dtype>::_backpropagation() {
	this->_output->print_data("outputData:");
	this->_input->print_data("inputData:");
	/*
	if(this->_output->is_nan_grad()) {
		cout << this->name << " output gradient nan ... " << endl;
		exit(1);
	}
	*/
	const Dtype* d_outputData = this->_output->device_data();
	const Dtype* d_outputGrad = this->_output->device_grad();
	const Dtype* d_inputData = this->_input->device_data();
	Dtype* d_inputGrad = this->_input->mutable_device_grad();
	pooling_fn->backward(this->outputTensorDesc, d_outputData, d_outputGrad,
			this->inputTensorDesc, d_inputData, d_inputGrad);

	this->_input->print_grad("inputGrad:");
}


template void PoolingLayer<float>::_shape(bool recursive);
template void PoolingLayer<float>::_feedforward();
template void PoolingLayer<float>::_backpropagation();

#endif




