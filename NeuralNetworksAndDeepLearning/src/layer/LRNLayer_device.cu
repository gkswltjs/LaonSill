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
	this->_input->print_data("inputData:");

	const Dtype* d_inputData = this->_input->device_data();
	Dtype* d_outputData = this->_output->mutable_device_data();
	checkCUDNN(cudnnLRNCrossChannelForward(Cuda::cudnnHandle,
			lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&Cuda::alpha, this->inputTensorDesc, d_inputData,
			&Cuda::beta, this->outputTensorDesc, d_outputData));

	this->_output->print_data(this->name+string("/d_output:"));
}

template <typename Dtype>
void LRNLayer<Dtype>::_backpropagation() {

	const Dtype* d_outputData = this->_output->device_data();
	const Dtype* d_outputGrad = this->_output->device_grad();
	const Dtype* d_inputData = this->_input->device_data();
	Dtype* d_inputGrad = this->_input->mutable_device_grad();
	checkCUDNN(cudnnLRNCrossChannelBackward(Cuda::cudnnHandle,
			lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&Cuda::alpha, this->outputTensorDesc, d_outputData, this->outputTensorDesc, d_outputGrad,
			this->inputTensorDesc, d_inputData,
			&Cuda::beta, this->inputTensorDesc, d_inputGrad));

	/*
	if(this->_input->is_nan_grad()) {
		Data<Dtype>::printConfig = 1;
		this->_output->print_grad("outputGrad:");
		this->_input->print_grad("inputGrad:");
		Data<Dtype>::printConfig = 0;
		exit(1);
	}
	*/


}


template LRNLayer<float>::~LRNLayer();
template void LRNLayer<float>::initialize(lrn_dim lrn_d);
template void LRNLayer<float>::_feedforward();
template void LRNLayer<float>::_backpropagation();

#endif










