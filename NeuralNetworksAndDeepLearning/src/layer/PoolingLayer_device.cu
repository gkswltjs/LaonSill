/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#ifdef GPU_MODE

#include "PoolingLayer.h"

#define POOLINGLAYER_LOG 0

using namespace std;

template <typename Dtype>
void PoolingLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	int n = 0, c = 0, h = 0, w = 0;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(
			pooling_fn->getPoolDesc(),
			this->inputTensorDesc,
			&n, &c, &h, &w));

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c, h, w));

	uint32_t obatches = static_cast<uint32_t>(n);
	uint32_t ochannels = static_cast<uint32_t>(c);
	uint32_t orows = static_cast<uint32_t>(h);
	uint32_t ocols = static_cast<uint32_t>(w);

#if POOLINGLAYER_LOG
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			this->name.c_str(), obatches, ochannels, orows, ocols);
#endif

	this->_inputShape[0] = inputShape;
	this->_outputData[0]->reshape({obatches, ochannels, orows, ocols});

	/*
	this->setInDimension(this->_inputData[0]->getShape());

	cudnnTensorDescriptor_t tempInputTensorDesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&tempInputTensorDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(tempInputTensorDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				this->in_dim.batches, this->in_dim.channels, this->in_dim.rows,
                this->in_dim.cols));

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
	*/
}

template <typename Dtype>
void PoolingLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

	this->_inputData[0]->print_data();

	pooling_fn->forward(this->inputTensorDesc, d_inputData,
			this->outputTensorDesc, d_outputData);

	this->_outputData[0]->print_data();
}

template <typename Dtype>
void PoolingLayer<Dtype>::backpropagation() {
	if (this->_propDown[0]) {
		this->_outputData[0]->print_data();
		this->_inputData[0]->print_data();

		const Dtype* d_outputData = this->_outputData[0]->device_data();
		const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
		const Dtype* d_inputData = this->_inputData[0]->device_data();
		Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();
		pooling_fn->backward(this->outputTensorDesc, d_outputData, d_outputGrad,
				this->inputTensorDesc, d_inputData, d_inputGrad);

		this->_inputData[0]->print_grad();
	}
}


template void PoolingLayer<float>::reshape();
template void PoolingLayer<float>::feedforward();
template void PoolingLayer<float>::backpropagation();

#endif
