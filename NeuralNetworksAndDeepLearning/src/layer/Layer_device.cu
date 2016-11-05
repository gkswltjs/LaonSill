/*
 * Layer.cu
 *
 *  Created on: 2016. 8. 25.
 *      Author: jhkim
 */


#ifdef GPU_MODE

#include "Layer.h"
#include "../cuda/Cuda.h"

using namespace std;

template <typename Dtype>
void Layer<Dtype>::_shape(bool recursive) {
	char message[256];
	sprintf(message, "%s---_shape():in-%dx%dx%dx%d, out-%dx%dx%dx%d",
			name.c_str(), in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches,
			out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches);
	Util::setPrint(true);
	Util::printMessage(string(message));
	Util::setPrint(false);

	//checkCudaErrors(Util::ucudaMalloc(&this->d_input, sizeof(Dtype)*in_dim.batchsize()));		//batch size 고려
	//checkCudaErrors(Util::ucudaMalloc(&this->d_output, sizeof(Dtype)*out_dim.batchsize()));		//batch size 고려
	if (_inputData[0]->getCount() == 0)
		_inputData[0]->shape({in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols});
	if (_outputData[0]->getCount() == 0)
		_outputData[0]->shape({out_dim.batches, out_dim.channels, out_dim.rows, out_dim.cols});

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));

	checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols));
	checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			out_dim.batches, out_dim.channels, out_dim.rows, out_dim.cols));
}

template <typename Dtype>
void Layer<Dtype>::_clearShape() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));

	//delete _input;
	//delete _output;
	//_input = NULL;
	//_output = NULL;
	inputTensorDesc = NULL;
	outputTensorDesc = NULL;
}

template void Layer<float>::_shape(bool recursive);
template void Layer<float>::_clearShape();

#endif



































