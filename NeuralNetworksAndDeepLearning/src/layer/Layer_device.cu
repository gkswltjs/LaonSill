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
	printf("%15s_shape():in-%dx%dx%dx%d, out-%dx%dx%dx%d\n",
			name.c_str(), in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches,
			out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches);

	// 다른 레이어에 의해 이미 shape 처리되지 않은 Data에 대해서만 shape를 수행한다.
	if (_inputData.size() > 0 && _inputData[0]->getCount() == 0)
		_inputData[0]->shape({in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols});
	if (_outputData.size() > 0 &&_outputData[0]->getCount() == 0)
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



































