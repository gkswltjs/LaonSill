/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#include "PoolingLayer.h"
#include "PropMgmt.h"

#define POOLINGLAYER_LOG 1

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
			this->pooling_fn->getPoolDesc(),
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
	const string name = SLPROP_BASE(name);
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			name.c_str(), obatches, ochannels, orows, ocols);
#endif

	this->_inputShape[0] = inputShape;
	this->_outputData[0]->reshape({obatches, ochannels, orows, ocols});
}

template <typename Dtype>
void PoolingLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

	this->pooling_fn->forward(this->inputTensorDesc, d_inputData,
			this->outputTensorDesc, d_outputData);
}

template <typename Dtype>
void PoolingLayer<Dtype>::backpropagation() {
	const vector<bool> propDown = SLPROP_BASE(propDown);
	if (propDown[0]) {
		const Dtype* d_outputData = this->_outputData[0]->device_data();
		const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
		const Dtype* d_inputData = this->_inputData[0]->device_data();
		Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();
		this->pooling_fn->backward(this->outputTensorDesc, d_outputData, d_outputGrad,
				this->inputTensorDesc, d_inputData, d_inputGrad);
	}
}


template void PoolingLayer<float>::reshape();
template void PoolingLayer<float>::feedforward();
template void PoolingLayer<float>::backpropagation();





/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* PoolingLayer<Dtype>::initLayer() {
    PoolingLayer* layer = new PoolingLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void PoolingLayer<Dtype>::destroyLayer(void* instancePtr) {
    PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void PoolingLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool PoolingLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void PoolingLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void PoolingLayer<Dtype>::backwardTensor(void* instancePtr) {
	PoolingLayer<Dtype>* layer = (PoolingLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void PoolingLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template void* PoolingLayer<float>::initLayer();
template void PoolingLayer<float>::destroyLayer(void* instancePtr);
template void PoolingLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool PoolingLayer<float>::allocLayerTensors(void* instancePtr);
template void PoolingLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void PoolingLayer<float>::backwardTensor(void* instancePtr);
template void PoolingLayer<float>::learnTensor(void* instancePtr);


