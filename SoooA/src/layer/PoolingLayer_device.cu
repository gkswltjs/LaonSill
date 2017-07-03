/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#include "PoolingLayer.h"
#include "PropMgmt.h"

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

	//int n = 0, c = 0, h = 0, w = 0;
	/*
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(
			this->pooling_fn->getPoolDesc(),
			this->inputTensorDesc,
			&n, &c, &h, &w));
			*/

	pool_dim pool_d = SLPROP(Pooling, poolDim);

	int pooledHeight = static_cast<int>(ceil(static_cast<float>(
			rows + 2 * pool_d.pad - pool_d.rows) / pool_d.stride)) + 1;
	int pooledWidth = static_cast<int>(ceil(static_cast<float>(
			cols + 2 * pool_d.pad - pool_d.cols) / pool_d.stride)) + 1;

	if (pool_d.pad) {
		if ((pooledHeight - 1) * pool_d.stride >= rows + pool_d.pad) {
			pooledHeight--;
		}
		if ((pooledWidth - 1) * pool_d.stride >= cols + pool_d.pad) {
			pooledWidth--;
		}
		assert((pooledHeight - 1) * pool_d.stride < rows + pool_d.pad);
		assert((pooledWidth - 1) * pool_d.stride < cols + pool_d.pad);
	}

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, pooledHeight, pooledWidth));

	/*
	uint32_t obatches = static_cast<uint32_t>(batches);
	uint32_t ochannels = static_cast<uint32_t>(channels);
	uint32_t orows = static_cast<uint32_t>(pooledHeight);
	uint32_t ocols = static_cast<uint32_t>(pooledWidth);
	*/

#if POOLINGLAYER_LOG
	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			this->name.c_str(), obatches, ochannels, orows, ocols);
#endif

	this->_inputShape[0] = inputShape;
	this->_outputData[0]->reshape({
		static_cast<uint32_t>(batches),
		static_cast<uint32_t>(channels),
		static_cast<uint32_t>(pooledHeight),
		static_cast<uint32_t>(pooledWidth)});

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
		Layer<Dtype>::_shape();
	}
	*/
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
	if (SLPROP_BASE(propDown)[0]) {
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


