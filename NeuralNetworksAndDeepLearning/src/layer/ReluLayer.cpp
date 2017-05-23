/*
 * ReluLayer.cpp
 *
 *  Created on: Jan 25, 2017
 *      Author: jkim
 */

#include "ReluLayer.h"
#include "PropMgmt.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
ReluLayer<Dtype>::ReluLayer(Builder* builder)
: Layer<Dtype>(builder) {
	initialize(builder->_useLeaky, builder->_leaky);
}

template<typename Dtype>
ReluLayer<Dtype>::ReluLayer(const string& name) 
: Layer<Dtype>(name) {
	initialize(false, 0.5);
}

template <typename Dtype>
ReluLayer<Dtype>::~ReluLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(this->tensorDesc));
	checkCUDNN(cudnnDestroyActivationDescriptor(this->activationDesc));
}

template <typename Dtype>
void ReluLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	this->_inputShape[0] = inputShape;

	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->tensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	this->_outputData[0]->reshape(inputShape);
}

template <typename Dtype>
void ReluLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();

    if (this->useLeaky) {
        applyLeakyForward();
    } else {
	    checkCUDNN(cudnnActivationForward(Cuda::cudnnHandle, this->activationDesc,
					&Cuda::alpha, this->tensorDesc, d_inputData,
					&Cuda::beta, this->tensorDesc, d_outputData));
    }

	/*
	Data<Dtype>::printConfig = true;
	this->_inputData[0]->print_data({}, true);
	this->_outputData[0]->print_data({}, true);
	Data<Dtype>::printConfig = false;
	*/
}

template <typename Dtype>
void ReluLayer<Dtype>::backpropagation() {
	const Dtype* d_outputData = this->_outputData[0]->device_data();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();

    if (this->useLeaky) {
        applyLeakyBackward();
    } else {
	    checkCUDNN(cudnnActivationBackward(Cuda::cudnnHandle, this->activationDesc,
					&Cuda::alpha, this->tensorDesc, d_outputData, this->tensorDesc,
					d_outputGrad, this->tensorDesc, d_inputData,
					&Cuda::beta, this->tensorDesc, d_inputGrad));
    }

	/*
	Data<Dtype>::printConfig = true;
	this->_outputData[0]->print_grad({}, true);
	this->_inputData[0]->print_grad({}, true);
	Data<Dtype>::printConfig = false;
	*/
}

template <typename Dtype>
void ReluLayer<Dtype>::initialize(bool useLeaky, double leaky) {
	this->type = Layer<Dtype>::Relu;
    this->useLeaky = useLeaky;
    this->leaky = leaky;

	checkCUDNN(cudnnCreateTensorDescriptor(&this->tensorDesc));

	checkCUDNN(cudnnCreateActivationDescriptor(&this->activationDesc));
	checkCUDNN(cudnnSetActivationDescriptor(this->activationDesc, CUDNN_ACTIVATION_RELU,
			CUDNN_PROPAGATE_NAN, 0.0));
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* ReluLayer<Dtype>::initLayer() {
    ReluLayer* layer = new ReluLayer<Dtype>(SLPROP_BASE(name));
    return (void*)layer;
}

template<typename Dtype>
void ReluLayer<Dtype>::destroyLayer(void* instancePtr) {
    ReluLayer<Dtype>* layer = (ReluLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void ReluLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    ReluLayer<Dtype>* layer = (ReluLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool ReluLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    ReluLayer<Dtype>* layer = (ReluLayer<Dtype>*)instancePtr;
    //layer->reshape();
    return true;
}

template<typename Dtype>
void ReluLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    cout << "ReluLayer.. forward(). miniBatchIndex : " << miniBatchIdx << endl;
}

template<typename Dtype>
void ReluLayer<Dtype>::backwardTensor(void* instancePtr) {
    cout << "ReluLayer.. backward()" << endl;
}

template<typename Dtype>
void ReluLayer<Dtype>::learnTensor(void* instancePtr) {
    cout << "ReluLayer.. learn()" << endl;
}

template class ReluLayer<float>;
