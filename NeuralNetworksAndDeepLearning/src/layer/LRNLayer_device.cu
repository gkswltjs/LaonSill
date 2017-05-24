/*
 * LRNLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#ifdef GPU_MODE

#include "LRNLayer.h"
#include "PropMgmt.h"
#include "Util.h"

using namespace std;

template <typename Dtype>
LRNLayer<Dtype>::~LRNLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
	checkCUDNN(cudnnDestroyLRNDescriptor(lrnDesc));
}

template <typename Dtype>
void LRNLayer<Dtype>::initialize(lrn_dim lrn_d) {
	this->type = Layer<Dtype>::LRN;
	this->lrn_d = lrn_d;

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
	checkCUDNN(cudnnCreateLRNDescriptor(&lrnDesc));
	checkCUDNN(cudnnSetLRNDescriptor(lrnDesc, lrn_d.local_size, lrn_d.alpha, 
                                     lrn_d.beta, lrn_d.k));
}

// (1 + alpha/n * sigma(i)(xi^2))^beta
template <typename Dtype>
void LRNLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* d_inputData = this->_inputData[0]->device_data();
	Dtype* d_outputData = this->_outputData[0]->mutable_device_data();
	checkCUDNN(cudnnLRNCrossChannelForward(Cuda::cudnnHandle,
			lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&Cuda::alpha, this->inputTensorDesc, d_inputData,
			&Cuda::beta, this->outputTensorDesc, d_outputData));

	this->_outputData[0]->print_data(this->name+string("/d_output:"));
}

template <typename Dtype>
void LRNLayer<Dtype>::backpropagation() {
	if (this->_propDown[0]) {
		const Dtype* d_outputData = this->_outputData[0]->device_data();
		const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
		const Dtype* d_inputData = this->_inputData[0]->device_data();
		Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();
		checkCUDNN(cudnnLRNCrossChannelBackward(Cuda::cudnnHandle,
				lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &Cuda::alpha, this->outputTensorDesc,
                d_outputData, this->outputTensorDesc, d_outputGrad,
				this->inputTensorDesc, d_inputData,
				&Cuda::beta, this->inputTensorDesc, d_inputGrad));
	}
}

template LRNLayer<float>::~LRNLayer();
template void LRNLayer<float>::initialize(lrn_dim lrn_d);
template void LRNLayer<float>::feedforward();
template void LRNLayer<float>::backpropagation();

#endif




/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* LRNLayer<Dtype>::initLayer() {
    LRNLayer* layer = new LRNLayer<Dtype>(SLPROP_BASE(name));
    return (void*)layer;
}

template<typename Dtype>
void LRNLayer<Dtype>::destroyLayer(void* instancePtr) {
    LRNLayer<Dtype>* layer = (LRNLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void LRNLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    LRNLayer<Dtype>* layer = (LRNLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool LRNLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    LRNLayer<Dtype>* layer = (LRNLayer<Dtype>*)instancePtr;
    //layer->reshape();
    return true;
}

template<typename Dtype>
void LRNLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    cout << "LRNLayer.. forward(). miniBatchIndex : " << miniBatchIdx << endl;
}

template<typename Dtype>
void LRNLayer<Dtype>::backwardTensor(void* instancePtr) {
    cout << "LRNLayer.. backward()" << endl;
}

template<typename Dtype>
void LRNLayer<Dtype>::learnTensor(void* instancePtr) {
    cout << "LRNLayer.. learn()" << endl;
}

template void* LRNLayer<float>::initLayer();
template void LRNLayer<float>::destroyLayer(void* instancePtr);
template void LRNLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool LRNLayer<float>::allocLayerTensors(void* instancePtr);
template void LRNLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void LRNLayer<float>::backwardTensor(void* instancePtr);
template void LRNLayer<float>::learnTensor(void* instancePtr);
