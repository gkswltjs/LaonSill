/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#include "PoolingLayer.h"
#include "PropMgmt.h"
#include "StdOutLog.h"

using namespace std;

#define POOLINGLAYER_LOG 0

template <typename Dtype>
PoolingLayer<Dtype>::PoolingLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Pooling;

#if POOLINGLAYER_LOG
	STDOUT_LOG("Layer: %s", SLPROP_BASE(name).c_str());
	SLPROP(Pooling, poolDim).print();
#endif
	this->globalPooling = SLPROP(Pooling, globalPooling);
	this->poolDim = SLPROP(Pooling, poolDim);
	this->poolingType = SLPROP(Pooling, poolingType);

	if (this->globalPooling) {
		SASSERT(this->poolDim.pad == 0 && this->poolDim.stride == 1,
				"With globalPooling true, only pad = 0 and stride = 1 is supported.");
	}


	//this->pooling_fn = PoolingFactory<Dtype>::create(poolingType, poolDim);

	checkCUDNN(cudnnCreateTensorDescriptor(&this->inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&this->outputTensorDesc));
}


template <typename Dtype>
PoolingLayer<Dtype>::~PoolingLayer() {
	PoolingFactory<Dtype>::destroy(this->pooling_fn);
	checkCUDNN(cudnnDestroyTensorDescriptor(this->inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(this->outputTensorDesc));
}


template class PoolingLayer<float>;
