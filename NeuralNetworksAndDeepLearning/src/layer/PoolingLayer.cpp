/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#include "PoolingLayer.h"
#include "PropMgmt.h"

using namespace std;


template <typename Dtype>
PoolingLayer<Dtype>::PoolingLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::Pooling;

	const pool_dim& poolDim = SLPROP(Pooling, poolDim);
	const PoolingType poolingType = SLPROP(Pooling, poolingType);
	this->pooling_fn = PoolingFactory<Dtype>::create(poolingType, poolDim);

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
