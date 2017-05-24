/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#include "PoolingLayer.h"

using namespace std;


template <typename Dtype>
PoolingLayer<Dtype>::PoolingLayer(const string& name)
: Layer<Dtype>(name) {

}

template <typename Dtype>
PoolingLayer<Dtype>::PoolingLayer(Builder* builder)
	: Layer<Dtype>(builder) {
	initialize(builder->_poolDim, builder->_poolingType);
}

template <typename Dtype>
PoolingLayer<Dtype>::~PoolingLayer() {
	PoolingFactory<Dtype>::destroy(pooling_fn);
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
}

template <typename Dtype>
void PoolingLayer<Dtype>::initialize(pool_dim pool_d, PoolingType poolingType) {
	this->type = Layer<Dtype>::Pooling;
	this->pool_d = pool_d;
	this->pooling_fn = PoolingFactory<Dtype>::create(poolingType, pool_d);

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
}

#ifndef GPU_MODE
template <typename Dtype>
void PoolingLayer<Dtype>::initialize(pool_dim pool_d,
    typename PoolingType poolingType) {
	this->type = Layer<Dtype>::Pooling;

	this->out_dim.rows = in_dim.rows / pool_d.rows;
	this->out_dim.cols = in_dim.rows / pool_d.cols;
	this->out_dim.channels = in_dim.channels;

	//this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);

	this->pool_d = pool_d;

	this->pooling_fn = PoolingFactory::create(poolingType);

	this->pool_map.set_size(in_dim.rows/pool_d.stride, in_dim.cols/pool_d.stride,
                            in_dim.channels);
	this->output.set_size(size(pool_map));
	this->delta_input.set_size(size(input));
	this->delta_input.zeros();
}

template <typename Dtype>
void PoolingLayer<Dtype>::feedforward(uint32_t idx, const rcube &input, const char *end=0) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	Util::convertCube(input, this->input);
	pooling_fn->forward(pool_d, this->input, pool_map, output);

	propFeedforward(this->output, end);
}

template <typename Dtype>
void PoolingLayer<Dtype>::backpropagation(uint32_t idx, Layer *next_layer) {
	// TODO w_next_delta를 모두 합하여 한 번에 d_pool하는 것이 연산적으로 유리, 수정 필요
	rcube w_next_delta(size(output));

	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);
	Util::printCube(next_layer->getDeltaInput(), "delta input:");
	Util::printCube(w_next_delta, "w_next_delta:");

	rcube temp(size(delta_input));
	pooling_fn->backward(pool_d, w_next_delta, pool_map, temp);
	delta_input += temp;
	Util::printCube(delta_input, "delta_input:");


	// dx가 모두 aggregate된 후 이전 레이어로 back propagate한다.
	if(!isLastNextLayerRequest(idx)) return;

	propBackpropagation();
	delta_input.zeros();
}
#endif

template class PoolingLayer<float>;
