/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#include "PoolingLayer.h"

PoolingLayer::PoolingLayer(string name, io_dim in_dim, pool_dim pool_d, PoolingType poolingType)
	: HiddenLayer(name, in_dim, in_dim) {
	this->type = LayerType::Pooling;

	this->out_dim.rows = in_dim.rows / pool_d.rows;
	this->out_dim.cols = in_dim.rows / pool_d.cols;
	this->out_dim.channels = in_dim.channels;

	//this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);

	this->pool_d = pool_d;

	this->pooling_fn = PoolingFactory::create(poolingType);

	this->pool_map.set_size(in_dim.rows/pool_d.stride, in_dim.cols/pool_d.stride, in_dim.channels);
	this->output.set_size(size(pool_map));
	this->delta_input.set_size(size(input));
	this->delta_input.zeros();
}

PoolingLayer::~PoolingLayer() {
	PoolingFactory::destroy(pooling_fn);
}


void PoolingLayer::feedforward(UINT idx, const rcube &input) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();





	Util::convertCube(input, this->input);
	pooling_fn->pool(pool_d, this->input, pool_map, output);

	propFeedforward(this->output);
}



void PoolingLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	// TODO w_next_delta를 모두 합하여 한 번에 d_pool하는 것이 연산적으로 유리, 수정 필요
	rcube w_next_delta(size(output));

	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);
	Util::printCube(next_layer->getDeltaInput(), "delta input:");
	Util::printCube(w_next_delta, "w_next_delta:");

	rcube temp(size(delta_input));
	pooling_fn->d_pool(pool_d, w_next_delta, pool_map, temp);
	delta_input += temp;
	Util::printCube(delta_input, "delta_input:");


	// dx가 모두 aggregate된 후 이전 레이어로 back propagate한다.
	if(!isLastNextLayerRequest(idx)) return;

	propBackpropagation();
	delta_input.zeros();
}


































