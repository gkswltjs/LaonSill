/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#include "PoolingLayer.h"

PoolingLayer::PoolingLayer(io_dim in_dim, pool_dim pool_d, Pooling *pooling_fn)
	: HiddenLayer(in_dim, in_dim) {

	this->out_dim.rows = in_dim.rows / pool_d.rows;
	this->out_dim.cols = in_dim.rows / pool_d.cols;
	this->out_dim.channels = in_dim.channels;

	//this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);

	this->pool_d = pool_d;

	this->pooling_fn = pooling_fn;

	this->pool_map.set_size(in_dim.rows/pool_d.stride, in_dim.cols/pool_d.stride, in_dim.channels);
	this->output.set_size(size(pool_map));
	this->delta_input.set_size(size(input));

}

PoolingLayer::~PoolingLayer() {
	// TODO Auto-generated destructor stub
}


void PoolingLayer::feedforward(const cube &input) {
	Util::convertCube(input, this->input);
	pooling_fn->pool(pool_d, this->input, pool_map, output);

}



void PoolingLayer::backpropagation(HiddenLayer *next_layer) {
	cube w_next_delta(size(output));

	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);
	Util::printCube(next_layer->getDeltaInput(), "delta input:");
	Util::printCube(w_next_delta, "w_next_delta:");

	pooling_fn->d_pool(pool_d, w_next_delta, pool_map, delta_input);
	Util::printCube(delta_input, "delta_input:");
}



void PoolingLayer::reset_nabla() {}


void PoolingLayer::update(double eta, double lambda, int n, int miniBatchSize) {}

































