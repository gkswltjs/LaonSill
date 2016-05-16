/*
 * ConvPoolLayer.cpp
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#include "ConvPoolLayer.h"

ConvPoolLayer::ConvPoolLayer(io_dim in_dim, filter_dim filter_d, pool_dim pool_d, Activation *activation_fn)
	: HiddenLayer(in_dim, in_dim) {
	this->in_dim = in_dim;
	this->filter_d = filter_d;
	this->pool_d = pool_d;

	filters = new cube[filter_d.filters];
	nabla_w = new cube[filter_d.filters];
	for(int i = 0; i < filter_d.filters; i++) {
		filters[i].set_size(filter_d.rows, filter_d.cols, filter_d.channels);
		filters[i].randn();
		nabla_w[i].set_size(filter_d.rows, filter_d.cols, filter_d.channels);
		nabla_w[i].fill(0.0);
	}

	biases.set_size(filter_d.filters);
	biases.randn();
	nabla_b.set_size(filter_d.filters);
	nabla_b.randn();

	// TODO z 크기 conv2 결과에 맞춰 조정해야 함
	z.set_size(in_dim.rows-filter_d.rows+1, in_dim.cols-filter_d.cols+1, filter_d.filters);
	activated.set_size(in_dim.rows-filter_d.rows+1, in_dim.cols-filter_d.cols+1, filter_d.filters);
	poold.set_size(activated.n_rows/pool_d.rows, activated.n_cols/pool_d.cols, filter_d.filters);

	z.fill(0.0);

	// TODO activation에 따라 weight 초기화
	this->activation_fn = activation_fn;
	//if(this->activation_fn) this->activation_fn->initialize_weight();
}

ConvPoolLayer::~ConvPoolLayer() {}


void ConvPoolLayer::feedforward(const cube &input) {
	// for i, features (about output)
	for(int i = 0; i < filter_d.filters; i++) {
		// for j, channels (about input)
		for(int j = 0; j < filter_d.channels; j++) {
			z.slice(i) += conv2(input.slice(j), filters[i].slice(j), "same");
		}
		z.slice(i) += biases.row(i);
		//z(i) += biases.row(i);
		z.slice(i) = z.slice(i).submat(0, 0, z.slice(i).n_rows-2, z.slice(i).n_cols-2);
		//z(i) = z(i).submat(0, 0, z(i).n_rows-2, z(i).n_cols-2);

	}


}

void ConvPoolLayer::backpropagation(const mat &next_w, const vec &next_delta, const vec &input) {

}

void ConvPoolLayer::reset_nabla() {

}


void ConvPoolLayer::update(double eta, double lambda, int n, int miniBatchSize) {

}
