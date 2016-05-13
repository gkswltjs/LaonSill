/*
 * ConvPoolLayer.cpp
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#include "ConvPoolLayer.h"

ConvPoolLayer::ConvPoolLayer(int n_in, int n_out)
	: HiddenLayer(n_in, n_out) {
	// TODO Auto-generated constructor stub

}

ConvPoolLayer::~ConvPoolLayer() {
	// TODO Auto-generated destructor stub
}


void ConvPoolLayer::feedforward(const vec &input) {

}

void ConvPoolLayer::backpropagation(const mat &next_w, const vec &next_delta, const vec &input) {

}

void ConvPoolLayer::reset_nabla() {

}


void ConvPoolLayer::update(double eta, double lambda, int n, int miniBatchSize) {

}
