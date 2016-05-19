/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"
#include "../Util.h"

FullyConnectedLayer::FullyConnectedLayer(int n_in, int n_out, double p_dropout, Activation *activation_fn)
	: HiddenLayer(n_in, n_out) {
	initialize(p_dropout, activation_fn);
}

FullyConnectedLayer::FullyConnectedLayer(io_dim in_dim, io_dim out_dim, double p_dropout, Activation *activation_fn)
	: HiddenLayer(in_dim, out_dim) {
	initialize(p_dropout, activation_fn);
}

FullyConnectedLayer::~FullyConnectedLayer() {

}


void FullyConnectedLayer::initialize(double p_dropout, Activation *activation_fn) {
	this->p_dropout = p_dropout;

	int n_in = in_dim.size();
	int n_out = out_dim.size();

	this->bias.set_size(n_out, 1);
	this->weight.set_size(n_out, n_in);
	this->bias.randn();

	//this->weight.randn();
	//this->weight *= 1/sqrt(n_in);				// initial point scaling

	this->nabla_b.set_size(n_out, 1);
	this->nabla_w.set_size(n_out, n_in);
	this->nabla_b.fill(0.0);
	this->nabla_w.fill(0.0);

	this->z.set_size(n_out, 1, 1);
	this->output.set_size(n_out, 1, 1);
	this->delta.set_size(n_out, 1, 1);

	/**
	 * HiddenLayer에서 activation_fn이 할당되는 곳에서 weight initialize 필요
	 * 잊어버리기 쉬울 것 같으니 대책이 필요
	 */
	this->activation_fn = activation_fn;
	if(this->activation_fn) activation_fn->initialize_weight(n_in, this->weight);
}





void FullyConnectedLayer::feedforward(const cube &input) {

	Util::printCube(input, "input:");
	Util::convertCube(input, this->input);
	Util::printCube(this->input, "converted input:");
	//Util::dropoutLayer(this->input, this->p_dropout);
	Util::printMat(weight, "weight:");
	Util::printVec(bias, "bias:");

	z.slice(0) = weight*this->input.slice(0) + bias;
	Util::printCube(z, "z:");

	activation_fn->activate(z, output);
	Util::printCube(output, "output:");
}

void FullyConnectedLayer::backpropagation(HiddenLayer *next_layer) {
	cube sp;
	activation_fn->d_activate(output, sp);

	FullyConnectedLayer *fc_layer = dynamic_cast<FullyConnectedLayer *>(next_layer);
	if(fc_layer) delta.slice(0) = fc_layer->getWeight().t()*fc_layer->getDelta().slice(0) % sp.slice(0);
	else {
		// TODO fc 다음으로 CONV가 온 경우 처리
		//delta.slice(0) = next_delta.slice(0) % sp.slice(0);
	}

	//Util::printMat(delta.slice(0), "delta");
	//Util::printMat(next_w->t(), "next_w");
	//Util::printMat(next_delta.slice(0), "next_delta");

	nabla_b += delta.slice(0);
	nabla_w += delta.slice(0)*input.slice(0).t();
}




void FullyConnectedLayer::reset_nabla() {
	nabla_b.fill(0.0);
	nabla_w.fill(0.0);
}


void FullyConnectedLayer::update(double eta, double lambda, int n, int miniBatchSize) {
	weight = (1-eta*lambda/n)*weight - (eta/miniBatchSize)*nabla_w;
	bias -= eta/miniBatchSize*nabla_b;
}































