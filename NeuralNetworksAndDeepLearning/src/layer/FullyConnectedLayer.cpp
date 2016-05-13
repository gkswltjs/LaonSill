/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"



FullyConnectedLayer::FullyConnectedLayer(int n_in, int n_out, Activation *activation_fn)
	: HiddenLayer(n_in, n_out) {

	//this->n_in = n_in;
	//this->n_out = n_out;

	this->bias.set_size(n_out, 1);
	this->weight.set_size(n_out, n_in);
	this->bias.randn();

	//this->weight.randn();
	//this->weight *= 1/sqrt(n_in);				// initial point scaling

	this->nabla_b.set_size(n_out, 1);
	this->nabla_w.set_size(n_out, n_in);
	this->nabla_b.fill(0.0);
	this->nabla_w.fill(0.0);

	this->z.set_size(n_out, 1);
	this->output.set_size(n_out, 1);
	this->delta.set_size(n_out, 1);

	/**
	 * HiddenLayer에서 activation_fn이 할당되는 곳에서 weight initialize 필요
	 * 잊어버리기 쉬울 것 같으니 대책이 필요
	 */
	this->activation_fn = activation_fn;
	if(activation_fn) activation_fn->initialize_weight(this->n_in, this->weight);
}


FullyConnectedLayer::~FullyConnectedLayer() {

}




void FullyConnectedLayer::feedforward(const vec &input) {
	z = weight*input + bias;
	activation_fn->activate(z, output);
}


void FullyConnectedLayer::backpropagation(const mat &next_w, const vec &next_delta, const vec &input) {
	vec sp;
	activation_fn->d_activate(output, sp);
	delta = next_w.t()*next_delta % sp;

	nabla_b += delta;
	nabla_w += delta*input.t();
}


/*
void FullyConnectedLayer::cost(const vec &target, const vec &input) {
	cost_fn->d_cost(z, activation, target, delta);

	nabla_b += delta;
	nabla_w += delta*input.t();
}
*/


void FullyConnectedLayer::reset_nabla() {
	nabla_b.fill(0.0);
	nabla_w.fill(0.0);
}


void FullyConnectedLayer::update(double eta, double lambda, int n, int miniBatchSize) {
	weight = (1-eta*lambda/n)*weight - (eta/miniBatchSize)*nabla_w;
	bias -= eta/miniBatchSize*nabla_b;
}































