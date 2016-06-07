/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"
#include "../Util.h"
#include "../exception/Exception.h"

FullyConnectedLayer::FullyConnectedLayer(string name, int n_in, int n_out, double p_dropout, Activation *activation_fn)
	: HiddenLayer(name, n_in, n_out) {
	initialize(p_dropout, activation_fn);
}

FullyConnectedLayer::FullyConnectedLayer(string name, io_dim in_dim, io_dim out_dim, double p_dropout, Activation *activation_fn)
	: HiddenLayer(name, in_dim, out_dim) {
	initialize(p_dropout, activation_fn);
}

FullyConnectedLayer::~FullyConnectedLayer() {

}


void FullyConnectedLayer::initialize(double p_dropout, Activation *activation_fn) {
	this->p_dropout = p_dropout;

	int n_in = in_dim.size();
	int n_out = out_dim.size();

	this->bias.set_size(n_out, 1);
	this->bias.zeros();

	this->weight.set_size(n_out, n_in);


	//this->weight.randn();
	//this->weight *= 1/sqrt(n_in);				// initial point scaling

	this->nabla_b.set_size(n_out, 1);
	this->nabla_w.set_size(n_out, n_in);
	this->nabla_b.zeros();
	this->nabla_w.zeros();

	this->z.set_size(n_out, 1, 1);
	this->output.set_size(n_out, 1, 1);
	this->delta.set_size(n_out, 1, 1);
	this->delta.zeros();
	this->delta_input.set_size(n_in, 1, 1);
	this->delta_input.zeros();

	/**
	 * HiddenLayer에서 activation_fn이 할당되는 곳에서 weight initialize 필요
	 * 잊어버리기 쉬울 것 같으니 대책이 필요
	 */
	this->activation_fn = activation_fn;
	if(this->activation_fn) activation_fn->initialize_weight(n_in, this->weight);
}





void FullyConnectedLayer::feedforward(UINT idx, const rcube &input) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	Util::printCube(input, "input:");
	Util::convertCube(input, this->input);
	Util::printCube(this->input, "converted input:");

	//Util::dropoutLayer(this->input, this->p_dropout);
	//Util::printCube(this->input, "dropped out:");

	Util::printMat(weight, "weight:");
	Util::printVec(bias, "bias:");

	z.slice(0) = weight*this->input.slice(0) + bias;
	Util::printCube(z, "z:");

	activation_fn->activate(z, output);
	Util::printCube(output, "output:");

	Layer::feedforward(idx, this->output);
}

void FullyConnectedLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	if(!isLastNextLayerRequest(idx)) throw Exception();


	/*
	FullyConnectedLayer *fc_layer = dynamic_cast<FullyConnectedLayer *>(next_layer);
	if(fc_layer) {
		delta.slice(0) = fc_layer->getDelta().slice(0);
	}
	else {
		throw Exception();
		// TODO fc 다음으로 CONV가 온 경우 처리
		//delta.slice(0) = next_delta.slice(0) % sp.slice(0);
	}
	*/

	rcube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);

	Util::printMat(w_next_delta.slice(0), "w_next_delta");
	//Util::printMat(next_w->t(), "next_w");
	//Util::printMat(next_delta.slice(0), "next_delta");

	rcube sp;
	activation_fn->d_activate(output, sp);

	// delta l = dC/dz
	delta.slice(0) = w_next_delta.slice(0) % sp.slice(0);
	Util::printMat(delta.slice(0), "delta:");

	nabla_b += delta.slice(0);
	// delta lw = dC/dw
	nabla_w += delta.slice(0)*input.slice(0).t();



	// delta lx = dC/dx
	delta_input.slice(0) = weight.t()*delta.slice(0);
	//fc_layer->getWeight().t()*fc_layer->getDelta().slice(0)

	HiddenLayer::backpropagation(idx, this);
	delta.zeros();
	delta_input.zeros();

}




void FullyConnectedLayer::reset_nabla(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	nabla_b.zeros();
	nabla_w.zeros();

	Layer::reset_nabla(idx);
}


void FullyConnectedLayer::update(UINT idx, double eta, double lambda, int n, int miniBatchSize) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	weight = (1-eta*lambda/n)*weight - (eta/miniBatchSize)*nabla_w;
	bias -= eta/miniBatchSize*nabla_b;

	Layer::update(idx, eta, lambda, n, miniBatchSize);
}































