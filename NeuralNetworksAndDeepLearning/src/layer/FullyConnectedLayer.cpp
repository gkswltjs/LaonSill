/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"
#include "../Util.h"
#include "../exception/Exception.h"

FullyConnectedLayer::FullyConnectedLayer(string name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, ActivationType activationType)
	: HiddenLayer(name, n_in, n_out) {
	initialize(p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
}

FullyConnectedLayer::FullyConnectedLayer(string name, io_dim in_dim, io_dim out_dim, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, ActivationType activationType)
	: HiddenLayer(name, in_dim, out_dim) {
	initialize(p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
}

FullyConnectedLayer::~FullyConnectedLayer() {
	ActivationFactory::destory(activation_fn);
}


void FullyConnectedLayer::initialize(double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, ActivationType activationType) {
	this->type = LayerType::FullyConnected;

	this->p_dropout = p_dropout;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;


	int n_in = in_dim.size();
	int n_out = out_dim.size();

	this->bias.set_size(n_out, 1);
	//this->bias.zeros();
	this->bias_filler.fill(this->bias, n_in);

	this->weight.set_size(n_out, n_in);
	this->weight_filler.fill(this->weight, n_in);


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
	this->activation_fn = ActivationFactory::create(activationType);
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

	propFeedforward(this->output);
}

void FullyConnectedLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	if(!isLastNextLayerRequest(idx)) throw Exception();


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

	propBackpropagation();
	delta.zeros();
	delta_input.zeros();

}




void FullyConnectedLayer::reset_nabla(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	nabla_b.zeros();
	nabla_w.zeros();

	propResetNParam();
}


void FullyConnectedLayer::update(UINT idx, int n, int miniBatchSize) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	//weight = (1-eta*lambda/n)*weight - (eta/miniBatchSize)*nabla_w;
	//bias -= eta/miniBatchSize*nabla_b;

	weight = (1-weight_update_param.lr_mult*weight_update_param.decay_mult/n)*weight - (weight_update_param.lr_mult/miniBatchSize)*nabla_w;
	bias -= bias_update_param.lr_mult/miniBatchSize*nabla_b;

	propUpdate(n, miniBatchSize);
}































