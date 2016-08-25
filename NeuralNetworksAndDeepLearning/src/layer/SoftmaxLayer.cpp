/*
 * SoftmaxLayer.cpp
 *
 *  Created on: 2016. 8. 1.
 *      Author: jhkim
 */

#include "SoftmaxLayer.h"





SoftmaxLayer::SoftmaxLayer() {
	this->type = Layer::Softmax;
}

SoftmaxLayer::SoftmaxLayer(Builder* builder)
	: OutputLayer(builder) {
	initialize();
}

SoftmaxLayer::SoftmaxLayer(const string name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler)
	: OutputLayer(name, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler,
			Activation::Softmax, Cost::LogLikelihood) {
	initialize();
}

SoftmaxLayer::~SoftmaxLayer() {}




void SoftmaxLayer::initialize() {
	this->type = Layer::Softmax;

	//this->cost_fn = CostFactory::create(Cost::LogLikelihood);
	//this->activation_fn = ActivationFactory::create(Activation::Softmax);
	//this->activation_fn->initialize_weight(in_dim.size(), weight);

	//weight.zeros();
	//bias.zeros();
}

void SoftmaxLayer::_shape(bool recursive) {
	if(recursive) {
		OutputLayer::_shape();
	}
}

void SoftmaxLayer::_clearShape() {
	OutputLayer::_clearShape();
}

void SoftmaxLayer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	OutputLayer::_load(ifs, layerMap);
	initialize();
	SoftmaxLayer::_shape(false);
}






#ifndef GPU_MODE
void SoftmaxLayer::cost(const rvec &target) {
	// delta
	cost_fn->d_cost(z, output, target, delta);
	Util::printVec(nabla_b, "bias:");
	Util::printMat(nabla_w, "weight");
	Util::printCube(delta, "delta:");
	Util::printCube(input, "input:");
	nabla_b += delta.slice(0);
	// delta weight
	nabla_w += delta.slice(0)*input.slice(0).t();

	// delta input
	delta_input.slice(0) = weight.t()*delta.slice(0);

	propBackpropagation();
}
#endif


















