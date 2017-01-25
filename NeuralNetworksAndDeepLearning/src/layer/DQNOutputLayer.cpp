/**
 * @file DQNOutputLayer.cpp
 * @date 2016-12-26
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "DQNOutputLayer.h"

using namespace std;

template <typename Dtype>
DQNOutputLayer<Dtype>::DQNOutputLayer() {
	this->type = Layer<Dtype>::DQNOutput;
}

template <typename Dtype>
DQNOutputLayer<Dtype>::DQNOutputLayer(Builder* builder)
	: OutputLayer<Dtype>(builder) {
	initialize();
}

template <typename Dtype>
DQNOutputLayer<Dtype>::DQNOutputLayer(
		const string name,
		int n_out,
		double p_dropout,
		update_param weight_update_param,
		update_param bias_update_param,
		param_filler<Dtype> weight_filler,
		param_filler<Dtype> bias_filler)
	: OutputLayer<Dtype>(name, n_out, p_dropout, weight_update_param, bias_update_param,
            weight_filler, bias_filler, Cost<Dtype>::DQN) {
	initialize();
}

template <typename Dtype>
DQNOutputLayer<Dtype>::~DQNOutputLayer() {}



template <typename Dtype>
void DQNOutputLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::DQNOutput;

	//this->cost_fn = CostFactory::create(Cost<Dtype>::LogLikelihood);
	//this->activation_fn = ActivationFactory::create(Activation::Softmax);
	//this->activation_fn->initialize_weight(in_dim.size(), weight);

	//weight.zeros();
	//bias.zeros();
}

#ifndef GPU_MODE
template <typename Dtype>
void DQNOutputLayer<Dtype>::cost(const rvec &target) {
	// delta
	cost_fn->backward(z, output, target, delta);
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

template class DQNOutputLayer<float>;
