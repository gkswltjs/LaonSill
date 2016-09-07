/*
 * SoftmaxLayer.cpp
 *
 *  Created on: 2016. 8. 1.
 *      Author: jhkim
 */

#include "SoftmaxLayer.h"




template <typename Dtype>
SoftmaxLayer<Dtype>::SoftmaxLayer() {
	this->type = Layer<Dtype>::Softmax;
}

template <typename Dtype>
SoftmaxLayer<Dtype>::SoftmaxLayer(Builder* builder)
	: OutputLayer<Dtype>(builder) {
	initialize();
}

template <typename Dtype>
SoftmaxLayer<Dtype>::SoftmaxLayer(
		const string name,
		int n_out,
		double p_dropout,
		update_param weight_update_param,
		update_param bias_update_param,
		param_filler<Dtype> weight_filler,
		param_filler<Dtype> bias_filler)
	: OutputLayer<Dtype>(name, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler,
			Activation<Dtype>::Softmax, Cost<Dtype>::LogLikelihood) {
	initialize();
}

template <typename Dtype>
SoftmaxLayer<Dtype>::~SoftmaxLayer() {}



template <typename Dtype>
void SoftmaxLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::Softmax;

	//this->cost_fn = CostFactory::create(Cost<Dtype>::LogLikelihood);
	//this->activation_fn = ActivationFactory::create(Activation::Softmax);
	//this->activation_fn->initialize_weight(in_dim.size(), weight);

	//weight.zeros();
	//bias.zeros();
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::_shape(bool recursive) {
	if(recursive) {
		OutputLayer<Dtype>::_shape();
	}
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::_clearShape() {
	OutputLayer<Dtype>::_clearShape();
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::_load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*>& layerMap) {
	OutputLayer<Dtype>::_load(ifs, layerMap);
	initialize();
	SoftmaxLayer<Dtype>::_shape(false);
}






#ifndef GPU_MODE
template <typename Dtype>
void SoftmaxLayer<Dtype>::cost(const rvec &target) {
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








template class SoftmaxLayer<float>;









