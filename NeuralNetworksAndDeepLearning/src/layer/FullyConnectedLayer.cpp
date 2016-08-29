/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"
#include "../Util.h"
#include "../exception/Exception.h"


template <typename Dtype>
FullyConnectedLayer<Dtype>::FullyConnectedLayer() {
	this->type = Layer<Dtype>::FullyConnected;
}

template <typename Dtype>
FullyConnectedLayer<Dtype>::FullyConnectedLayer(Builder* builder)
	: HiddenLayer<Dtype>(builder) {
	initialize(builder->_nOut, builder->_pDropout, builder->_weightUpdateParam, builder->_biasUpdateParam,
			builder->_weightFiller, builder->_biasFiller, builder->_activationType);
}

template <typename Dtype>
FullyConnectedLayer<Dtype>::FullyConnectedLayer(const string name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, Activation::Type activationType)
	: HiddenLayer<Dtype>(name) {
	initialize(n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::initialize(int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, Activation::Type activationType) {
	Cuda::refresh();

	// out_dim의 batches는 _shape()에서 in_dim값에 따라 결정된다.
	this->out_dim = io_dim(n_out, 1, 1, 1);
	this->type = Layer<Dtype>::FullyConnected;

	this->p_dropout = p_dropout;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;

	this->activation_fn = ActivationFactory::create(activationType);
	this->scale = 1. / (1. - p_dropout);

	this->_params.resize(2);
	this->_params[ParamType::Weight] = new Data<Dtype>();			// weight
	this->_params[ParamType::Bias] = new Data<Dtype>();			// bias

	this->_paramsHistory.resize(2);
	this->_paramsHistory[ParamType::Weight] = new Data<Dtype>();	// weight history
	this->_paramsHistory[ParamType::Bias] = new Data<Dtype>(); 	// bias history

	this->_preActivation = new Data<Dtype>();						// weighted sum (pre activation)
}

template <typename Dtype>
double FullyConnectedLayer<Dtype>::sumSquareParamsData() {
	double result = 0.0;
	for(uint32_t i = 0; i < _params.size(); i++) {
		result += _params[i]->sumsq_device_data();
	}
	return result;
}

template <typename Dtype>
double FullyConnectedLayer<Dtype>::sumSquareParamsGrad() {
	double result = 0.0;
	for(uint32_t i = 0; i < _params.size(); i++) {
		result += _params[i]->sumsq_device_grad();
	}
	return result;
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::scaleParamsGrad(DATATYPE scale) {
	for(uint32_t i = 0; i < _params.size(); i++) {
		_params[i]->scale_device_grad(scale);
	}
}




template <typename Dtype>
void FullyConnectedLayer<Dtype>::_save(ofstream &ofs) {
	HiddenLayer<Dtype>::_save(ofs);

	int activationType = (int)activation_fn->getType();

	ofs.write((char *)&this->out_dim.rows, sizeof(uint32_t));
	ofs.write((char *)&p_dropout, sizeof(double));
	ofs.write((char *)&activationType, sizeof(int));
	ofs.write((char *)&weight_update_param, sizeof(update_param));
	ofs.write((char *)&bias_update_param, sizeof(update_param));
	ofs.write((char *)&weight_filler, sizeof(param_filler));
	ofs.write((char *)&bias_filler, sizeof(param_filler));


	//checkCudaErrors(cudaMemcpyAsync(weight, d_weight, sizeof(Dtype)*out_dim.unitsize()*in_dim.unitsize(), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpyAsync(bias, d_bias, sizeof(Dtype)*out_dim.unitsize(), cudaMemcpyDeviceToHost));

	const Dtype* weight = _params[ParamType::Weight]->host_data();
	const Dtype* bias = _params[ParamType::Bias]->host_data();
	ofs.write((char *)weight, sizeof(Dtype)*this->out_dim.unitsize()*this->in_dim.unitsize());
	ofs.write((char *)bias, sizeof(Dtype)*this->out_dim.unitsize());
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_load(ifstream& ifs, map<Layer<Dtype>*, Layer<Dtype>*>& layerMap) {
	HiddenLayer<Dtype>::_load(ifs, layerMap);

	uint32_t n_out = 0;
	double p_dropout;
	Activation::Type activationType;
	update_param weight_update_param, bias_update_param;
	param_filler weight_filler, bias_filler;

	ifs.read((char *)&n_out, sizeof(uint32_t));
	ifs.read((char *)&p_dropout, sizeof(double));
	ifs.read((char *)&activationType, sizeof(int));
	ifs.read((char *)&weight_update_param, sizeof(update_param));
	ifs.read((char *)&bias_update_param, sizeof(update_param));
	ifs.read((char *)&weight_filler, sizeof(param_filler));
	ifs.read((char *)&bias_filler, sizeof(param_filler));

	initialize(n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
	FullyConnectedLayer<Dtype>::_shape(false);

	Dtype* weight = _params[ParamType::Weight]->mutable_host_data();
	Dtype* bias = _params[ParamType::Bias]->mutable_host_data();
	// initialize() 내부에서 weight, bias를 초기화하므로 initialize() 후에 weight, bias load를 수행해야 함
	ifs.read((char *)weight, sizeof(Dtype)*this->out_dim.unitsize()*this->in_dim.unitsize());
	ifs.read((char *)bias, sizeof(Dtype)*this->out_dim.unitsize());

}










#ifndef GPU_MODE
template <typename Dtype>
FullyConnectedLayer<Dtype>::~FullyConnectedLayer() {
	ActivationFactory::destory(activation_fn);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer<Dtype>::_load(ifs, layerMap);

	double p_dropout;
	ifs.read((char *)&p_dropout, sizeof(double));

	Activation::Type activationType;
	ifs.read((char *)&activationType, sizeof(int));

	update_param weight_update_param;
	ifs.read((char *)&weight_update_param, sizeof(update_param));

	update_param bias_update_param;
	ifs.read((char *)&bias_update_param, sizeof(update_param));

	param_filler weight_filler;
	ifs.read((char *)&weight_filler, sizeof(param_filler));

	param_filler bias_filler;
	ifs.read((char *)&bias_filler, sizeof(param_filler));

	initialize(p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);

	// initialize() 내부에서 weight, bias를 초기화하므로 initialize() 후에 weight, bias load를 수행해야 함


	weight.load(ifs, file_type::arma_binary);
	//weight.print("load-weight:");
	bias.load(ifs, file_type::arma_binary);
	//bias.print("load-bias:");

}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::initialize(double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, Activation::Type activationType) {
	this->type = Layer<Dtype>::FullyConnected;
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

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_save(ofstream &ofs) {
	HiddenLayer<Dtype>::_save(ofs);

	ofs.write((char *)&p_dropout, sizeof(double));

	int activationType = (int)activation_fn->getType();
	ofs.write((char *)&activationType, sizeof(int));
	ofs.write((char *)&weight_update_param, sizeof(update_param));
	ofs.write((char *)&bias_update_param, sizeof(update_param));
	ofs.write((char *)&weight_filler, sizeof(param_filler));
	ofs.write((char *)&bias_filler, sizeof(param_filler));
	//ofs.write((char *)&weight, sizeof(rmat));
	//ofs.write((char *)&bias, sizeof(rvec));

	//weight.print("save-weight:");
	weight.save(ofs, file_type::arma_binary);
	//bias.print("save-bias:");
	bias.save(ofs, file_type::arma_binary);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::_feedforward() {
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

	activation_fn->forward(z, output);
	Util::printCube(output, "output:");

	propFeedforward(this->output, end);
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::backpropagation(uint32_t idx, HiddenLayer *next_layer) {
	if(!isLastNextLayerRequest(idx)) throw Exception();


	rcube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);

	Util::printMat(w_next_delta.slice(0), "w_next_delta");
	//Util::printMat(next_w->t(), "next_w");
	//Util::printMat(next_delta.slice(0), "next_delta");

	rcube sp;
	activation_fn->backward(output, sp);

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

template <typename Dtype>
void FullyConnectedLayer<Dtype>::reset_nabla(uint32_t idx) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	nabla_b.zeros();
	nabla_w.zeros();

	propResetNParam();
}

template <typename Dtype>
void FullyConnectedLayer<Dtype>::update(uint32_t idx, uint32_t n, uint32_t miniBatchSize) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	//weight = (1-eta*lambda/n)*weight - (eta/miniBatchSize)*nabla_w;
	//bias -= eta/miniBatchSize*nabla_b;

	weight = (1-weight_update_param.lr_mult*weight_update_param.decay_mult/n)*weight - (weight_update_param.lr_mult/miniBatchSize)*nabla_w;
	bias -= bias_update_param.lr_mult/miniBatchSize*nabla_b;

	propUpdate(n, miniBatchSize);
}

#endif



template class FullyConnectedLayer<float>;

