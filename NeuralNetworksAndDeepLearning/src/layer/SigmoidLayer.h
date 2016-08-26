/**
 * @file SigmoidLayer.h
 * @date 2016/5/12
 * @author jhkim
 * @brief
 * @details
 */


#ifndef LAYER_SIGMOIDLAYER_H_
#define LAYER_SIGMOIDLAYER_H_

#include "OutputLayer.h"
#include "../cost/Cost.h"
#include "../activation/Sigmoid.h"



#ifndef GPU_MODE


class SigmoidLayer : public OutputLayer {
public:
	SigmoidLayer() { this->type = Layer::Sigmoid; }
	SigmoidLayer(const string name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, Cost::Type costType)
		: OutputLayer(name, n_in, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, Activation::Sigmoid, costType) {
		initialize();
	}
	SigmoidLayer(const string name, io_dim in_dim, io_dim out_dim, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, Cost::Type costType)
		: OutputLayer(name, in_dim, out_dim, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, Activation::Sigmoid, costType) {
		initialize();
	}
	virtual ~SigmoidLayer() {
		//ActivationFactory::destory(activation_fn);
		//CostFactory::destroy(cost_fn);
	}

	void cost(const rvec &target) {
		cost_fn->backward(z, output, target, delta);

		nabla_b += delta.slice(0);
		nabla_w += delta.slice(0)*input.slice(0).t();

		propBackpropagation();
	}

	void _load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
		OutputLayer::_load(ifs, layerMap);
		initialize();
	}

private:
	void initialize() {
		this->type = Layer::Sigmoid;

		//this->cost_fn = CostFactory::create(costType);
		//this->activation_fn = ActivationFactory::create(Activation::Sigmoid);
		//this->activation_fn->initialize_weight(in_dim.rows, weight);
	}
};



#else



#endif

#endif /* LAYER_SIGMOIDLAYER_H_ */
