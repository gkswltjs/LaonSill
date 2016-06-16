/*
 * SigmoidLayer.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef LAYER_SIGMOIDLAYER_H_
#define LAYER_SIGMOIDLAYER_H_

#include "OutputLayer.h"
#include "../cost/Cost.h"
#include "../activation/Sigmoid.h"



#if CPU_MODE


class SigmoidLayer : public OutputLayer {
public:
	SigmoidLayer() {}
	SigmoidLayer(const char *name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, CostType costType)
		: OutputLayer(name, n_in, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, ActivationType::Sigmoid, costType) {
		initialize();
	}
	SigmoidLayer(const char *name, io_dim in_dim, io_dim out_dim, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, CostType costType)
		: OutputLayer(name, in_dim, out_dim, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, ActivationType::Sigmoid, costType) {
		initialize();
	}
	virtual ~SigmoidLayer() {
		//ActivationFactory::destory(activation_fn);
		//CostFactory::destroy(cost_fn);
	}

	void cost(const rvec &target) {
		cost_fn->d_cost(z, output, target, delta);

		nabla_b += delta.slice(0);
		nabla_w += delta.slice(0)*input.slice(0).t();

		propBackpropagation();
	}

	void save(UINT idx, ofstream &ofs) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		OutputLayer::save(ofs);
		propSave(ofs);
	}

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
		OutputLayer::load(ifs, layerMap);
		initialize();
	}

private:
	void initialize() {
		this->type = LayerType::Sigmoid;
		this->id = Layer::generateLayerId();

		//this->cost_fn = CostFactory::create(costType);
		//this->activation_fn = ActivationFactory::create(ActivationType::Sigmoid);
		//this->activation_fn->initialize_weight(in_dim.rows, weight);
	}
};



#else



#endif

#endif /* LAYER_SIGMOIDLAYER_H_ */
