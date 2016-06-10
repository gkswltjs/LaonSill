/*
 * SoftmaxLayer.h
 *
 * C = -ln ay
 * dai/dzj =
 * 			ai*(1-ai) 	: when i = j
 * 			-ai*aj		: when i != j
 *
 * dC/dz = dC/da * da/dz
 * dC/da = -1/a
 *
 * dC/dz = -1/a *
 * 			ai*(1-ai) = ai-1	: when i = j
 * 			-ai*aj = aj-0		: when i != j
 *
 * dC/dz = a - y
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef LAYER_SOFTMAXLAYER_H_
#define LAYER_SOFTMAXLAYER_H_

#include "OutputLayer.h"
#include "../cost/LogLikelihoodCost.h"
#include "../activation/Softmax.h"
#include "../exception/Exception.h"
#include <armadillo>

using namespace arma;



class SoftmaxLayer : public OutputLayer {
public:
	SoftmaxLayer() {}
	SoftmaxLayer(const char *name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler)
		: OutputLayer(name, n_in, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler) {
		initialize();
	}
	SoftmaxLayer(const char *name, io_dim in_dim, io_dim out_dim, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler)
		: OutputLayer(name, in_dim, out_dim, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler) {
		initialize();
	}
	virtual ~SoftmaxLayer() {
		ActivationFactory::destory(activation_fn);
		CostFactory::destroy(cost_fn);
	}

	void cost(const rvec &target) {
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
		this->type = LayerType::Softmax;
		this->id = Layer::generateLayerId();

		this->cost_fn = CostFactory::create(CostType::LogLikelihood);
		this->activation_fn = ActivationFactory::create(ActivationType::Softmax);
		//this->activation_fn->initialize_weight(in_dim.size(), weight);

		//weight.zeros();
		//bias.zeros();
	}


};

#endif /* LAYER_SOFTMAXLAYER_H_ */
















