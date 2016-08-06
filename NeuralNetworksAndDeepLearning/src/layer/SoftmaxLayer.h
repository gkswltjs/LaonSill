/*
 * SoftmaxLayer.h
 *
 *  Created on: 2016. 8. 1.
 *      Author: jhkim
 */

#ifndef SOFTMAXLAYER_H_
#define SOFTMAXLAYER_H_

#include "OutputLayer.h"
#include "../cost/LogLikelihoodCost.h"
#include "../activation/Softmax.h"
#include "../exception/Exception.h"
#include <armadillo>

using namespace arma;









class SoftmaxLayer : public OutputLayer {
public:
	SoftmaxLayer();
	SoftmaxLayer(const char *name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler);
	virtual ~SoftmaxLayer();

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

#if CPU_MODE
public:
	SoftmaxLayer(const char *name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler)
		: OutputLayer(name, n_in, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler,
				ActivationType::Softmax, CostType::LogLikelihood) {
		initialize();
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

#else
public:
	void cost(const UINT *target);
#endif


protected:

#if CPU_MODE
protected:
	void initialize() {
		this->type = LayerType::Softmax;

		//this->cost_fn = CostFactory::create(CostType::LogLikelihood);
		//this->activation_fn = ActivationFactory::create(ActivationType::Softmax);
		//this->activation_fn->initialize_weight(in_dim.size(), weight);

		//weight.zeros();
		//bias.zeros();
	}
#else
protected:
	void initialize();
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
#endif

};





#endif /* SOFTMAXLAYER_H_ */
