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
#include <armadillo>

using namespace arma;



class SoftmaxLayer : public OutputLayer {
public:
	SoftmaxLayer(string name, int n_in, int n_out, double p_dropout)
		: OutputLayer(name, n_in, n_out, p_dropout) {
		initialize();
	}
	SoftmaxLayer(string name, io_dim in_dim, io_dim out_dim, double p_dropout)
		: OutputLayer(name, in_dim, out_dim, p_dropout) {
		initialize();
	}
	virtual ~SoftmaxLayer() {
		if(cost_fn) delete cost_fn;
		if(activation_fn) delete activation_fn;
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


		HiddenLayer::backpropagation(0, this);
	}

private:
	void initialize() {
		this->cost_fn = new LogLikelihoodCost();
		this->activation_fn = new Softmax();
		this->activation_fn->initialize_weight(in_dim.size(), weight);

	}
};

#endif /* LAYER_SOFTMAXLAYER_H_ */
