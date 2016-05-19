/*
 * SoftmaxLayer.h
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
	SoftmaxLayer(int n_in, int n_out, double p_dropout)
		: OutputLayer(n_in, n_out, p_dropout) {
		initialize();
	}
	SoftmaxLayer(io_dim in_dim, io_dim out_dim, double p_dropout)
		: OutputLayer(in_dim, out_dim, p_dropout) {
		initialize();
	}
	virtual ~SoftmaxLayer() {
		if(cost_fn) delete cost_fn;
		if(activation_fn) delete activation_fn;
	}

	void cost(const vec &target) {
		cost_fn->d_cost(z, output, target, delta);

		Util::printVec(nabla_b, "bias:");
		Util::printMat(nabla_w, "weight");
		Util::printCube(delta, "delta:");
		Util::printCube(input, "input:");

		nabla_b += delta.slice(0);
		nabla_w += delta.slice(0)*input.slice(0).t();
	}

private:
	void initialize() {
		this->cost_fn = new LogLikelihoodCost();
		this->activation_fn = new Softmax();
		this->activation_fn->initialize_weight(in_dim.rows, weight);

		Util::printMat(weight, "weight:");
	}
};

#endif /* LAYER_SOFTMAXLAYER_H_ */
