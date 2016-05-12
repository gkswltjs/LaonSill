/*
 * SoftmaxLayer.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef LAYER_SOFTMAXLAYER_H_
#define LAYER_SOFTMAXLAYER_H_

#include "OutputLayer.h"
#include "../cost/Cost.h"
#include <armadillo>

using namespace arma;


class SoftmaxLayer : public OutputLayer {
public:
	SoftmaxLayer(int n_in, int n_out, Cost *cost_fn)
		: OutputLayer(n_in, n_out) {
		this->cost_fn = cost_fn;
		this->activation_fn = new Sigmoid();
	}
	virtual ~SoftmaxLayer() {
		if(activation_fn) delete activation_fn;
	}

	void cost(const vec &target, const vec &input) {
		cost_fn->d_cost(z, activation, target, delta);

		nabla_b += delta;
		nabla_w += delta*input.t();
	}
};

#endif /* LAYER_SOFTMAXLAYER_H_ */
