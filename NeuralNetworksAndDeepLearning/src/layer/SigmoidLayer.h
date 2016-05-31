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


class SigmoidLayer : public OutputLayer {
public:
	SigmoidLayer(string name, int n_in, int n_out, double p_dropout, Cost *cost_fn)
		: OutputLayer(name, n_in, n_out, p_dropout) {
		initialize(cost_fn);
	}
	SigmoidLayer(string name, io_dim in_dim, io_dim out_dim, double p_dropout, Cost *cost_fn)
		: OutputLayer(name, in_dim, out_dim, p_dropout) {
		initialize(cost_fn);
	}
	virtual ~SigmoidLayer() {
		if(activation_fn) delete activation_fn;
	}

	void cost(const vec &target) {
		cost_fn->d_cost(z, output, target, delta);

		nabla_b += delta.slice(0);
		nabla_w += delta.slice(0)*input.slice(0).t();

		HiddenLayer::backpropagation(0, this);
	}

private:
	void initialize(Cost *cost_fn) {
		this->cost_fn = cost_fn;
		this->activation_fn = new Sigmoid();
		this->activation_fn->initialize_weight(in_dim.rows, weight);
	}
};

#endif /* LAYER_SIGMOIDLAYER_H_ */
