/*
 * Sigmoid.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef ACTIVATION_SIGMOID_H_
#define ACTIVATION_SIGMOID_H_

#include "Activation.h"

using namespace arma;


class Sigmoid : public Activation {
public:
	Sigmoid() {}
	virtual ~Sigmoid() {}

	void initialize_weight(int n_in, mat &weight) {
		weight.randn();
		weight *= 1 / sqrt(n_in);				// initial point scaling
	}
	void activate(const cube &z, cube &activation) {
		activation = 1 / (1.0 + exp(-1 * z));
	}
	void d_activate(const cube &activation, cube &da) {
		da = activation % (1.0 - activation);
	}
};

#endif /* ACTIVATION_SIGMOID_H_ */
