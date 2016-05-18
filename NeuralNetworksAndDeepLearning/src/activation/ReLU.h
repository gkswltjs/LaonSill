/*
 * ReLU.h
 *
 *  Created on: 2016. 5. 18.
 *      Author: jhkim
 */

#ifndef ACTIVATION_RELU_H_
#define ACTIVATION_RELU_H_

#include "Activation.h"

using namespace arma;

class ReLU : public Activation {
public:
	ReLU() {}
	virtual ~ReLU() {}

	void initialize_weight(int n_in, mat &weight) {
		weight.randn();
		weight *= 1 / sqrt(n_in);				// initial point scaling
	}
	void activate(const cube &z, cube &activation) {
		zero.set_size(size(z));
		zero.fill(0.0);
		activation = max(z, zero);
	}
	void d_activate(const cube &activation, cube &da) {
		Util::printCube(activation, "d_activate-activation:");
		zero.set_size(size(activation));
		zero.fill(0.0);
		da = conv_to<cube>::from(activation > zero);
		Util::printCube(da, "d_activate-da:");
	}

private:
	cube zero;

};

#endif /* ACTIVATION_RELU_H_ */
