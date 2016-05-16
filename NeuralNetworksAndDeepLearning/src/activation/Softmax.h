/*
 * Softmax.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef ACTIVATION_SOFTMAX_H_
#define ACTIVATION_SOFTMAX_H_

#include "Activation.h"
#include "../Util.h"

using namespace arma;

class Softmax : public Activation {
public:
	Softmax() {}
	virtual ~Softmax() {}

	void initialize_weight(int n_in, mat &weight) {
		weight.fill(0.0);
	}
	void activate(const cube &z, cube &activation) {
		cube temp = exp(z);
		activation = temp / accu(temp);
	}
	void d_activate(const cube &activation, cube &da) {
		da = activation % (1.0 - activation);
	}
};

#endif /* ACTIVATION_SOFTMAX_H_ */
