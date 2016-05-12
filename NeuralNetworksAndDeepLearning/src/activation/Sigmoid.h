/*
 * Sigmoid.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef ACTIVATION_SIGMOID_H_
#define ACTIVATION_SIGMOID_H_

#include <Activation.h>

using namespace arma;


class Sigmoid : public Activation {
public:
	Sigmoid() {};
	virtual ~Sigmoid() {};

	void activate(const vec &z, vec &activation) {
		vec temp = ones<vec>(z.n_rows);
		activation = temp / (1.0+exp(-1*z));
	}
	void d_activate(const vec &activation, vec &da) {
		da = activation%(1.0-activation);
	}
};

#endif /* ACTIVATION_SIGMOID_H_ */
