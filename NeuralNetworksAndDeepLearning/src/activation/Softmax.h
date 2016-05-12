/*
 * Softmax.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef ACTIVATION_SOFTMAX_H_
#define ACTIVATION_SOFTMAX_H_

#include <Activation.h>
#include "../Util.h"

using namespace arma;

class Softmax : public Activation {
public:
	Softmax() {}
	virtual ~Softmax() {}

	void activate(const vec &z, vec &activation) {
		vec temp = exp(z);
		activation = temp/accu(temp);
	}
	void d_activate(const vec &activation, vec &da) {
		da = activation%(1.0-activation);
	}
};

#endif /* ACTIVATION_SOFTMAX_H_ */
