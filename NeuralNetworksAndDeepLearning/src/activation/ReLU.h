/*
 * ReLU.h
 *
 *  Created on: 2016. 5. 18.
 *      Author: jhkim
 */

#ifndef ACTIVATION_RELU_H_
#define ACTIVATION_RELU_H_

#include "Activation.h"



class ReLU : public Activation {
public:
	ReLU(io_dim activation_dim) {
		zero.set_size(activation_dim.rows, activation_dim.cols, activation_dim.channels);
		zero.zeros();
	}
	virtual ~ReLU() {}

	void initialize_weight(int n_in, rmat &weight) {
		weight.randn();
		weight *= 1.0 / n_in;				// initial point scaling
	}
	void initialize_weight(int n_in, rcube &weight) {
		weight.randn();
		weight *= 1.0 / n_in;				// initial point scaling
	}
	void activate(const rcube &z, rcube &activation) {
		activation = arma::max(z, zero);
	}
	void d_activate(const rcube &activation, rcube &da) {
		Util::printCube(activation, "d_activate-activation:");
		da = conv_to<rcube>::from(activation > zero);
		Util::printCube(da, "d_activate-da:");
	}

private:
	rcube zero;
};

#endif /* ACTIVATION_RELU_H_ */
