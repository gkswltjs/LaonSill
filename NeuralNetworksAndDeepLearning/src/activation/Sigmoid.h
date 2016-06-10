/*
 * Sigmoid.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef ACTIVATION_SIGMOID_H_
#define ACTIVATION_SIGMOID_H_

#include "Activation.h"


class Sigmoid : public Activation {
public:
	Sigmoid() {
		this->type = ActivationType::Sigmoid;
	}
	virtual ~Sigmoid() {}

	/*
	void initialize_weight(int n_in, rmat &weight) {
		weight.randn();
		weight *= 1.0/sqrt(n_in);				// initial point scaling
	}
	void initialize_weight(int n_in, rcube &weight) {
		weight.randn();
		weight *= 1.0/sqrt(n_in);				// initial point scaling
	}
	*/

	/*
	void initialize_weight(int filters, void *weight, int type) {
		if(type) {
			mat *w = (mat *)weight;
			w->randn();
			(*w) *= 1 / sqrt(w->n_rows);
		}
		else {
			cube *w = (cube *)weight;
			for(int i = 0; i < filters; i++) {
				w[i].randn();
				w[i] *= 1 / sqrt(w[i]);
			}
		}
	}
	*/
	void activate(const rcube &z, rcube &activation) {
		activation = 1.0 / (1.0 + exp(-1 * z));
	}
	void d_activate(const rcube &activation, rcube &da) {
		Util::printCube(activation, "d_activate-activation:");
		da = activation % (1.0 - activation);
		Util::printCube(da, "d_activate-da:");
	}
};

#endif /* ACTIVATION_SIGMOID_H_ */
