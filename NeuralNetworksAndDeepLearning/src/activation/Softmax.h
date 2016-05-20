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
		// TODO for debug
		//weight.randn();
		weight.fill(0.0);
	}
	/*
	void initialize_weight(int filters, void *weight, int type) {
		if(type) {
			mat *w = (mat *)weight;
			w->fill(0.0);
		}
		else {
			cube *w = (cube *)weight;
			for(int i = 0; i < filters; i++) {
				w[i].fill(0.0);
			}
		}
	}
	*/
	void activate(const cube &z, cube &activation) {
		// TODO softmax는 output layer only,
		// vector형태의 output을 전제
		//Util::printCube(z, "z:");
		//cout << "maxz: " << max(max(z.slice(0)));
		cube temp = exp(z-max(max(z.slice(0))));
		//Util::printCube(temp, "temp:");
		activation = temp / accu(temp);
		//Util::printCube(activation, "activation:");
	}
	void d_activate(const cube &activation, cube &da) {
		//Util::printCube(activation, "activation:");
		da = activation % (1.0 - activation);
		//Util::printCube(da, "da:");
	}
};

#endif /* ACTIVATION_SOFTMAX_H_ */
