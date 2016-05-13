/*
 * InputLayer.h
 *
 *  Created on: 2016. 5. 11.
 *      Author: jhkim
 */

#ifndef LAYER_INPUTLAYER_H_
#define LAYER_INPUTLAYER_H_

#include "Layer.h"
#include <armadillo>

using namespace arma;


class InputLayer : public Layer {
public:
	InputLayer(int n_in)
		: Layer(n_in, n_in) {
		this->activation.set_size(n_in, 1);
	}
	virtual ~InputLayer() {}

	virtual void feedforward(const vec &input) {
		this->activation = input;
	}
};

#endif /* LAYER_INPUTLAYER_H_ */
