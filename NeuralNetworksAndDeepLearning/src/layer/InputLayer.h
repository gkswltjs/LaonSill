/*
 * InputLayer.h
 *
 *  Created on: 2016. 5. 11.
 *      Author: jhkim
 */

#ifndef LAYER_INPUTLAYER_H_
#define LAYER_INPUTLAYER_H_

#include "Layer.h"
#include "LayerConfig.h"
#include <armadillo>

using namespace arma;


class InputLayer : public Layer {
public:
	InputLayer(int n_in) : Layer(n_in, n_in) {}
	InputLayer(io_dim in_dim) : Layer(in_dim, in_dim) {}
	virtual ~InputLayer() {}

	virtual void feedforward(const cube &input) {
		convertInputDim(input, this->output);
	}
protected:
};

#endif /* LAYER_INPUTLAYER_H_ */
