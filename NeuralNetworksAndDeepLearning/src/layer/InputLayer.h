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
#include "../Util.h"
#include "../exception/Exception.h"
#include <armadillo>

using namespace arma;


class InputLayer : public Layer {
public:
	InputLayer(string name, int n_in) : Layer(name, n_in, n_in) {
		initialize();
	}
	InputLayer(string name, io_dim in_dim) : Layer(name, in_dim, in_dim) {
		initialize();
	}
	virtual ~InputLayer() {}

	/**
	 * Input 무조건 첫번째 layer,
	 * feedforward로 들어오는 input외의 input에 대해서는 고려하지 않음
	 */
	void feedforward(UINT idx, const rcube &input) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();

		Util::convertCube(input, this->input);
		Util::convertCube(this->input, this->output);
		Util::printCube(input, "input:");
		Util::printCube(this->output, "output:");

		propFeedforward(this->output);
	}

	void save(int idx, ofstream &ofs) {






		propSave(ofs);
	}

protected:
	void initialize() {
		this->type = LayerType::Input;
	}
};

#endif /* LAYER_INPUTLAYER_H_ */
