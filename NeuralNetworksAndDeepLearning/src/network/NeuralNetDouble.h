/*
 * NeuralNetDouble.h
 *
 *  Created on: 2016. 6. 2.
 *      Author: jhkim
 */

#ifndef NETWORK_NEURALNETDOUBLE_H_
#define NETWORK_NEURALNETDOUBLE_H_

#include "../activation/ReLU.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/SoftmaxLayer.h"
#include "Network.h"

class NeuralNetDouble : public Network {
public:
	NeuralNetDouble() : Network() {
		double lr_mult = 0.1;
		double decay_mult = 5.0;

		InputLayer *inputLayer = new InputLayer(
				"input",
				io_dim(28, 28, 1)
				);

		HiddenLayer *fc1Layer = new FullyConnectedLayer(
				"fc1",
				28*28*1,
				100,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *fc2Layer = new FullyConnectedLayer(
				"fc2",
				100,
				10,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax",
				10,
				10,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1)
				);

		Network::addLayerRelation(inputLayer, fc1Layer);
		Network::addLayerRelation(fc1Layer, fc2Layer);
		Network::addLayerRelation(fc2Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~NeuralNetDouble() {}
};

#endif /* NETWORK_NEURALNETDOUBLE_H_ */
