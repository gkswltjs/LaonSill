/*
 * InceptionNetAux.h
 *
 *  Created on: 2016. 6. 2.
 *      Author: jhkim
 */

#ifndef NETWORK_INCEPTIONNETAUX_H_
#define NETWORK_INCEPTIONNETAUX_H_



class InceptionNetAux : public Network {
public:
	InceptionNetAux() : Network(0, 0, 0) {
		double lr_mult = 0.1;
		double decay_mult = 5.0;

		InputLayer *inputLayer = new InputLayer(
				"input",
				io_dim(28, 28, 1)
				);

		HiddenLayer *conv1Layer = new ConvLayer(
				"conv1",
				io_dim(28, 28, 1),
				filter_dim(5, 5, 1, 40, 1),
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *pool1Layer = new PoolingLayer(
				"pool1",
				io_dim(28, 28, 40),
				pool_dim(3, 3, 2),
				PoolingType::Max
				);

		HiddenLayer *incept1Layer = new InceptionLayer(
				"incept1",
				io_dim(14, 14, 40),
				io_dim(14, 14, 60),
				10, 15, 30, 5, 10, 10
				);

		HiddenLayer *fc1Layer = new FullyConnectedLayer(
				"fc1",
				14*14*60,
				100,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU);

		OutputLayer *softmaxLayer = new SoftmaxLayer("softmax",
				100,
				10,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1)
				);

		Network::addLayerRelation(inputLayer, conv1Layer);
		Network::addLayerRelation(conv1Layer, pool1Layer);
		Network::addLayerRelation(pool1Layer, incept1Layer);
		Network::addLayerRelation(incept1Layer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~InceptionNetAux() {}
};

#endif /* NETWORK_INCEPTIONNETAUX_H_ */
