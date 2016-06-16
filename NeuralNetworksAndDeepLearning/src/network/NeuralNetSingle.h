/*
 * NeuralNetSingle.h
 *
 *  Created on: 2016. 6. 2.
 *      Author: jhkim
 */

#ifndef NETWORK_NEURALNETSINGLE_H_
#define NETWORK_NEURALNETSINGLE_H_


#if CPU_MODE


class NeuralNetSingle : public Network {
public:
	NeuralNetSingle() : Network() {
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
				param_filler(ParamFillerType::Gaussian, 1),
				ActivationType::ReLU
				);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax",
				100,
				10,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Constant, 0),
				param_filler(ParamFillerType::Gaussian, 1)
				);

		Network::addLayerRelation(inputLayer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~NeuralNetSingle() {}
};


#else


#endif



#endif /* NETWORK_NEURALNETSINGLE_H_ */
