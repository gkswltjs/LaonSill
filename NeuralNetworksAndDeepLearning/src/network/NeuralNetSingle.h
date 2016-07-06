/*
 * NeuralNetSingle.h
 *
 *  Created on: 2016. 6. 2.
 *      Author: jhkim
 */

#ifndef NETWORK_NEURALNETSINGLE_H_
#define NETWORK_NEURALNETSINGLE_H_





class NeuralNetSingle : public Network {
public:
	NeuralNetSingle(UINT batchSize=1, double lr_mult=0.1, double decay_mult=5.0) : Network(batchSize) {

		InputLayer *inputLayer = new InputLayer(
				"input",
				io_dim(28, 28, 1, batchSize)
				);

		HiddenLayer *fc1Layer = new FullyConnectedLayer(
				"fc1",
				io_dim(28*28*1, 1, 1, batchSize),
				io_dim(100, 1, 1, batchSize),
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Gaussian, 1),
				ActivationType::ReLU
				);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax",
				io_dim(100, 1, 1, batchSize),
				io_dim(10, 1, 1, batchSize),
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



#endif /* NETWORK_NEURALNETSINGLE_H_ */
