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
	NeuralNetSingle() : Network(0, 0, 0) {
		InputLayer *inputLayer = new InputLayer("input", io_dim(28, 28, 1));
		HiddenLayer *fc1Layer = new FullyConnectedLayer("fc1", 28*28*1, 100, 0.5, ActivationType::ReLU);
		OutputLayer *softmaxLayer = new SoftmaxLayer("softmax", 100, 10, 0.5);

		Network::addLayerRelation(inputLayer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~NeuralNetSingle() {}
};

#endif /* NETWORK_NEURALNETSINGLE_H_ */
