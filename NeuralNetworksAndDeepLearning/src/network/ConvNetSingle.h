/*
 * ConvNetSingle.h
 *
 *  Created on: 2016. 6. 2.
 *      Author: jhkim
 */

#ifndef NETWORK_CONVNETSINGLE_H_
#define NETWORK_CONVNETSINGLE_H_

#include "../activation/ReLU.h"
#include "../layer/ConvLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/PoolingLayer.h"
#include "../layer/SoftmaxLayer.h"
#include "../pooling/MaxPooling.h"
#include "Network.h"


class ConvNetSingle : public Network {
public:
	ConvNetSingle(NetworkListener *networkListener) : Network(0, 0, 0) {
		InputLayer *inputLayer = new InputLayer("input", io_dim(28, 28, 1));
		HiddenLayer *conv1Layer = new ConvLayer("conv1", io_dim(28, 28, 1), filter_dim(5, 5, 1, 20, 1), new ReLU(io_dim(28, 28, 20)));
		HiddenLayer *pool1Layer = new PoolingLayer("pool1", io_dim(28, 28, 20), pool_dim(3, 3, 2), new MaxPooling());
		HiddenLayer *fc1Layer = new FullyConnectedLayer("fc1", 14*14*20, 100, 0.5, new ReLU(io_dim(100, 1, 1)));
		OutputLayer *softmaxLayer = new SoftmaxLayer("softmax", 100, 10, 0.5);

		Network::addLayerRelation(inputLayer, conv1Layer);
		Network::addLayerRelation(conv1Layer, pool1Layer);
		Network::addLayerRelation(pool1Layer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~ConvNetSingle() {}
};

#endif /* NETWORK_CONVNETSINGLE_H_ */
