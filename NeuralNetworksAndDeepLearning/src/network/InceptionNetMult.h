/*
 * InceptionNetSingle.h
 *
 *  Created on: 2016. 6. 2.
 *      Author: jhkim
 */

#ifndef NETWORK_INCEPTIONNETMULT_H_
#define NETWORK_INCEPTIONNETMULT_H_

#include "../layer/InceptionLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/SoftmaxLayer.h"
#include "Network.h"




class InceptionNetMult : public Network {
public:
	InceptionNetMult(UINT batchSize=1, NetworkListener *networkListener=0, double lr_mult=0.01, double decay_mult=5.0)
		: Network(batchSize, networkListener) {
		//double lr_mult = 0.01;
		//double decay_mult = 5.0;

		InputLayer *inputLayer = new InputLayer(
				"input",
				io_dim(28, 28, 1, batchSize)
				);

		HiddenLayer *incept1Layer = new InceptionLayer(
				"incept1",
				io_dim(28, 28, 1, batchSize),
				io_dim(28, 28, 12, batchSize),
				3, 2, 3, 2, 3, 3
				);

		HiddenLayer *incept2Layer = new InceptionLayer(
				"incept2",
				io_dim(28, 28, 12, batchSize),
				io_dim(28, 28, 24, batchSize),
				6, 4, 6, 4, 6, 6
				);

		/*
		HiddenLayer *incept2Layer = new InceptiornLayer(
				"incept2",
				io_dim(28, 28, 24, batchSize),
				io_dim(28, 28, 36, batchSize),
				9, 6, 9, 6, 9, 9
				);
				*/

		FullyConnectedLayer *fcLayer = new FullyConnectedLayer(
				"fc1",
				io_dim(28*28*24, 1, 1, batchSize),
				io_dim(1000, 1, 1, batchSize),
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Gaussian, 1),
				ActivationType::ReLU);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax",
				io_dim(1000, 1, 1, batchSize),
				io_dim(10, 1, 1, batchSize),
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1)
				);

		Network::addLayerRelation(inputLayer, incept1Layer);
		Network::addLayerRelation(incept1Layer, incept2Layer);
		Network::addLayerRelation(incept2Layer, fcLayer);
		Network::addLayerRelation(fcLayer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~InceptionNetMult() {}
};


#endif /* NETWORK_INCEPTIONNETMULT_H_ */
