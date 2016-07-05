/*
 * InceptionNetSingle.h
 *
 *  Created on: 2016. 6. 2.
 *      Author: jhkim
 */

#ifndef NETWORK_INCEPTIONNETSINGLE_H_
#define NETWORK_INCEPTIONNETSINGLE_H_

#include "../layer/InceptionLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/SoftmaxLayer.h"
#include "Network.h"




class InceptionNetSingle : public Network {
public:
	InceptionNetSingle(UINT batchSize=1, double lr_mult=0.01, double decay_mult=5.0) : Network(batchSize) {
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

		/*
		HiddenLayer *incept2Layer = new InceptionLayer(
				"incept2",
				io_dim(28, 28, 12, batchSize),
				io_dim(28, 28, 24, batchSize),
				6, 4, 6, 4, 6, 6
				);
				*/

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax",
				io_dim(28*28*12, 1, 1, batchSize),
				io_dim(10, 1, 1, batchSize),
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1)
				);

		Network::addLayerRelation(inputLayer, incept1Layer);
		//Network::addLayerRelation(incept1Layer, incept2Layer);
		Network::addLayerRelation(incept1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~InceptionNetSingle() {}
};


#endif /* NETWORK_INCEPTIONNETSINGLE_H_ */
