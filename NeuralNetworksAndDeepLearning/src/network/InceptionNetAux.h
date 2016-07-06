/*
 * InceptionNetAux.h
 *
 *  Created on: 2016. 6. 2.
 *      Author: jhkim
 */

#ifndef NETWORK_INCEPTIONNETAUX_H_
#define NETWORK_INCEPTIONNETAUX_H_

#include "../activation/Activation.h"
#include "../layer/ConvLayer.h"
#include "../layer/InceptionLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/PoolingLayer.h"
#include "../layer/SoftmaxLayer.h"
#include "../pooling/Pooling.h"
#include "../Util.h"
#include "Network.h"


class InceptionNetAux : public Network {
public:
	InceptionNetAux(UINT batchSize=10, double lr_mult=0.5, double decay_mult=0.1) : Network(batchSize) {

			InputLayer *inputLayer = new InputLayer(
					"input",
					io_dim(28, 28, 1, batchSize)
					);

			HiddenLayer *conv1Layer = new ConvLayer(
					"conv1",
					io_dim(28, 28, 1, batchSize),
					io_dim(28, 28, 10, batchSize),
					filter_dim(5, 5, 1, 10, 1),
					update_param(lr_mult, decay_mult),
					update_param(lr_mult, decay_mult),
					param_filler(ParamFillerType::Xavier),
					param_filler(ParamFillerType::Constant, 0.1),
					ActivationType::ReLU
					);

			HiddenLayer *pool1Layer = new PoolingLayer(
					"pool1",
					io_dim(28, 28, 10, batchSize),
					io_dim(14, 14, 10, batchSize),
					pool_dim(3, 3, 2),
					PoolingType::Max
					);

			HiddenLayer *lrn1Layer = new LRNLayer(
					"lrn1",
					io_dim(14, 14, 10, batchSize),
					lrn_dim(5)
					);

			HiddenLayer *incept1Layer = new InceptionLayer(
					"incept1",
					io_dim(14, 14, 10, batchSize),
					io_dim(14, 14, 60, batchSize),
					15, 10, 15, 10, 15, 15
					);


			/*
			HiddenLayer *incept2Layer = new InceptionLayer(
					"incept2",
					io_dim(28, 28, 12, batchSize),
					io_dim(28, 28, 24, batchSize),
					6, 4, 6, 4, 6, 6
					);
					*/

			/*
			HiddenLayer *fc1Layer = new FullyConnectedLayer(
							"fc1",
							io_dim(14*14*120, 1, 1, batchSize),
							io_dim(500, 1, 1, batchSize),
							0.5,
							update_param(lr_mult, decay_mult),
							update_param(lr_mult, decay_mult),
							param_filler(ParamFillerType::Xavier),
							param_filler(ParamFillerType::Constant, 0.1),
							ActivationType::ReLU);
							*/

			OutputLayer *softmaxLayer = new SoftmaxLayer(
					"softmax",
					io_dim(14*14*60, 1, 1, batchSize),
					io_dim(10, 1, 1, batchSize),
					0.5,
					update_param(lr_mult, decay_mult),
					update_param(lr_mult, decay_mult),
					param_filler(ParamFillerType::Xavier),
					param_filler(ParamFillerType::Constant, 0.1)
					);

			Network::addLayerRelation(inputLayer, conv1Layer);
			Network::addLayerRelation(conv1Layer, pool1Layer);
			Network::addLayerRelation(pool1Layer, lrn1Layer);
			Network::addLayerRelation(lrn1Layer, incept1Layer);
			//Network::addLayerRelation(pool1Layer, incept1Layer);
			Network::addLayerRelation(incept1Layer, softmaxLayer);

			this->inputLayer = inputLayer;
			addOutputLayer(softmaxLayer);
	}
	virtual ~InceptionNetAux() {}
};


#endif /* NETWORK_INCEPTIONNETAUX_H_ */
