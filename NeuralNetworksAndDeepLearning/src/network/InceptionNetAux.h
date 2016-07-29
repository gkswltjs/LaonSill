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
	InceptionNetAux(NetworkListener *networkListener=0, double lr_mult=0.0025, double decay_mult=0.0) : Network(networkListener) {
			update_param weight_update_param(lr_mult, decay_mult);
			update_param bias_update_param(lr_mult, decay_mult);

			InputLayer *inputLayer = new InputLayer(
					"input"
					//io_dim(28, 28, 1, batchSize)
					);

			HiddenLayer *conv1Layer = new ConvLayer(
					"conv1",
					//io_dim(28, 28, 1, batchSize),
					//io_dim(28, 28, 10, batchSize),
					filter_dim(5, 5, 3, 10, 2),
					//update_param(lr_mult, decay_mult),
					//update_param(lr_mult, decay_mult),
					weight_update_param,
					bias_update_param,
					param_filler(ParamFillerType::Xavier),
					param_filler(ParamFillerType::Constant, 0.0),
					ActivationType::ReLU
					);

			HiddenLayer *pool1Layer = new PoolingLayer(
					"pool1",
					//io_dim(28, 28, 10, batchSize),
					//io_dim(14, 14, 10, batchSize),
					pool_dim(3, 3, 2),
					PoolingType::Max
					);

			HiddenLayer *lrn1Layer = new LRNLayer(
					"lrn1",
					//io_dim(14, 14, 10, batchSize),
					lrn_dim(5)
					);

			HiddenLayer *incept1Layer = new InceptionLayer(
					"incept1",
					//io_dim(14, 14, 10, batchSize),
					//io_dim(14, 14, 60, batchSize),
					10,
					15, 10, 15, 10, 15, 15,
					weight_update_param,
					bias_update_param
					);


			HiddenLayer *incept2Layer = new InceptionLayer(
					"incept2",
					//io_dim(14, 14, 60, batchSize),
					//io_dim(14, 14, 60, batchSize),
					60,
					15, 15, 15, 15, 15, 15,
					//update_param(lr_mult, decay_mult),
					//update_param(lr_mult, decay_mult),
					weight_update_param,
					bias_update_param
					);

			HiddenLayer *conv2Layer = new ConvLayer(
					"conv2",
					//io_dim(28, 28, 1, batchSize),
					//io_dim(28, 28, 10, batchSize),
					filter_dim(5, 5, 60, 80, 2),
					//update_param(lr_mult, decay_mult),
					//update_param(lr_mult, decay_mult),
					weight_update_param,
					bias_update_param,
					param_filler(ParamFillerType::Xavier),
					param_filler(ParamFillerType::Constant, 0.0),
					ActivationType::ReLU
					);

			HiddenLayer *pool2Layer = new PoolingLayer(
					"pool2",
					//io_dim(28, 28, 10, batchSize),
					//io_dim(14, 14, 10, batchSize),
					pool_dim(3, 3, 2),
					PoolingType::Max
					);

			/*
			HiddenLayer *incept3Layer = new InceptionLayer(
					"incept3",
					//io_dim(14, 14, 60, batchSize),
					//io_dim(14, 14, 60, batchSize),
					60,
					20, 15, 20, 15, 20, 20
					);

			HiddenLayer *incept4Layer = new InceptionLayer(
					"incept4",
					//io_dim(14, 14, 60, batchSize),
					//io_dim(14, 14, 60, batchSize),
					80,
					25, 20, 25, 20, 25, 25
					);

			HiddenLayer *conv2Layer = new ConvLayer(
					"conv2",
					//io_dim(28, 28, 1, batchSize),
					//io_dim(28, 28, 10, batchSize),
					filter_dim(5, 5, 100, 100, 2),
					update_param(lr_mult, decay_mult),
					update_param(lr_mult, decay_mult),
					param_filler(ParamFillerType::Xavier),
					param_filler(ParamFillerType::Constant, 0.0),
					ActivationType::ReLU
					);

			HiddenLayer *pool2Layer = new PoolingLayer(
					"pool2",
					//io_dim(28, 28, 10, batchSize),
					//io_dim(14, 14, 10, batchSize),
					pool_dim(3, 3, 2),
					PoolingType::Max
					);

			HiddenLayer *fc1Layer = new FullyConnectedLayer(
					"fc1",
					1000,
					0.5,
					update_param(lr_mult, decay_mult),
					update_param(lr_mult, decay_mult),
					param_filler(ParamFillerType::Xavier),
					param_filler(ParamFillerType::Constant, 0.0),
					ActivationType::ReLU);
					*/

			OutputLayer *softmaxLayer = new SoftmaxLayer(
					"softmax",
					//io_dim(14*14*60, 1, 1, batchSize),
					//io_dim(10, 1, 1, batchSize),
					10,
					0.5,
					//update_param(lr_mult, decay_mult),
					//update_param(lr_mult, decay_mult),
					weight_update_param,
					bias_update_param,
					param_filler(ParamFillerType::Constant, 0.0),
					param_filler(ParamFillerType::Constant, 0.0)
					);

			Network::addLayerRelation(inputLayer, conv1Layer);
			Network::addLayerRelation(conv1Layer, pool1Layer);
			Network::addLayerRelation(pool1Layer, lrn1Layer);
			Network::addLayerRelation(lrn1Layer, incept1Layer);
			Network::addLayerRelation(incept1Layer, incept2Layer);
			Network::addLayerRelation(incept2Layer, conv2Layer);
			Network::addLayerRelation(conv2Layer, pool2Layer);
			/*
			Network::addLayerRelation(incept2Layer, incept3Layer);
			Network::addLayerRelation(incept3Layer, incept4Layer);
			Network::addLayerRelation(incept4Layer, conv2Layer);
			Network::addLayerRelation(conv2Layer, pool2Layer);
			Network::addLayerRelation(pool2Layer, fc1Layer);
			*/
			Network::addLayerRelation(pool2Layer, softmaxLayer);

			this->inputLayer = inputLayer;
			addOutputLayer(softmaxLayer);
	}
	virtual ~InceptionNetAux() {}
};


#endif /* NETWORK_INCEPTIONNETAUX_H_ */
