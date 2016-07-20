/*
 * ConvNetMult.h
 *
 *  Created on: 2016. 7. 18.
 *      Author: jhkim
 */

#ifndef CONVNETMULT_H_
#define CONVNETMULT_H_



#include "../activation/ReLU.h"
#include "../layer/ConvLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/PoolingLayer.h"
#include "../layer/SoftmaxLayer.h"
#include "../pooling/MaxPooling.h"
#include "Network.h"





class ConvNetMult : public Network {
public:
	ConvNetMult(NetworkListener *networkListener=0, double lr_mult = 0.05, double decay_mult = 5.0) : Network(networkListener) {
		int filters1 = 20;
		int filters2 = 40;

		InputLayer *inputLayer = new InputLayer(
				"input"
				//io_dim(28, 28, 1, batchSize)
				);

		HiddenLayer *conv1Layer = new ConvLayer(
				"conv1",
				//io_dim(28, 28, 1, batchSize),
				//io_dim(28, 28, filters1, batchSize),
				filter_dim(5, 5, 3, 12, 2),
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.0),
				ActivationType::ReLU
				);

		HiddenLayer *pool1Layer = new PoolingLayer(
				"pool1",
				//io_dim(28, 28, filters1, batchSize),
				//io_dim(14, 14, filters1, batchSize),
				pool_dim(3, 3, 2),
				PoolingType::Max
				);

		HiddenLayer *conv2Layer = new ConvLayer(
				"conv2",
				//io_dim(14, 14, filters1, batchSize),
				//io_dim(14, 14, filters2, batchSize),
				filter_dim(5, 5, 12, 24, 1),
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.0),
				ActivationType::ReLU
				);

		HiddenLayer *pool2Layer = new PoolingLayer(
				"pool2",
				//io_dim(14, 14, filters2, batchSize),
				//io_dim(7, 7, filters2, batchSize),
				pool_dim(3, 3, 2),
				PoolingType::Max
				);

		HiddenLayer *conv3Layer = new ConvLayer(
				"conv3",
				//io_dim(14, 14, filters1, batchSize),
				//io_dim(14, 14, filters2, batchSize),
				filter_dim(5, 5, 24, 48, 2),
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.0),
				ActivationType::ReLU
				);

		HiddenLayer *pool3Layer = new PoolingLayer(
				"pool3",
				//io_dim(14, 14, filters2, batchSize),
				//io_dim(7, 7, filters2, batchSize),
				pool_dim(3, 3, 2),
				PoolingType::Max
				);

		/*
		HiddenLayer *conv4Layer = new ConvLayer(
				"conv4",
				//io_dim(14, 14, filters1, batchSize),
				//io_dim(14, 14, filters2, batchSize),
				filter_dim(5, 5, 64, 128, 1),
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Gaussian, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *pool4Layer = new PoolingLayer(
				"pool4",
				//io_dim(14, 14, filters2, batchSize),
				//io_dim(7, 7, filters2, batchSize),
				pool_dim(3, 3, 1),
				PoolingType::Max
				);
				*/


		//HiddenLayer *fc1Layer = new FullyConnectedLayer("fc1", 7*7*40, 100, 0.5, new ReLU(io_dim(100, 1, 1)));
		HiddenLayer *fc1Layer = new FullyConnectedLayer(
				"fc1",
				//io_dim(7*7*filters2, 1, 1, batchSize),
				//io_dim(100, 1, 1, batchSize),
				100,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.0),
				ActivationType::ReLU);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax",
				//io_dim(100, 1, 1, batchSize),
				//io_dim(10, 1, 1, batchSize),
				10,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Constant, 0.0),
				param_filler(ParamFillerType::Constant, 0.0)
				);

		Network::addLayerRelation(inputLayer, conv1Layer);
		Network::addLayerRelation(conv1Layer, pool1Layer);
		Network::addLayerRelation(pool1Layer, conv2Layer);
		Network::addLayerRelation(conv2Layer, pool2Layer);
		Network::addLayerRelation(pool2Layer, conv3Layer);
		Network::addLayerRelation(conv3Layer, pool3Layer);
		Network::addLayerRelation(pool3Layer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~ConvNetMult() {}
};





#endif /* CONVNETMULT_H_ */
