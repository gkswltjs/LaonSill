/*
 * GoogLeNetMnist.h
 *
 *  Created on: 2016. 6. 1.
 *      Author: jhkim
 */

#ifndef NETWORK_GOOGLENETMNIST_H_
#define NETWORK_GOOGLENETMNIST_H_

#include "../activation/ReLU.h"
#include "../layer/ConvLayer.h"
#include "../layer/InceptionLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/LRNLayer.h"
#include "../layer/PoolingLayer.h"
#include "../layer/SoftmaxLayer.h"
#include "../pooling/AvgPooling.h"
#include "../pooling/MaxPooling.h"
#include "Network.h"


class GoogLeNetMnist : public Network {
public:
	GoogLeNetMnist() : Network() {
		double weight_lr_mult = 1.0;
		double weight_decay_mult = 1.0;
		double bias_lr_mult = 2.0;
		double bias_decay_mult = 0.0;

		InputLayer *inputLayer = new InputLayer(
				"input",
				io_dim(28, 28, 1)
				);

		ConvLayer *conv1_7x7_s2 = new ConvLayer(
				"conv1_7x7_s2",
				io_dim(28, 28, 1),
				filter_dim(5, 5, 1, 8, 1),
				update_param(weight_lr_mult, weight_decay_mult),
				update_param(bias_lr_mult, bias_decay_mult),
				param_filler(ParamFillerType::Xavier, 0.1),
				param_filler(ParamFillerType::Constant, 0.2),
				ActivationType::ReLU
				);

		PoolingLayer *pool1_3x3_s2 = new PoolingLayer(
				"pool1_3x3_s2",
				io_dim(28, 28, 8),
				pool_dim(3, 3, 1),
				PoolingType::Max
				);

		LRNLayer *pool1_norm1 = new LRNLayer(
				"lrn1",
				io_dim(28, 28, 8),
				lrn_dim(5, 0.0001, 0.75)
				);

		ConvLayer *conv2_3x3_reduce = new ConvLayer(
				"conv2_3x3_reduce",
				io_dim(28, 28, 8),
				filter_dim(1, 1, 8, 12, 1),
				update_param(weight_lr_mult, weight_decay_mult),
				update_param(bias_lr_mult, bias_decay_mult),
				param_filler(ParamFillerType::Xavier, 0.1),
				param_filler(ParamFillerType::Constant, 0.2),
				ActivationType::ReLU
				);

		ConvLayer *conv2_3x3 = new ConvLayer(
				"conv2_3x3",
				io_dim(28, 28, 12),
				filter_dim(3, 3, 12, 16, 1),
				update_param(weight_lr_mult, weight_decay_mult),
				update_param(bias_lr_mult, bias_decay_mult),
				param_filler(ParamFillerType::Xavier, 0.03),
				param_filler(ParamFillerType::Constant, 0.2),
				ActivationType::ReLU
				);

		LRNLayer *conv2_norm2 = new LRNLayer(
				"lrn1",
				io_dim(28, 28, 16),
				lrn_dim(5, 0.0001, 0.75)
				);

		PoolingLayer *pool2_3x3_s2 = new PoolingLayer(
				"pool2_3x3_s2",
				io_dim(28, 28, 16),
				pool_dim(3, 3, 2),
				PoolingType::Max
				);

		InceptionLayer *inception_3a = new InceptionLayer(
				"inception_3a",
				io_dim(14, 14, 16),
				io_dim(14, 14, 16),
				4, 2, 4, 2, 4, 4
				);

		InceptionLayer *inception_3b = new InceptionLayer(
				"inception_3b",
				io_dim(14, 14, 16),
				io_dim(14, 14, 20),
				5, 3, 5, 3, 5, 5
				);

		PoolingLayer *pool3_3x3_s2 = new PoolingLayer(
				"pool3_3x3_s2",
				io_dim(14, 14, 20),
				pool_dim(3, 3, 1),
				PoolingType::Max
				);

		InceptionLayer *inception_4a = new InceptionLayer(
				"inception_4a",
				io_dim(14, 14, 20),
				io_dim(14, 14, 24),
				6, 3, 6, 3, 6, 6
				);

		InceptionLayer *inception_4b = new InceptionLayer(
				"inception_4b",
				io_dim(14, 14, 24),
				io_dim(14, 14, 28),
				7, 4, 7, 4, 7, 7
				);

		InceptionLayer *inception_4c = new InceptionLayer(
				"inception_4c",
				io_dim(14, 14, 28),
				io_dim(14, 14, 32),
				8, 4, 8, 4, 8, 8
				);

		InceptionLayer *inception_4d = new InceptionLayer(
				"inception_4d",
				io_dim(14, 14, 32),
				io_dim(14, 14, 36),
				9, 5, 9, 5, 9, 9
				);

		InceptionLayer *inception_4e = new InceptionLayer(
				"inception_4e",
				io_dim(14, 14, 36),
				io_dim(14, 14, 40),
				10, 5, 10, 5, 10, 10
				);

		PoolingLayer *pool4_3x3_s2 = new PoolingLayer(
				"pool4_3x3_s2",
				io_dim(14, 14, 40),
				pool_dim(3, 3, 2),
				PoolingType::Max
				);

		InceptionLayer *inception_5a = new InceptionLayer(
				"inception_5a",
				io_dim(7, 7, 40),
				io_dim(7, 7, 44),
				11, 6, 11, 6, 11, 11
				);

		InceptionLayer *inception_5b = new InceptionLayer(
				"inception_5b",
				io_dim(7, 7, 44),
				io_dim(7, 7, 48),
				12, 6, 12, 6, 12, 12
				);

		PoolingLayer *pool5_7x7_s1 = new PoolingLayer(
				"pool5_7x7_s1",
				io_dim(7, 7, 48),
				pool_dim(7, 7, 4),
				PoolingType::Avg
				);

		//FullyConnectedLayer *fc1 = new FullyConnectedLayer("fc1", 48, 48, 0.4, ActivationType::ReLU);

		SoftmaxLayer *outputLayer = new SoftmaxLayer(
				"output",
				48,
				10,
				0.0,
				update_param(weight_lr_mult, weight_decay_mult),
				update_param(bias_lr_mult, bias_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.0)
				);

		Network::addLayerRelation(inputLayer, conv1_7x7_s2);
		Network::addLayerRelation(conv1_7x7_s2, pool1_3x3_s2);
		Network::addLayerRelation(pool1_3x3_s2, pool1_norm1);
		Network::addLayerRelation(pool1_norm1, conv2_3x3_reduce);
		Network::addLayerRelation(conv2_3x3_reduce, conv2_3x3);
		Network::addLayerRelation(conv2_3x3, conv2_norm2);
		Network::addLayerRelation(conv2_norm2, pool2_3x3_s2);
		Network::addLayerRelation(pool2_3x3_s2, inception_3a);
		Network::addLayerRelation(inception_3a, inception_3b);
		Network::addLayerRelation(inception_3b, pool3_3x3_s2);
		Network::addLayerRelation(pool3_3x3_s2, inception_4a);
		Network::addLayerRelation(inception_4a, inception_4b);
		Network::addLayerRelation(inception_4b, inception_4c);
		Network::addLayerRelation(inception_4c, inception_4d);
		Network::addLayerRelation(inception_4d, inception_4e);
		Network::addLayerRelation(inception_4e, pool4_3x3_s2);
		Network::addLayerRelation(pool4_3x3_s2, inception_5a);
		Network::addLayerRelation(inception_5a, inception_5b);
		Network::addLayerRelation(inception_5b, pool5_7x7_s1);
		Network::addLayerRelation(pool5_7x7_s1, outputLayer);
		//Network::addLayerRelation(fc1, outputLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(outputLayer);
	}
	virtual ~GoogLeNetMnist() {}
};

#endif /* NETWORK_GOOGLENETMNIST_H_ */
