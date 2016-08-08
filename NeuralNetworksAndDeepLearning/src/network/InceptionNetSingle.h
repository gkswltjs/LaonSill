/**
 * @file InceptionNetSingle.h
 * @date 2016/6/2
 * @author jhkim
 * @brief
 * @details
 */

#ifndef NETWORK_INCEPTIONNETSINGLE_H_
#define NETWORK_INCEPTIONNETSINGLE_H_

#include <cstdio>

#include "../activation/Activation.h"
#include "../layer/ConvLayer.h"
#include "../layer/DepthConcatLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/PoolingLayer.h"
#include "../layer/SoftmaxLayer.h"
#include "../pooling/Pooling.h"
#include "Network.h"



/**
 * @brief 하나의 인셉션 레이어를 가진 Network를 구현한 클래스
 */
class InceptionNetSingle : public Network {
public:
	InceptionNetSingle(NetworkListener *networkListener=0, double lr_mult=0.01, double decay_mult=5.0) : Network(networkListener) {
		//double lr_mult = 0.01;
		//double decay_mult = 5.0;

		update_param weight_update_param(lr_mult, decay_mult);
		update_param bias_update_param(lr_mult, decay_mult);

		InputLayer *inputLayer = new InputLayer(
				"input"
				);


		HiddenLayer *incept1Layer = new InceptionLayer(
				"incept1",
				3,
				3, 2, 3, 2, 3, 3,
				weight_update_param,
				bias_update_param
				);


		/*
		int ic = 3;
		int oc_cv1x1 = 3;
		int oc_cv3x3reduce = 2;
		int oc_cv3x3 = 3;
		int oc_cv5x5reduce = 2;
		int oc_cv5x5 = 3;
		int oc_cp = 3;

		const char *inceptionName = "inception1";
		char subLayerName[256];
		sprintf(subLayerName, "%s/%s", inceptionName, "conv1x1");
		ConvLayer *conv1x1Layer = new ConvLayer(
				subLayerName,
				filter_dim(1, 1, ic, oc_cv1x1, 1),
				//update_param(weight_lr_mult, weight_decay_mult),
				//update_param(bias_lr_mult, bias_decay_mult),
				weight_update_param,
				bias_update_param,
				param_filler(ParamFillerType::Xavier, 0.03),
				param_filler(ParamFillerType::Constant, 0.2),
				ActivationType::ReLU
				);

		sprintf(subLayerName, "%s/%s", inceptionName, "conv3x3reduce");
		ConvLayer *conv3x3reduceLayer = new ConvLayer(
				subLayerName,
				filter_dim(1, 1, ic, oc_cv3x3reduce, 1),
				//update_param(weight_lr_mult, weight_decay_mult),
				//update_param(bias_lr_mult, bias_decay_mult),
				weight_update_param,
				bias_update_param,
				param_filler(ParamFillerType::Xavier, 0.09),
				param_filler(ParamFillerType::Constant, 0.2),
				ActivationType::ReLU);

		sprintf(subLayerName, "%s/%s", inceptionName, "conv3x3");
		ConvLayer *conv3x3Layer = new ConvLayer(
				subLayerName,
				filter_dim(3, 3, oc_cv3x3reduce, oc_cv3x3, 1),
				//update_param(weight_lr_mult, weight_decay_mult),
				//update_param(bias_lr_mult, bias_decay_mult),
				weight_update_param,
				bias_update_param,
				param_filler(ParamFillerType::Xavier, 0.03),
				param_filler(ParamFillerType::Constant, 0.2),
				ActivationType::ReLU
				);

		sprintf(subLayerName, "%s/%s", inceptionName, "conv5x5reduce");
		ConvLayer *conv5x5recudeLayer = new ConvLayer(
				subLayerName,
				filter_dim(1, 1, ic, oc_cv5x5reduce, 1),
				//update_param(weight_lr_mult, weight_decay_mult),
				//update_param(bias_lr_mult, bias_decay_mult),
				weight_update_param,
				bias_update_param,
				param_filler(ParamFillerType::Xavier, 0.2),
				param_filler(ParamFillerType::Constant, 0.2),
				ActivationType::ReLU
				);

		sprintf(subLayerName, "%s/%s", inceptionName, "conv5x5");
		ConvLayer *conv5x5Layer = new ConvLayer(
				subLayerName,
				filter_dim(5, 5, oc_cv5x5reduce, oc_cv5x5, 1),
				//update_param(weight_lr_mult, weight_decay_mult),
				//update_param(bias_lr_mult, bias_decay_mult),
				weight_update_param,
				bias_update_param,
				param_filler(ParamFillerType::Xavier, 0.03),
				param_filler(ParamFillerType::Constant, 0.2),
				ActivationType::ReLU
				);

		sprintf(subLayerName, "%s/%s", inceptionName, "pool3x3");
		PoolingLayer *pool3x3Layer = new PoolingLayer(
				subLayerName,
				pool_dim(3, 3, 1),
				PoolingType::Max
				);

		sprintf(subLayerName, "%s/%s", inceptionName, "convProjection");
		ConvLayer *convProjectionLayer = new ConvLayer(
				subLayerName,
				filter_dim(1, 1, ic, oc_cp, 1),
				//update_param(weight_lr_mult, weight_decay_mult),
				//update_param(bias_lr_mult, bias_decay_mult),
				weight_update_param,
				bias_update_param,
				param_filler(ParamFillerType::Xavier, 0.1),
				param_filler(ParamFillerType::Constant, 0.2),
				ActivationType::ReLU);

		sprintf(subLayerName, "%s/%s", inceptionName, "depthConcat");
		DepthConcatLayer *depthConcatLayer = new DepthConcatLayer(
				subLayerName
				);

		*/





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
				100,
				0.5,
				weight_update_param,
				bias_update_param,
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1)
				);

		Network::addLayerRelation(inputLayer, incept1Layer);

		/*
		Network::addLayerRelation(inputLayer, conv1x1Layer);
		Network::addLayerRelation(inputLayer, conv3x3reduceLayer);
		Network::addLayerRelation(inputLayer, conv5x5recudeLayer);
		Network::addLayerRelation(inputLayer, pool3x3Layer);

		Network::addLayerRelation(conv3x3reduceLayer, conv3x3Layer);
		Network::addLayerRelation(conv5x5recudeLayer, conv5x5Layer);
		Network::addLayerRelation(pool3x3Layer, convProjectionLayer);
		Network::addLayerRelation(conv1x1Layer, depthConcatLayer);
		Network::addLayerRelation(conv3x3Layer, depthConcatLayer);
		Network::addLayerRelation(conv5x5Layer, depthConcatLayer);
		Network::addLayerRelation(convProjectionLayer, depthConcatLayer);
		Network::addLayerRelation(depthConcatLayer, softmaxLayer);
		*/
		Network::addLayerRelation(incept1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~InceptionNetSingle() {}
};


#endif /* NETWORK_INCEPTIONNETSINGLE_H_ */
