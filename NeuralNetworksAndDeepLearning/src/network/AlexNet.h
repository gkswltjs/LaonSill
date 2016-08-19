/**
 * @file AlexNet.h
 * @date 2016/7/21
 * @author jhkim
 * @brief
 * @details
 */


#ifndef ALEXNET_H_
#define ALEXNET_H_



#include "../activation/Activation.h"
#include "../layer/ConvLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/LRNLayer.h"
#include "../layer/PoolingLayer.h"
#include "../layer/SoftmaxLayer.h"
#include "../pooling/Pooling.h"
#include "Network.h"




#ifndef GPU_MODE

/**
 * @brief AlexNet을 구현한 Network 클래스
 */
class AlexNet : public Network {
public:
	AlexNet(NetworkListener *networkListener=0, double w_lr_mult=0.1, double w_decay_mult=1.0,
			double b_lr_mult=2.0, double b_decay_mult=0.0) : Network(networkListener) {
		int filters = 20;

		InputLayer *inputLayer = new InputLayer(
				"input"
				);

		HiddenLayer *conv1Layer = new ConvLayer(
				"conv1",
				filter_dim(11, 11, 3, 96, 4),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				Activation::ReLU
				);

		LRNLayer *pool_norm1 = new LRNLayer(
				"lrn1",
				lrn_dim(5, 0.0001, 0.75)
				);

		HiddenLayer *pool1Layer = new PoolingLayer(
				"pool1",
				pool_dim(3, 3, 2),
				Pooling::Max
				);

		HiddenLayer *conv2Layer = new ConvLayer(
				"conv2",
				filter_dim(5, 5, 96, 256, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				Activation::ReLU
				);

		LRNLayer *pool_norm2 = new LRNLayer(
				"lrn2",
				lrn_dim(5, 0.0001, 0.75)
				);

		HiddenLayer *pool2Layer = new PoolingLayer(
				"pool2",
				pool_dim(3, 3, 2),
				Pooling::Max
				);

		HiddenLayer *conv3Layer = new ConvLayer(
				"conv3",
				filter_dim(3, 3, 256, 384, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				Activation::ReLU
				);

		HiddenLayer *conv4Layer = new ConvLayer(
				"conv4",
				filter_dim(3, 3, 384, 384, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				Activation::ReLU
				);

		HiddenLayer *conv5Layer = new ConvLayer(
				"conv5",
				filter_dim(3, 3, 384, 256, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				Activation::ReLU
				);

		HiddenLayer *pool5Layer = new PoolingLayer(
				"pool5",
				pool_dim(3, 3, 2),
				Pooling::Max
				);

		/*
		HiddenLayer *fc6Layer = new FullyConnectedLayer(
				"fc6",
				4096,
				0.5,
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				Activation::ReLU
				);
				*/

		HiddenLayer *fc7Layer = new FullyConnectedLayer(
				"fc7",
				512,
				0.5,
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				Activation::ReLU
				);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax8",
				10,
				0.5,
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Constant, 0.0),
				param_filler(ParamFillerType::Constant, 0.1)
				);

		Network::addLayerRelation(inputLayer, conv1Layer);
		Network::addLayerRelation(conv1Layer, pool_norm1);
		Network::addLayerRelation(pool_norm1, pool1Layer);
		Network::addLayerRelation(pool1Layer, conv2Layer);
		Network::addLayerRelation(conv2Layer, pool_norm2);
		Network::addLayerRelation(pool_norm2, pool2Layer);
		Network::addLayerRelation(pool2Layer, conv3Layer);
		Network::addLayerRelation(conv3Layer, conv4Layer);
		Network::addLayerRelation(conv4Layer, conv5Layer);
		Network::addLayerRelation(conv5Layer, pool5Layer);
		Network::addLayerRelation(pool5Layer, fc7Layer);
		//Network::addLayerRelation(fc6Layer, fc7Layer);
		Network::addLayerRelation(fc7Layer, softmaxLayer);


		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~AlexNet() {}
};
#endif



#endif /* ALEXNET_H_ */
