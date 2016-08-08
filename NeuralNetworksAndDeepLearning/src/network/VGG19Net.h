/**
 * @file VGG19Net.h
 * @date 2016/7/22
 * @author jhkim
 * @brief
 * @details
 */

#ifndef VGG19NET_H_
#define VGG19NET_H_

#include "../activation/Activation.h"
#include "../layer/ConvLayer.h"
#include "../layer/InputLayer.h"
#include "../layer/LayerConfig.h"
#include "../layer/PoolingLayer.h"
#include "../layer/SoftmaxLayer.h"
#include "../pooling/Pooling.h"
#include "Network.h"






/**
 * @brief VGG19Net을 구현한 Network 클래스
 */
class VGG19Net : public Network {
public:
	VGG19Net(NetworkListener *networkListener=0, double w_lr_mult=0.1, double w_decay_mult=1.0,
			double b_lr_mult=2.0, double b_decay_mult=0.0) : Network(networkListener) {
		int filters = 20;

		/*
		InputLayer *inputLayer = new InputLayer(
				"input"
				);

		HiddenLayer *conv1_1Layer = new ConvLayer(
				"conv1_1",
				filter_dim(3, 3, 3, 3, 2),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv1_2Layer = new ConvLayer(
				"conv1_2",
				filter_dim(3, 3, 3, 3, 2),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *pool1Layer = new PoolingLayer(
				"pool5",
				pool_dim(2, 2, 2),
				PoolingType::Max
				);

		HiddenLayer *conv2_1Layer = new ConvLayer(
				"conv2_1",
				filter_dim(3, 3, 3, 3, 4),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax8",
				1,
				0.5,
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Constant, 0.0),
				param_filler(ParamFillerType::Constant, 0.1)
				);

		Network::addLayerRelation(inputLayer, conv1_1Layer);
		Network::addLayerRelation(conv1_1Layer, conv1_2Layer);
		Network::addLayerRelation(conv1_2Layer, pool1Layer);
		Network::addLayerRelation(pool1Layer, conv2_1Layer);
		Network::addLayerRelation(conv2_1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
		*/



		InputLayer *inputLayer = new InputLayer(
				"input"
				);
		HiddenLayer *conv1_1Layer = new ConvLayer(
				"conv1_1",
				filter_dim(3, 3, 3, 64, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv1_2Layer = new ConvLayer(
				"conv1_2",
				filter_dim(3, 3, 64, 64, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *pool1Layer = new PoolingLayer(
				"pool5",
				pool_dim(2, 2, 2),
				PoolingType::Avg
				);

		HiddenLayer *conv2_1Layer = new ConvLayer(
				"conv2_1",
				filter_dim(3, 3, 64, 128, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv2_2Layer = new ConvLayer(
				"conv2_2",
				filter_dim(3, 3, 128, 128, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *pool2Layer = new PoolingLayer(
				"pool2",
				pool_dim(2, 2, 2),
				PoolingType::Avg
				);

		HiddenLayer *conv3_1Layer = new ConvLayer(
				"conv3_1",
				filter_dim(3, 3, 128, 256, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);


		HiddenLayer *conv3_2Layer = new ConvLayer(
				"conv3_2",
				filter_dim(3, 3, 256, 256, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv3_3Layer = new ConvLayer(
				"conv3_3",
				filter_dim(3, 3, 256, 256, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv3_4Layer = new ConvLayer(
				"conv3_3",
				filter_dim(3, 3, 256, 256, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *pool3Layer = new PoolingLayer(
				"pool3",
				pool_dim(2, 2, 2),
				PoolingType::Avg
				);

		HiddenLayer *conv4_1Layer = new ConvLayer(
				"conv4_1",
				filter_dim(3, 3, 256, 512, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv4_2Layer = new ConvLayer(
				"conv4_1",
				filter_dim(3, 3, 512, 512, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv4_3Layer = new ConvLayer(
				"conv4_3",
				filter_dim(3, 3, 512, 512, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv4_4Layer = new ConvLayer(
				"conv4_4",
				filter_dim(3, 3, 512, 512, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *pool4Layer = new PoolingLayer(
				"pool4",
				pool_dim(2, 2, 2),
				PoolingType::Avg
				);

		HiddenLayer *conv5_1Layer = new ConvLayer(
				"conv5_1",
				filter_dim(3, 3, 512, 512, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv5_2Layer = new ConvLayer(
				"conv5_1",
				filter_dim(3, 3, 512, 512, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv5_3Layer = new ConvLayer(
				"conv5_3",
				filter_dim(3, 3, 512, 512, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *conv5_4Layer = new ConvLayer(
				"conv5_4",
				filter_dim(3, 3, 512, 512, 1),
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *pool5Layer = new PoolingLayer(
				"pool5",
				pool_dim(2, 2, 2),
				PoolingType::Avg
				);

		HiddenLayer *fc7Layer = new FullyConnectedLayer(
				"fc7",
				4096,
				0.5,
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		HiddenLayer *fc8Layer = new FullyConnectedLayer(
				"fc8",
				1000,
				0.5,
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				ActivationType::ReLU
				);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax8",
				100,
				0.5,
				update_param(w_lr_mult, w_decay_mult),
				update_param(b_lr_mult, b_decay_mult),
				param_filler(ParamFillerType::Constant, 0.0),
				param_filler(ParamFillerType::Constant, 0.1)
				);

		Network::addLayerRelation(inputLayer, conv1_1Layer);
		Network::addLayerRelation(conv1_1Layer, conv1_2Layer);
		Network::addLayerRelation(conv1_2Layer, pool1Layer);
		Network::addLayerRelation(pool1Layer, conv2_1Layer);
		Network::addLayerRelation(conv2_1Layer, conv2_2Layer);
		Network::addLayerRelation(conv2_2Layer, pool2Layer);
		Network::addLayerRelation(pool2Layer, conv3_1Layer);
		Network::addLayerRelation(conv3_1Layer, conv3_2Layer);
		Network::addLayerRelation(conv3_2Layer, conv3_3Layer);
		Network::addLayerRelation(conv3_3Layer, conv3_4Layer);
		Network::addLayerRelation(conv3_4Layer, pool3Layer);
		Network::addLayerRelation(pool3Layer, conv4_1Layer);
		Network::addLayerRelation(conv4_1Layer, conv4_2Layer);
		Network::addLayerRelation(conv4_2Layer, conv4_3Layer);
		Network::addLayerRelation(conv4_3Layer, conv4_4Layer);
		Network::addLayerRelation(conv4_4Layer, pool4Layer);
		Network::addLayerRelation(pool4Layer, conv5_1Layer);
		Network::addLayerRelation(conv5_1Layer, conv5_2Layer);
		Network::addLayerRelation(conv5_2Layer, conv5_3Layer);
		Network::addLayerRelation(conv5_3Layer, conv5_4Layer);
		Network::addLayerRelation(conv5_4Layer, pool5Layer);
		Network::addLayerRelation(pool5Layer, fc7Layer);
		Network::addLayerRelation(fc7Layer, fc8Layer);
		Network::addLayerRelation(fc8Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);


	}
	virtual ~VGG19Net() {}
};






#endif /* VGG19NET_H_ */
