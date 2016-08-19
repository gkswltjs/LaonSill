/**
 * @file ConvNetSingle.h
 * @date 2016/6/2
 * @author jhkim
 * @brief
 * @details
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


#ifndef GPU_MODE

/**
 * @brief 하나의 컨볼루션 레이어를 가진 Network를 구현한 클래스
 */
class ConvNetSingle : public Network {
public:
	ConvNetSingle(NetworkListener *networkListener=0, double lr_mult=0.1, double decay_mult=5.0) : Network(networkListener) {
		int filters = 20;

		InputLayer *inputLayer = new InputLayer(
				"input"
				//io_dim(28, 28, 1, batchSize)
				);

		HiddenLayer *conv1Layer = new ConvLayer(
				"conv1",
				//io_dim(28, 28, 1, batchSize),
				//io_dim(28, 28, filters, batchSize),
				filter_dim(5, 5, 3, filters, 1),
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				Activation::ReLU
				);

		HiddenLayer *pool1Layer = new PoolingLayer(
				"pool1",
				//io_dim(28, 28, filters, batchSize),
				//io_dim(14, 14, filters, batchSize),
				pool_dim(3, 3, 2),
				Pooling::Max
				);

		//HiddenLayer *fc1Layer = new FullyConnectedLayer("fc1", 14*14*20, 100, 0.5, new ReLU(io_dim(100, 1, 1)));
		HiddenLayer *fc1Layer = new FullyConnectedLayer(
				"fc1",
				//io_dim(14*14*filters, 1, 1, batchSize),
				//io_dim(100, 1, 1, batchSize),
				100,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1),
				Activation::ReLU
				);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax",
				//io_dim(100, 1, 1, batchSize),
				//io_dim(10, 1, 1, batchSize),
				10,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Constant, 0.1)
				);

		Network::addLayerRelation(inputLayer, conv1Layer);
		Network::addLayerRelation(conv1Layer, pool1Layer);
		Network::addLayerRelation(pool1Layer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~ConvNetSingle() {}
};

#endif

#endif /* NETWORK_CONVNETSINGLE_H_ */
