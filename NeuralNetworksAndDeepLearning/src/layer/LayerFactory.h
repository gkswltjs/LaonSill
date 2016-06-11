/*
 * LayerFactory.h
 *
 *  Created on: 2016. 6. 9.
 *      Author: jhkim
 */

#ifndef LAYER_LAYERFACTORY_H_
#define LAYER_LAYERFACTORY_H_

#include "../exception/Exception.h"
#include "ConvLayer.h"
#include "DepthConcatLayer.h"
#include "InceptionLayer.h"
#include "InputLayer.h"
#include "LRNLayer.h"
#include "PoolingLayer.h"
#include "SigmoidLayer.h"
#include "SoftmaxLayer.h"

class Layer;





class LayerFactory {
public:
	LayerFactory() {}
	virtual ~LayerFactory() {}

	static Layer *create(LayerType layerType) {
		switch(layerType) {
		case LayerType::Input: return new InputLayer();
		case LayerType::FullyConnected: return new FullyConnectedLayer();
		case LayerType::Conv: return new ConvLayer();
		case LayerType::Pooling: return new PoolingLayer();
		case LayerType::DepthConcat: return new DepthConcatLayer();
		case LayerType::Inception: return new InceptionLayer();
		case LayerType::LRN: return new LRNLayer();
		case LayerType::Sigmoid: return new SigmoidLayer();
		case LayerType::Softmax: return new SoftmaxLayer();
		default: throw new Exception();
		}
	}

	static void destroy(Layer *&layer) {
		if(layer) {
			delete layer;
			layer = NULL;
		}
	}

};

#endif /* LAYER_LAYERFACTORY_H_ */
