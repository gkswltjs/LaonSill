/*
 * LayerFactory.h
 *
 *  Created on: 2016. 6. 9.
 *      Author: jhkim
 */

#ifndef LAYER_LAYERFACTORY_H_
#define LAYER_LAYERFACTORY_H_

#include "Layer.h"
#include "InputLayer.h"
#include "SoftmaxLayer.h"
#include "../exception/Exception.h"


class LayerFactory {
public:
	LayerFactory() {}
	virtual ~LayerFactory() {}

	static Layer *create(LayerType layerType) {
		switch(layerType) {
		//case LayerType::Input: return new InputLayer();
		case LayerType::FullyConnected: return new FullyConnectedLayer();
		case LayerType::Softmax: return new SoftmaxLayer();
		default: throw new Exception();
			//Conv, Pooling, DepthConcat, Inception, LRN, Sigmoid
		}
	}

	static void destroy(Layer *layer) {
		if(layer) delete layer;
	}

};

#endif /* LAYER_LAYERFACTORY_H_ */
