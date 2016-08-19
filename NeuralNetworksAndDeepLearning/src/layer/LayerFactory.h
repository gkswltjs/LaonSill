/**
 * @file	LayerFactory.h
 * @date	2016/6/9
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_LAYERFACTORY_H_
#define LAYER_LAYERFACTORY_H_

#include <stddef.h>

#include "../exception/Exception.h"
#include "ConvLayer.h"
#include "DepthConcatLayer.h"
//#include "InceptionLayer.h"
#include "InputLayer.h"
#include "Layer.h"
#include "LRNLayer.h"
#include "PoolingLayer.h"
#include "SoftmaxLayer.h"


class Layer;




/**
 * @brief 주어진 레이어 타입에 따라 레이어 객체를 생성하여 반환
 * @details 주어진 레이어 타입에 따라 레이어 객체를 생성하여 반환하고
 *          사용이 완료된 레이어 객체를 소멸시키는 역할을 함.
 * @todo (객체를 생성한 곳에서 삭제한다는 원칙에 따라 만들었으나 수정이 필요)
 */
class LayerFactory {
public:
	LayerFactory() {}
	virtual ~LayerFactory() {}

	/**
	 * @details 주어진 레이어 타입에 따라 레이어 객체를 생성하여 반환.
	 * @param layerType 생성하고자 하는 레이어 객체의 타입.
	 * @return 생성한 레이어 객체.
	 */
	static Layer *create(Layer::Type layerType) {
		switch(layerType) {
		case Layer::Input: return new InputLayer();
		case Layer::FullyConnected: return new FullyConnectedLayer();
		case Layer::Conv: return new ConvLayer();
		case Layer::Pool: return new PoolingLayer();
		case Layer::DepthConcat: return new DepthConcatLayer();
		//case Layer::Inception: return new InceptionLayer();
		case Layer::LRN: return new LRNLayer();
		//case Layer::Sigmoid: return new SigmoidLayer();
		case Layer::Softmax: return new SoftmaxLayer();
		default: throw new Exception();
		}
	}

	/**
	 * @details LayerFactory에서 생성한 레이어 객체를 소멸.
	 * @param layer 레이어 객체에 대한 포인터 참조자.
	 */
	static void destroy(Layer *&layer) {
		if(layer) {
			delete layer;
			layer = NULL;
		}
	}

};

#endif /* LAYER_LAYERFACTORY_H_ */
