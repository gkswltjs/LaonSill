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

#include "common.h"
#include "Exception.h"
#include "ConvLayer.h"
#include "DepthConcatLayer.h"
#include "FullyConnectedLayer.h"
#include "InputLayer.h"
#include "Layer.h"
#include "LRNLayer.h"
#include "PoolingLayer.h"
#include "SoftmaxLayer.h"

/**
 * @brief 주어진 레이어 타입에 따라 레이어 객체를 생성하여 반환
 * @details 주어진 레이어 타입에 따라 레이어 객체를 생성하여 반환하고
 *          사용이 완료된 레이어 객체를 소멸시키는 역할을 함.
 * @todo (객체를 생성한 곳에서 삭제한다는 원칙에 따라 만들었으나 수정이 필요)
 */
template <typename Dtype>
class LayerFactory {
public:
	LayerFactory() {}
	virtual ~LayerFactory() {}

	/**
	 * @details 주어진 레이어 타입에 따라 레이어 객체를 생성하여 반환.
	 * @param layerType 생성하고자 하는 레이어 객체의 타입.
	 * @return 생성한 레이어 객체.
	 */
	static Layer<Dtype>* create(typename Layer<Dtype>::LayerType layerType) {
		switch(layerType) {
		case Layer<Dtype>::Input: return new InputLayer<Dtype>();
		case Layer<Dtype>::FullyConnected: return new FullyConnectedLayer<Dtype>();
		case Layer<Dtype>::Conv: return new ConvLayer<Dtype>();
		case Layer<Dtype>::Pool: return new PoolingLayer<Dtype>();
		case Layer<Dtype>::DepthConcat: return new DepthConcatLayer<Dtype>();
		//case Layer<Dtype>::Inception: return new InceptionLayer<Dtype>();
		case Layer<Dtype>::LRN: return new LRNLayer<Dtype>();
		//case Layer<Dtype>::Sigmoid: return new SigmoidLayer<Dtype>();
		case Layer<Dtype>::Softmax: return new SoftmaxLayer<Dtype>();
		default: throw new Exception();
		}
		//return 0;
	}


	/**
	 * @details LayerFactory에서 생성한 레이어 객체를 소멸.
	 * @param layer 레이어 객체에 대한 포인터 참조자.
	 */

	static void destroy(Layer<Dtype>*& layer) {
		if(layer) {
			delete layer;
			layer = NULL;
		}
	}


};

template <typename Dtype>
class LayerBuilderFactory {
public:
	LayerBuilderFactory() {}
	virtual ~LayerBuilderFactory() {}

	static typename Layer<Dtype>::Builder* create(typename Layer<Dtype>::LayerType layerType) {
		switch(layerType) {
		case Layer<Dtype>::Input: return new typename InputLayer<Dtype>::Builder();
		case Layer<Dtype>::FullyConnected: return new typename FullyConnectedLayer<Dtype>::Builder();
		case Layer<Dtype>::Conv: return new typename ConvLayer<Dtype>::Builder();
		case Layer<Dtype>::Pool: return new typename PoolingLayer<Dtype>::Builder();
		case Layer<Dtype>::DepthConcat: return new typename DepthConcatLayer<Dtype>::Builder();
		//case Layer<Dtype>::Inception: return new InceptionLayer<Dtype>();
		case Layer<Dtype>::LRN: return new typename LRNLayer<Dtype>::Builder();
		//case Layer<Dtype>::Sigmoid: return new SigmoidLayer<Dtype>();
		case Layer<Dtype>::Softmax: return new typename SoftmaxLayer<Dtype>::Builder();
		default: throw new Exception();
		}
		//return 0;
	}
};





template class LayerFactory<float>;


#endif /* LAYER_LAYERFACTORY_H_ */
