/**
 * @file PoolingLayer.h
 * @date 2016/5/23
 * @author jhkim
 * @brief
 * @details
 */


#ifndef LAYER_POOLINGLAYER_H_
#define LAYER_POOLINGLAYER_H_

#include "HiddenLayer.h"
#include "../pooling/Pooling.h"
#include "../pooling/PoolingFactory.h"
#include "../exception/Exception.h"





/**
 * @brief 풀링 레이어
 * @details Max 풀링, Average 풀링 제공
 *          padding에 관한 옵션을 제공하지 않고 있고
 *          GoogLeNet에 따라 Max 풀링의 경우 padding을 기본으로, Average 풀링의 경우 Non padding을 기본으로 하고 있다.
 */
class PoolingLayer : public HiddenLayer {
public:
	class Builder : public HiddenLayer::Builder {
	public:
		pool_dim _poolDim;
		Pooling::Type _poolingType;

		Builder() {
			_poolDim.cols = 0;
			_poolDim.rows = 0;
			_poolDim.stride = 0;
			_poolingType = Pooling::Max;
		}
		Builder* poolDim(uint32_t cols, uint32_t rows, uint32_t stride) {
			this->_poolDim.cols = cols;
			this->_poolDim.rows = rows;
			this->_poolDim.stride = stride;
			return this;
		}
		Builder* poolingType(Pooling::Type poolingType) {
			this->_poolingType = poolingType;
			return this;
		}
		virtual Builder* name(const string name) {
			HiddenLayer::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer::Builder::id(id);
			return this;
		}
		virtual Builder* nextLayerIndices(const vector<uint32_t>& nextLayerIndices) {
			HiddenLayer::Builder::nextLayerIndices(nextLayerIndices);
			return this;
		}
		virtual Builder* prevLayerIndices(const vector<uint32_t>& prevLayerIndices) {
			HiddenLayer::Builder::prevLayerIndices(prevLayerIndices);
			return this;
		}
		Layer* build() {
			return new PoolingLayer(this);
		}
	};

	/**
	 * @details PoolingLayer 기본 생성자
	 */
	PoolingLayer();
	PoolingLayer(Builder* builder);
	/**
	 * @details PoolingLayer 생성자
	 * @param name 레이어 이름 문자열 포인터
	 * @param pool_d 풀링 연산 관련 파라미터 구조체
	 */
	PoolingLayer(const string name, pool_dim pool_d, Pooling::Type poolingType);
	/**
	 * @details PoolingLayer 소멸자
	 */
	virtual ~PoolingLayer();


protected:
	void initialize(pool_dim pool_d, Pooling::Type poolingType);
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	virtual void _save(ofstream &ofs);
	virtual void _load(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	virtual void _backpropagation();
	virtual void _feedforward();

protected:
	pool_dim pool_d;				///< 풀링 연산 관련 파라미터 구조체
	Pooling *pooling_fn;			///< 풀링 객체

#ifndef GPU_MODE
	ucube pool_map;
	rcube delta;
	rcube delta_input;
#else
	//DATATYPE *d_delta;				///< 다음 레이어에서 전달된 gradient 장치 메모리 포인터 (복수의 다음 레이어가 있는 경우 gradient를 누적하는 메모리)
#endif



};





#endif /* LAYER_POOLINGLAYER_H_ */
