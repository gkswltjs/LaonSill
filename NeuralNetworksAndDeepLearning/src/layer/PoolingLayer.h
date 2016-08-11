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
	PoolingLayer() { this->type = LayerType::Pooling; }
	/**
	 * @details PoolingLayer 생성자
	 * @param name 레이어 이름 문자열 포인터
	 * @param pool_d 풀링 연산 관련 파라미터 구조체
	 */
	PoolingLayer(const string name, pool_dim pool_d, PoolingType poolingType);
	virtual ~PoolingLayer();

	void backpropagation(UINT idx, DATATYPE *next_delta_input);
	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);


#ifndef GPU_MODE
public:
	rcube &getDeltaInput() { return this->delta_input; }
	void _feedforward(UINT idx, const rcube &input, const char *end=0);
	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		propResetNParam();
	}

#else
public:
	DATATYPE *getDeltaInput() { return this->d_delta_input; }

#endif

protected:
	void initialize(pool_dim pool_d, PoolingType poolingType);
	virtual void _save(ofstream &ofs);
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	virtual void _feedforward(const DATATYPE *input, const char *end=0);

	pool_dim pool_d;				///< 풀링 연산 관련 파라미터 구조체
	Pooling *pooling_fn;			///< 풀링 객체

#ifndef GPU_MODE
protected:
	ucube pool_map;
	rcube delta;
	rcube delta_input;
#else
protected:
	DATATYPE *d_delta;				///< 다음 레이어에서 전달된 gradient 장치 메모리 포인터 (복수의 다음 레이어가 있는 경우 gradient를 누적하는 메모리)
#endif

};





#endif /* LAYER_POOLINGLAYER_H_ */
