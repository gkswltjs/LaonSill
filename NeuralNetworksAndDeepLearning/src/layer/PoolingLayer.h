/*
 * PoolingLayer.h
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#ifndef LAYER_POOLINGLAYER_H_
#define LAYER_POOLINGLAYER_H_

#include "HiddenLayer.h"
#include "../pooling/Pooling.h"


class PoolingLayer : public HiddenLayer {
public:
	PoolingLayer(io_dim in_dim, pool_dim pool_d, Pooling *pooling_fn);
	virtual ~PoolingLayer();

	cube &getDeltaInput() { return this->delta_input; }


	void feedforward(const cube &input);

	/**
	 * 네트워크 cost에 대한 weight update양 계산
	 * @param next_w: 다음 레이어의 weight
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	void backpropagation(HiddenLayer *next_layer);

	/**
	 * 한 번의 batch 종료 후 재사용을 위해 w, b 누적 업데이트를 reset
	 */
	void reset_nabla();

	/**
	 * 한 번의 batch 종료 후 w, b 누적 업데이트를 레이어 w, b에 적용
	 * @param eta:
	 * @param lambda:
	 * @param n:
	 * @param miniBatchSize:
	 */
	void update(double eta, double lambda, int n, int miniBatchSize);

private:
	ucube pool_map;
	cube delta;
	cube delta_input;

	pool_dim pool_d;
	Pooling *pooling_fn;
};

#endif /* LAYER_POOLINGLAYER_H_ */
