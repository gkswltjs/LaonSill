/*
 * HiddenLayer.h
 *
 *  Created on: 2016. 5. 11.
 *      Author: jhkim
 */

#ifndef LAYER_HIDDENLAYER_H_
#define LAYER_HIDDENLAYER_H_

#include "Layer.h"
#include <armadillo>

using namespace arma;


class HiddenLayer : public Layer {
public:
	HiddenLayer(int n_in, int n_out)
		: Layer(n_in, n_in) {}
	virtual ~HiddenLayer() {}


	mat &getWeight() { return this->weight; }
	vec &getDelta() { return this->delta; }


	/**
	 * 네트워크 cost에 대한 weight update양 계산
	 * @param next_w: 다음 레이어의 weight
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void backpropagation(const mat &next_w, const vec &next_delta, const vec &input)=0;

	/**
	 * 한 번의 batch 종료 후 재사용을 위해 w, b 누적 업데이트를 reset
	 */
	virtual void reset_nabla()=0;

	/**
	 * 한 번의 batch 종료 후 w, b 누적 업데이트를 레이어 w, b에 적용
	 * @param eta:
	 * @param lambda:
	 * @param n:
	 * @param miniBatchSize:
	 */
	virtual void update(double eta, double lambda, int n, int miniBatchSize)=0;

protected:
	mat weight;
	vec delta;
};

#endif /* LAYER_HIDDENLAYER_H_ */
