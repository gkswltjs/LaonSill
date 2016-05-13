/*
 * ConvPoolLayer.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef LAYER_CONVPOOLLAYER_H_
#define LAYER_CONVPOOLLAYER_H_

#include "HiddenLayer.h"
#include <armadillo>

using namespace arma;


class ConvPoolLayer : public HiddenLayer {
public:
	ConvPoolLayer(int n_in, int n_out);
	virtual ~ConvPoolLayer();

	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	void feedforward(const vec &input);

	/**
	 * 네트워크 cost에 대한 weight update양 계산
	 * @param next_w: 다음 레이어의 weight
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	void backpropagation(const mat &next_w, const vec &next_delta, const vec &input);

	/**
	 * 현재 레이어가 최종 레이어인 경우 δL을 계산
	 * @param target: 현재 데이터에 대한 목적값
	 * @param output: 레이어 출력
	 */
	//void cost(const vec &target, const vec &input);

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
};

#endif /* LAYER_CONVPOOLLAYER_H_ */
