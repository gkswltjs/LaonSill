/*
 * FullyConnectedLayer.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef LAYER_FULLYCONNECTEDLAYER_H_
#define LAYER_FULLYCONNECTEDLAYER_H_

#include "HiddenLayer.h"
#include "../activation/Activation.h"
#include "../cost/Cost.h"

class FullyConnectedLayer : public HiddenLayer {
public:
	FullyConnectedLayer(int n_in, int n_out, Activation *activation_fn = 0);
	virtual ~FullyConnectedLayer();

	void setActivation(Activation *activation_fn) { this->activation_fn = activation_fn; }


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

protected:
	int n_in;
	int n_out;

	vec bias;

	vec nabla_b;
	mat nabla_w;

	vec z;
	Activation *activation_fn;
};

#endif /* LAYER_FULLYCONNECTEDLAYER_H_ */
