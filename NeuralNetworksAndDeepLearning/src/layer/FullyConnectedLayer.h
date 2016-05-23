/*
 * FullyConnectedLayer.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef LAYER_FULLYCONNECTEDLAYER_H_
#define LAYER_FULLYCONNECTEDLAYER_H_

#include "HiddenLayer.h"
#include "LayerConfig.h"
#include "../activation/Activation.h"
#include "../cost/Cost.h"

class FullyConnectedLayer : public HiddenLayer {
public:
	FullyConnectedLayer(int n_in, int n_out, double p_dropout, Activation *activation_fn=0);
	FullyConnectedLayer(io_dim in_dim, io_dim out_dim, double p_dropout, Activation *activation_fn=0);
	virtual ~FullyConnectedLayer();

	mat &getWeight() { return this->weight; }
	cube &getDeltaInput() { return this->delta_input; }

	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
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
	void initialize(double p_dropout, Activation *activation_fn);

protected:
	double p_dropout;

	mat weight;
	vec bias;

	vec nabla_b;
	mat nabla_w;

	cube z;
	cube delta;
	cube delta_input;
	Activation *activation_fn;
};

#endif /* LAYER_FULLYCONNECTEDLAYER_H_ */
