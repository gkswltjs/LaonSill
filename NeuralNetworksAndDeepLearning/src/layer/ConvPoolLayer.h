/*
 * ConvPoolLayer.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef LAYER_CONVPOOLLAYER_H_
#define LAYER_CONVPOOLLAYER_H_

#include "HiddenLayer.h"
#include "LayerConfig.h"
#include "../activation/Activation.h"
#include "../pooling/Pooling.h"
#include <armadillo>

using namespace arma;


class ConvPoolLayer : public HiddenLayer {
public:
	ConvPoolLayer(io_dim in_dim, filter_dim filter_d, pool_dim pool_d, Activation *activation_fn, Pooling *pooling_fn);
	virtual ~ConvPoolLayer();

	filter_dim &get_filter_dim() { return this->filter_d; }
	cube *getWeight() { return this->filters; }
	cube &getDelta() { return this->delta; }


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
	void convolution(const mat &image, const mat &filter, mat &result);
	void d_convolution(const mat &conv, const mat &filter, mat &result);

	io_dim in_dim;
	filter_dim filter_d;
	pool_dim pool_d;

	cube *filters;		// weights
	vec biases;

	cube *nabla_w;
	vec nabla_b;

	cube z;
	cube activated;
	ucube pool_map;
	cube delta;

	Activation *activation_fn;
	Pooling *pooling_fn;
};

#endif /* LAYER_CONVPOOLLAYER_H_ */

































