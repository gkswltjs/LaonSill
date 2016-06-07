/*
 * ConvLayer.h
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#ifndef LAYER_CONVLAYER_H_
#define LAYER_CONVLAYER_H_

#include <string>

#include "../activation/ActivationFactory.h"
#include "../Util.h"
#include "HiddenLayer.h"
#include "LayerConfig.h"


class ConvLayer : public HiddenLayer {
public:
	ConvLayer(string name, io_dim in_dim, filter_dim filter_d, ActivationType activationType);
	virtual ~ConvLayer();

	filter_dim &get_filter_dim() { return this->filter_d; }
	rcube *getWeight() { return this->filters; }
	rcube &getDeltaInput() { return this->delta_input; }


	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	void feedforward(UINT idx, const rcube &input);

	/**
	 * 네트워크 cost에 대한 weight update양 계산
	 * @param next_w: 다음 레이어의 weight
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	void backpropagation(UINT idx, HiddenLayer *next_layer);

	/**
	 * 현재 레이어가 최종 레이어인 경우 δL을 계산
	 * @param target: 현재 데이터에 대한 목적값
	 * @param output: 레이어 출력
	 */
	//void cost(const vec &target, const vec &input);

	/**
	 * 한 번의 batch 종료 후 재사용을 위해 w, b 누적 업데이트를 reset
	 */
	void reset_nabla(UINT idx);

	/**
	 * 한 번의 batch 종료 후 w, b 누적 업데이트를 레이어 w, b에 적용
	 * @param eta:
	 * @param lambda:
	 * @param n:
	 * @param miniBatchSize:
	 */
	void update(UINT idx, double eta, double lambda, int n, int miniBatchSize);



protected:
	void convolution(const rmat &x, const rmat &w, rmat &result, int stride);
	void dw_convolution(const rmat &d, const rmat &x, rmat &result);
	void dx_convolution(const rmat &d, const rmat &w, rmat &result);


	filter_dim filter_d;

	rcube *filters;		// weights
	rvec biases;

	rcube *nabla_w;
	rvec nabla_b;

	rcube z;
	rcube delta;
	rcube delta_input;

	Activation *activation_fn;
};

#endif /* LAYER_CONVLAYER_H_ */
