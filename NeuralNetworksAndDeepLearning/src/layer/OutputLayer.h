/*
 * OutputLayer.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef LAYER_OUTPUTLAYER_H_
#define LAYER_OUTPUTLAYER_H_

#include "FullyConnectedLayer.h"
#include <armadillo>

using namespace arma;


class OutputLayer : public FullyConnectedLayer {
public:
	OutputLayer(int n_in, int n_out, double p_dropout) : FullyConnectedLayer(n_in, n_out, p_dropout) {}
	OutputLayer(int n_in, int n_out, double p_dropout, Activation *activation_fn, Cost *cost_fn)
		: FullyConnectedLayer(n_in, n_out, p_dropout) {
		initialize(activation_fn, cost_fn);
	}
	OutputLayer(io_dim in_dim, io_dim out_dim, double p_dropout) : FullyConnectedLayer(in_dim, out_dim, p_dropout) {}
	OutputLayer(io_dim in_dim, io_dim out_dim, double p_dropout, Activation *activation_fn, Cost *cost_fn)
		:FullyConnectedLayer(in_dim, out_dim, p_dropout) {
		initialize(activation_fn, cost_fn);
	};
	virtual ~OutputLayer() {};

	/**
	 * 현재 레이어가 최종 레이어인 경우 δL을 계산
	 * @param target: 현재 데이터에 대한 목적값
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void cost(const vec &target)=0;

private:
	void initialize(Activation *activation_fn, Cost *cost_fn) {
		this->activation_fn = activation_fn;
		this->activation_fn->initialize_weight(in_dim.rows, weight);
		this->cost_fn = cost_fn;
	}


protected:
	Cost *cost_fn;

};

#endif /* LAYER_OUTPUTLAYER_H_ */
