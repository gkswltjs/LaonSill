/*
 * OutputLayer.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef LAYER_OUTPUTLAYER_H_
#define LAYER_OUTPUTLAYER_H_

#include "FullyConnectedLayer.h"
#include "../cost/CostFactory.h"
#include <armadillo>

using namespace arma;


class OutputLayer : public FullyConnectedLayer {
public:
	OutputLayer(string name, int n_in, int n_out, double p_dropout) : FullyConnectedLayer(name, n_in, n_out, p_dropout) {}
	OutputLayer(string name, int n_in, int n_out, double p_dropout, ActivationType activationType, CostType costType)
		: FullyConnectedLayer(name, n_in, n_out, p_dropout) {
		initialize(activationType, costType);
	}
	OutputLayer(string name, io_dim in_dim, io_dim out_dim, double p_dropout) : FullyConnectedLayer(name, in_dim, out_dim, p_dropout) {}
	OutputLayer(string name, io_dim in_dim, io_dim out_dim, double p_dropout, ActivationType activationType, CostType costType)
		:FullyConnectedLayer(name, in_dim, out_dim, p_dropout) {
		initialize(activationType, costType);
	};
	virtual ~OutputLayer() {
		CostFactory::destroy(cost_fn);
		ActivationFactory::destory(activation_fn);
	};

	/**
	 * 현재 레이어가 최종 레이어인 경우 δL을 계산
	 * @param target: 현재 데이터에 대한 목적값
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void cost(const rvec &target)=0;

private:
	void initialize(ActivationType activationType, CostType costType) {
		this->activation_fn = ActivationFactory::create(activationType);
		if(this->activation_fn) this->activation_fn->initialize_weight(in_dim.rows, weight);
		this->cost_fn = CostFactory::create(costType);
	}

protected:
	Cost *cost_fn;

};

#endif /* LAYER_OUTPUTLAYER_H_ */
