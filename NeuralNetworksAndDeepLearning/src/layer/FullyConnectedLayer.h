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
#include "../activation/ActivationFactory.h"
#include "../cost/Cost.h"






class FullyConnectedLayer : public HiddenLayer {
public:
	FullyConnectedLayer() { this->type = LayerType::FullyConnected; }

	FullyConnectedLayer(const char *name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType=ActivationType::None);
	virtual ~FullyConnectedLayer();

	/**
	 * 네트워크 cost에 대한 weight update양 계산
	 * @param next_w: 다음 레이어의 weight
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void backpropagation(UINT idx, HiddenLayer *next_layer);

	/**
	 * 한 번의 batch 종료 후 재사용을 위해 w, b 누적 업데이트를 reset
	 */
	virtual void reset_nabla(UINT idx);

	/**
	 * 한 번의 batch 종료 후 w, b 누적 업데이트를 레이어 w, b에 적용
	 * @param eta:
	 * @param lambda:
	 * @param n:
	 * @param miniBatchSize:
	 */
	virtual void update(UINT idx, UINT n, UINT miniBatchSize);

	virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

#if CPU_MODE
public:
	FullyConnectedLayer(const char *name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType=ActivationType::None);

	rmat &getWeight() { return this->weight; }
	rcube &getDeltaInput() { return this->delta_input; }

	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void feedforward(UINT idx, const rcube &input, const char *end=0);
#else
public:
	DATATYPE *getWeight() { return this->d_weight; }
	DATATYPE *getDeltaInput() { return this->d_delta_input; }

	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void feedforward(UINT idx, const DATATYPE *input, const char *end=0);

#endif





private:
	void initialize(int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType);

protected:
	virtual void _save(ofstream &ofs);
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();

	double p_dropout;

	update_param weight_update_param;
	update_param bias_update_param;

	param_filler weight_filler;
	param_filler bias_filler;

	Activation *activation_fn;

#if CPU_MODE
protected:
	rmat weight;
	rvec bias;

	rvec nabla_b;
	rmat nabla_w;

	rcube z;
	rcube delta;
	rcube delta_input;
#else
protected:
	DATATYPE *weight;
	DATATYPE *bias;

	DATATYPE *d_weight;
	DATATYPE *d_bias;

	DATATYPE *d_z;
	DATATYPE *d_delta;
	DATATYPE *d_delta_weight;
	DATATYPE *d_delta_bias;

	DATATYPE *d_onevec;
#endif


};






#endif /* LAYER_FULLYCONNECTEDLAYER_H_ */
