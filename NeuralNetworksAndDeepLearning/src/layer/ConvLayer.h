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
	ConvLayer() { this->type = LayerType::Conv; }
	ConvLayer(const char *name, filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_f, param_filler bias_filler, ActivationType activationType);
	virtual ~ConvLayer();

	filter_dim &get_filter_dim() { return this->filter_d; }
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
	void update(UINT idx, UINT n, UINT miniBatchSize);


	void save(UINT idx, ofstream &ofs);
	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

#if CPU_MODE
public:

	rcube *getWeight() { return this->filters; }
	rcube &getDeltaInput() { return this->delta_input; }


	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	void feedforward(UINT idx, const rcube &input);

#else
	//static void init();
	//static void destroy();

	DATATYPE *getWeight() { return this->filters; }
	DATATYPE *getDeltaInput() { return this->d_delta_input; }
	void feedforward(UINT idx, const DATATYPE *input);
#endif


protected:
	void initialize(filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType);
	void save(ofstream &ofs);

	virtual void _shape();
	virtual void _reshape();

	filter_dim filter_d;
	Activation *activation_fn;

	update_param weight_update_param;
	update_param bias_update_param;
	param_filler weight_filler;
	param_filler bias_filler;

#if CPU_MODE
protected:
	void convolution(const rmat &x, const rmat &w, rmat &result, int stride);
	void dw_convolution(const rmat &d, const rmat &x, rmat &result);
	void dx_convolution(const rmat &d, const rmat &w, rmat &result);

	rcube *filters;		// weights
	rvec biases;

	rcube *nabla_w;
	rvec nabla_b;

	rcube z;
	rcube delta;
	rcube delta_input;
#else
protected:
	DATATYPE *filters;
	DATATYPE *biases;

	DATATYPE *d_filters;
	DATATYPE *d_biases;

	DATATYPE *d_z;
	DATATYPE *d_delta;
	DATATYPE *d_delta_input;
	DATATYPE *d_delta_weight;
	DATATYPE *d_delta_bias;

	cudnnTensorDescriptor_t biasTensorDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t convFwdAlgo;
	cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
	cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;

	size_t workspaceSize;
	void *d_workspace;
#endif


};



#endif /* LAYER_CONVLAYER_H_ */
