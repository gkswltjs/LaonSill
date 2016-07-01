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
	OutputLayer() {}
	//OutputLayer(const char *name, int n_in, int n_out, double p_dropout) : FullyConnectedLayer(name, n_in, n_out, p_dropout) {}
	OutputLayer(const char *name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType=ActivationType::None, CostType costType=CostType::None)
		: FullyConnectedLayer(name, n_in, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType) {
		initialize(costType);
	}
	//OutputLayer(const char *name, io_dim in_dim, io_dim out_dim, double p_dropout) : FullyConnectedLayer(name, in_dim, out_dim, p_dropout) {}
	OutputLayer(const char *name, io_dim in_dim, io_dim out_dim, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType=ActivationType::None, CostType costType=CostType::None)
		:FullyConnectedLayer(name, in_dim, out_dim, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType) {
		initialize(costType);
	};
	virtual ~OutputLayer() {
		CostFactory::destroy(cost_fn);
	};

protected:
	void initialize(CostType costType) {
		//if(this->activation_fn) this->activation_fn->initialize_weight(in_dim.rows, weight);
		this->cost_fn = CostFactory::create(costType);
	}

#if CPU_MODE
public:
	/**
	 * 현재 레이어가 최종 레이어인 경우 δL을 계산
	 * @param target: 현재 데이터에 대한 목적값
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void cost(const rvec &target)=0;
	virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
		FullyConnectedLayer::load(ifs, layerMap);

		CostType type;
		ifs.read((char *)&type, sizeof(int));

		initialize(type);
	}


protected:
	virtual void save(ofstream &ofs) {
		FullyConnectedLayer::save(ofs);

		int costType = (int)cost_fn->getType();
		ofs.write((char *)&costType, sizeof(int));
	}

	Cost *cost_fn;


#else

public:
	/**
	 * 현재 레이어가 최종 레이어인 경우 δL을 계산
	 * @param target: 현재 데이터에 대한 목적값
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void cost(const UINT *target)=0;
	virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
		FullyConnectedLayer::load(ifs, layerMap);
		//CostType type;
		//ifs.read((char *)&type, sizeof(int));
		//initialize(type);
	}


protected:
	virtual void save(ofstream &ofs) {
		FullyConnectedLayer::save(ofs);
		//int costType = (int)cost_fn->getType();
		//ofs.write((char *)&costType, sizeof(int));
	}
	Cost *cost_fn;

#endif


};






#endif /* LAYER_OUTPUTLAYER_H_ */








































