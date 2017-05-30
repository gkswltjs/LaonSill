/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"
#include "Util.h"
#include "SysLog.h"
#include "PropMgmt.h"

using namespace std;


template<typename Dtype>
FullyConnectedLayer<Dtype>::FullyConnectedLayer() : LearnableLayer<Dtype>() {
	this->type = Layer<Dtype>::FullyConnected;

	this->scale = 1. / (1. - SLPROP(FullyConnected, pDropOut));

	const string& name = SLPROP_BASE(name);
	this->_params.resize(2);
	this->_params[ParamType::Weight] = new Data<Dtype>(name + "_weight");
	this->_params[ParamType::Bias] = new Data<Dtype>(name + "_bias");

	this->_paramsInitialized.resize(2);
	this->_paramsInitialized[ParamType::Weight] = false;
	this->_paramsInitialized[ParamType::Bias] = false;

	this->_paramsHistory.resize(2);
	this->_paramsHistory[ParamType::Weight] = new Data<Dtype>(name + "_weight_history");
	this->_paramsHistory[ParamType::Bias] = new Data<Dtype>(name + "_bias_history");

	this->_paramsHistory2.resize(2);
	this->_paramsHistory2[ParamType::Weight] = 
		new Data<Dtype>(name + "_weight_history2");
	this->_paramsHistory2[ParamType::Bias] = new Data<Dtype>(name + "_bias_history2");
}


template<typename Dtype>
void FullyConnectedLayer<Dtype>::donateParam(FullyConnectedLayer<Dtype>* receiver) {

}

template class FullyConnectedLayer<float>;
