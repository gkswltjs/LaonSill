/**
 * @file BatchNormLayer.cpp
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "BatchNormLayer.h"
#include "Util.h"
#include "Exception.h"
#include "SysLog.h"
#include "ColdLog.h"

using namespace std;


template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer(Builder* builder)
	: LearnableLayer<Dtype>(builder) {
	initialize(builder->_epsilon);
}

template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer(const string name, 
    double epsilon) : LearnableLayer<Dtype>(name) {
	initialize(epsilon);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::initialize(double epsilon) {
	this->type                  = Layer<Dtype>::BatchNorm;
    this->epsilon               = epsilon;

    this->depth                 = 0;
    this->batchSetCount         = 0;

	this->_paramsInitialized.resize(2);
	this->_paramsInitialized[ParamType::Gamma] = false;
	this->_paramsInitialized[ParamType::Beta] = false;

	this->_params.resize(2);
	this->_params[ParamType::Gamma] = new Data<Dtype>(this->name + "_gamma");
	this->_params[ParamType::Beta] = new Data<Dtype>(this->name + "_beta");

	this->_paramsHistory.resize(2);
	this->_paramsHistory[ParamType::Gamma] = new Data<Dtype>(this->name + "_gamma_history");
	this->_paramsHistory[ParamType::Beta] = new Data<Dtype>(this->name + "_beta_history");

	this->_paramsHistory2.resize(2);
	this->_paramsHistory2[ParamType::Gamma] = new Data<Dtype>(this->name + "_gamma_history2");
	this->_paramsHistory2[ParamType::Beta] = new Data<Dtype>(this->name + "_beta_history2");

    this->meanSet               = new Data<Dtype>(this->name + "_mean");
    this->varSet                = new Data<Dtype>(this->name + "_variance");
    this->normInputSet          = new Data<Dtype>(this->name + "_normalizedInput");

    shared_ptr<SyncMem<Dtype>> tempMeanSumSet(new SyncMem<Dtype>());
    shared_ptr<SyncMem<Dtype>> tempVarSumSet(new SyncMem<Dtype>());

    this->meanSumSet            = tempMeanSumSet;
    this->varSumSet             = tempVarSumSet;
}


#ifndef GPU_MODE
template <typename Dtype>
BatchNormLayer<Dtype>::~BatchNormLayer() {
    // TODO:
}

template <typename Dtype>
void BatchNormLayer<Dtype>::feedforward() {
    // TODO:
}

template <typename Dtype>
void BatchNormLayer<Dtype>::backpropagation(uint32_t idx, Layer *next_layer) {
    // TODO:
}

#endif

template<typename Dtype>
void BatchNormLayer<Dtype>::donateParam(BatchNormLayer<Dtype>* receiver) {
    // XXX: hmm... does this layer really need param donation??
    return;
}

template class BatchNormLayer<float>;
