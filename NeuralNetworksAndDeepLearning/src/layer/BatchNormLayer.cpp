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
	initialize(builder->_epsilon, builder->_train);
}

template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer(const string name, 
    double epsilon, bool train) : LearnableLayer<Dtype>(name) {
	initialize(epsilon, train);
}

template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer(const string name)
: LearnableLayer<Dtype>(name) {
	//initialize(epsilon, train);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::initialize(double epsilon, bool train) {
	this->type                  = Layer<Dtype>::BatchNorm;
    this->epsilon               = epsilon;
    this->train                 = train;

    this->depth                 = 0;

	this->_paramsInitialized.resize(5);
	this->_paramsInitialized[ParamType::Gamma] = false;
	this->_paramsInitialized[ParamType::Beta] = false;
	this->_paramsInitialized[ParamType::GlobalMean] = false;
	this->_paramsInitialized[ParamType::GlobalVar] = false;
	this->_paramsInitialized[ParamType::GlobalCount] = false;

	this->_params.resize(5);
	this->_params[ParamType::Gamma] = new Data<Dtype>(this->name + "_gamma");
	this->_params[ParamType::Beta] = new Data<Dtype>(this->name + "_beta");
	this->_params[ParamType::GlobalMean] = new Data<Dtype>(this->name + "_global_mean");
	this->_params[ParamType::GlobalVar] = new Data<Dtype>(this->name + "_global_var");
	this->_params[ParamType::GlobalCount] = new Data<Dtype>(this->name + "_global_count");

	this->_paramsHistory.resize(5);
	this->_paramsHistory[ParamType::Gamma] = new Data<Dtype>(this->name + "_gamma_history");
	this->_paramsHistory[ParamType::Beta] = new Data<Dtype>(this->name + "_beta_history");
	this->_paramsHistory[ParamType::GlobalMean] =
        new Data<Dtype>(this->name + "_global_mean_history");
	this->_paramsHistory[ParamType::GlobalVar] =
        new Data<Dtype>(this->name + "_global_var_history");
	this->_paramsHistory[ParamType::GlobalCount] =
        new Data<Dtype>(this->name + "_global_count_history");

	this->_paramsHistory2.resize(5);
	this->_paramsHistory2[ParamType::Gamma] = new Data<Dtype>(this->name + "_gamma_history2");
	this->_paramsHistory2[ParamType::Beta] = new Data<Dtype>(this->name + "_beta_history2");
	this->_paramsHistory2[ParamType::GlobalMean] =
        new Data<Dtype>(this->name + "_global_mean_history2");
	this->_paramsHistory2[ParamType::GlobalVar] =
        new Data<Dtype>(this->name + "_global_var_history2");
	this->_paramsHistory2[ParamType::GlobalCount] =
        new Data<Dtype>(this->name + "_global_count_history2");

    this->meanSet               = new Data<Dtype>(this->name + "_mean");
    this->varSet                = new Data<Dtype>(this->name + "_variance");
    this->normInputSet          = new Data<Dtype>(this->name + "_normalizedInput");

    this->decayedBeta1 = 1.0;
    this->decayedBeta2 = 1.0;
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
#if GPU_MODE
    receiver->_params.clear();
    receiver->_paramsHistory.clear();
    receiver->_paramsHistory2.clear();
    receiver->_paramsInitialized.clear();

    for (int i = 0; i < this->_params.size(); i++) {
        receiver->_params.push_back(this->_params[i]);
    }

    SASSERT0(this->_paramsHistory.size() == this->_paramsHistory2.size());

    for (int i = 0; i < this->_paramsHistory.size(); i++) {
        receiver->_paramsHistory.push_back(this->_paramsHistory[i]);
        receiver->_paramsHistory2.push_back(this->_paramsHistory2[i]);
    }

    for (int i = 0; i < this->_paramsInitialized.size(); i++) {
        receiver->_paramsInitialized.push_back(this->_paramsInitialized[i]);
    }
#endif
}

template class BatchNormLayer<float>;
