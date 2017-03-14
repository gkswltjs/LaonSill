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
    this->gammaSet              = new Data<Dtype>(this->name + "_gamma");
    this->betaSet               = new Data<Dtype>(this->name + "_beta");
    this->gammaCacheSet         = new Data<Dtype>(this->name + "_gammaCache");
    this->betaCacheSet          = new Data<Dtype>(this->name + "_betaCache");
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
