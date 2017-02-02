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
BatchNormLayer<Dtype>::BatchNormLayer() {
	this->type = Layer<Dtype>::BatchNorm;
}

template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer(Builder* builder)
	: HiddenLayer<Dtype>(builder) {
	initialize(builder->_kernelMapCount, builder->_epsilon);
}

template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer(const string name, int kernelMapCount,
    double epsilon) : HiddenLayer<Dtype>(name) {
	initialize(kernelMapCount, epsilon);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::initialize(int kernelMapCount, double epsilon) {
	this->type                  = Layer<Dtype>::BatchNorm;
	this->kernelMapCount        = kernelMapCount;
    this->epsilon               = epsilon;

    this->depth                 = 0;
    this->batchSetCount         = 0;
    this->gammaSets             = NULL;
    this->betaSets              = NULL;
    this->meanSumSets           = NULL;
    this->varianceSumSets       = NULL;
    this->localMeanSets         = NULL;
    this->localVarianceSets     = NULL;
    this->normInputValues       = NULL;
    this->normInputGradValues   = NULL;
    this->varianceGradValues    = NULL;
    this->meanGradValues        = NULL;
    this->gammaGradValues       = NULL;
    this->betaGradValues        = NULL;
}

template <typename Dtype>
double BatchNormLayer<Dtype>::sumSquareParamsData() {
    // TODO:
	double result = 0.0;
	return result;
}

template <typename Dtype>
double BatchNormLayer<Dtype>::sumSquareParamsGrad() {
    // TODO:
	double result = 0.0;
	return result;
}

template <typename Dtype>
void BatchNormLayer<Dtype>::scaleParamsGrad(float scale) {
    // TODO:
}

template <typename Dtype>
uint32_t BatchNormLayer<Dtype>::boundParams() {
    // TODO:
	uint32_t updateCount = 1;
	return updateCount;
}

template <typename Dtype>
uint32_t BatchNormLayer<Dtype>::numParams() {
    // TODO:
	return 1;
}

template <typename Dtype>
void BatchNormLayer<Dtype>::saveParams(ofstream& ofs) {
    // TODO:
}

template <typename Dtype>
void BatchNormLayer<Dtype>::loadParams(ifstream& ifs) {
    // TODO:
}

template <typename Dtype>
void BatchNormLayer<Dtype>::loadParams(map<string, Data<Dtype>*>& dataMap) {
    // TODO:
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
void BatchNormLayer<Dtype>::backpropagation(uint32_t idx, HiddenLayer *next_layer) {
    // TODO:
}

#endif

template class BatchNormLayer<float>;
