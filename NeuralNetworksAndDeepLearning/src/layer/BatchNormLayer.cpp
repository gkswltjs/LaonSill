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

using namespace std;


template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer() {
	this->type = Layer<Dtype>::BatchNorm;
}

template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer(Builder* builder)
	: HiddenLayer<Dtype>(builder) {
	initialize(builder->_nOut, builder->_epsilon);
}

template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer(const string name, int n_out,
    double epsilon) : HiddenLayer<Dtype>(name) {
	initialize(n_out, epsilon);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::initialize(int n_out, double epsilon) {

	// out_dim의 batches는 _shape()에서 in_dim값에 따라 결정된다.
	//this->out_dim = io_dim(n_out, 1, 1, 1);
	this->type = Layer<Dtype>::BatchNorm;
	this->n_out = n_out;
    this->epsilon = epsilon;
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
void BatchNormLayer<Dtype>::initialize(double p_dropout, double epsilon) {
	this->type = Layer<Dtype>::BatchNorm;
	this->p_dropout = p_dropout;
    this->epsilon = epsilon;
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
