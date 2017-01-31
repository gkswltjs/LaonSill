/**
 * @file AAALayer.cpp
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "AAALayer.h"
#include "Util.h"
#include "Exception.h"
#include "SysLog.h"

using namespace std;


template <typename Dtype>
AAALayer<Dtype>::AAALayer() {
	this->type = Layer<Dtype>::AAA;
}

template <typename Dtype>
AAALayer<Dtype>::AAALayer(Builder* builder)
	: HiddenLayer<Dtype>(builder) {
	initialize(builder->_var1, builder->_var2);
}

template <typename Dtype>
AAALayer<Dtype>::AAALayer(
    const string name, int var1, double var2) : HiddenLayer<Dtype>(name) {
	initialize(var1, var2);
}

template <typename Dtype>
void AAALayer<Dtype>::initialize(int var1, double var2) {
	this->type = Layer<Dtype>::AAA;
	this->var1 = var1;
    this->var2 = var2;
}

template <typename Dtype>
double AAALayer<Dtype>::sumSquareParamsData() {
	return 0.0;
}

template <typename Dtype>
double AAALayer<Dtype>::sumSquareParamsGrad() {
	return 0.0;
}

template <typename Dtype>
void AAALayer<Dtype>::scaleParamsGrad(float scale) {
}

template <typename Dtype>
uint32_t AAALayer<Dtype>::boundParams() {
	return 1;
}

template <typename Dtype>
uint32_t AAALayer<Dtype>::numParams() {
	return 1;
}

template <typename Dtype>
void AAALayer<Dtype>::saveParams(ofstream& ofs) {
}

template <typename Dtype>
void AAALayer<Dtype>::loadParams(ifstream& ifs) {
}

template <typename Dtype>
void AAALayer<Dtype>::loadParams(map<string, Data<Dtype>*>& dataMap) {
}

#ifndef GPU_MODE
template <typename Dtype>
AAALayer<Dtype>::~AAALayer() {
}

template <typename Dtype>
void AAALayer<Dtype>::initialize(double var1, double var2) {
	this->type = Layer<Dtype>::AAA;
	this->var1 = var1;
    this->var2 = var2;
}

template <typename Dtype>
void AAALayer<Dtype>::feedforward() {
}

template <typename Dtype>
void AAALayer<Dtype>::backpropagation(uint32_t idx, HiddenLayer *next_layer) {
}

#endif

template class AAALayer<float>;
