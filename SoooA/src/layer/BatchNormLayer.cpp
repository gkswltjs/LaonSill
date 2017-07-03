/**
 * @file BatchNormLayer.cpp
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "BatchNormLayer.h"
#include "Util.h"
#include "SysLog.h"
#include "ColdLog.h"
#include "PropMgmt.h"

using namespace std;

template <typename Dtype>
BatchNormLayer<Dtype>::BatchNormLayer() : LearnableLayer<Dtype>() {
	this->type                  = Layer<Dtype>::BatchNorm;
    this->depth                 = 0;

	this->_paramsInitialized.resize(5);
	this->_paramsInitialized[ParamType::Gamma] = false;
	this->_paramsInitialized[ParamType::Beta] = false;
	this->_paramsInitialized[ParamType::GlobalMean] = false;
	this->_paramsInitialized[ParamType::GlobalVar] = false;
	this->_paramsInitialized[ParamType::GlobalCount] = false;

	this->_params.resize(5);
	this->_params[ParamType::Gamma] = new Data<Dtype>(SLPROP_BASE(name) + "_gamma");
	this->_params[ParamType::Beta] = new Data<Dtype>(SLPROP_BASE(name) + "_beta");
	this->_params[ParamType::GlobalMean] =
        new Data<Dtype>(SLPROP_BASE(name) + "_global_mean");
	this->_params[ParamType::GlobalVar] =
        new Data<Dtype>(SLPROP_BASE(name) + "_global_var");
	this->_params[ParamType::GlobalCount] =
        new Data<Dtype>(SLPROP_BASE(name) + "_global_count");

	this->_paramsHistory.resize(5);
	this->_paramsHistory[ParamType::Gamma] =
        new Data<Dtype>(SLPROP_BASE(name) + "_gamma_history");
	this->_paramsHistory[ParamType::Beta] =
        new Data<Dtype>(SLPROP_BASE(name) + "_beta_history");
	this->_paramsHistory[ParamType::GlobalMean] =
        new Data<Dtype>(SLPROP_BASE(name) + "_global_mean_history");
	this->_paramsHistory[ParamType::GlobalVar] =
        new Data<Dtype>(SLPROP_BASE(name) + "_global_var_history");
	this->_paramsHistory[ParamType::GlobalCount] =
        new Data<Dtype>(SLPROP_BASE(name) + "_global_count_history");

	this->_paramsHistory2.resize(5);
	this->_paramsHistory2[ParamType::Gamma] =
        new Data<Dtype>(SLPROP_BASE(name) + "_gamma_history2");
	this->_paramsHistory2[ParamType::Beta] =
        new Data<Dtype>(SLPROP_BASE(name) + "_beta_history2");
	this->_paramsHistory2[ParamType::GlobalMean] =
        new Data<Dtype>(SLPROP_BASE(name) + "_global_mean_history2");
	this->_paramsHistory2[ParamType::GlobalVar] =
        new Data<Dtype>(SLPROP_BASE(name) + "_global_var_history2");
	this->_paramsHistory2[ParamType::GlobalCount] =
        new Data<Dtype>(SLPROP_BASE(name) + "_global_count_history2");

    this->meanSet               = new Data<Dtype>(SLPROP_BASE(name) + "_mean");
    this->varSet                = new Data<Dtype>(SLPROP_BASE(name) + "_variance");
    this->normInputSet          = new Data<Dtype>(SLPROP_BASE(name) + "_normalizedInput");
}

template<typename Dtype>
void BatchNormLayer<Dtype>::setTrain(bool train) {
    SLPROP(BatchNorm, train) = train;
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* BatchNormLayer<Dtype>::initLayer() {
    BatchNormLayer* layer = new BatchNormLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void BatchNormLayer<Dtype>::destroyLayer(void* instancePtr) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void BatchNormLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool BatchNormLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void BatchNormLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void BatchNormLayer<Dtype>::backwardTensor(void* instancePtr) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void BatchNormLayer<Dtype>::learnTensor(void* instancePtr) {
    BatchNormLayer<Dtype>* layer = (BatchNormLayer<Dtype>*)instancePtr;
    layer->update();
}

template class BatchNormLayer<float>;
