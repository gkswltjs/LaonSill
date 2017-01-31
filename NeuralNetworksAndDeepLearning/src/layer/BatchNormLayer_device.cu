/**
 * @file BatchNormLayer_device.cu
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "BatchNormLayer.h"
#include "Exception.h"
#include "NetworkConfig.h"
#include "SysLog.h"

using namespace std;

#ifdef GPU_MODE

template<typename Dtype>
BatchNormLayer<Dtype>::~BatchNormLayer() {

}

template <typename Dtype>
void BatchNormLayer<Dtype>::update() {

}

template <typename Dtype>
void BatchNormLayer<Dtype>::feedforward() {

}

template <typename Dtype>
void BatchNormLayer<Dtype>::reshape() {

}

template <typename Dtype>
void BatchNormLayer<Dtype>::backpropagation() {

}

template BatchNormLayer<float>::~BatchNormLayer();
template void BatchNormLayer<float>::reshape();
template void BatchNormLayer<float>::update();
template void BatchNormLayer<float>::feedforward();
template void BatchNormLayer<float>::backpropagation();

#endif
