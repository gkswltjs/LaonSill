/**
 * @file AAALayer_device.cu
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "cuda_runtime.h"

#include "AAALayer.h"
#include "Exception.h"
#include "NetworkConfig.h"
#include "SysLog.h"

using namespace std;

template<typename Dtype>
AAALayer<Dtype>::~AAALayer() {

}

template <typename Dtype>
void AAALayer<Dtype>::feedforward() {

}

template <typename Dtype>
void AAALayer<Dtype>::reshape() {

}

template <typename Dtype>
void AAALayer<Dtype>::backpropagation() {

}

template AAALayer<float>::~AAALayer();
template void AAALayer<float>::reshape();
template void AAALayer<float>::feedforward();
template void AAALayer<float>::backpropagation();
