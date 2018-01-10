/**
 * @file YOLODetectionOutputLayer.cpp
 * @date 2018-01-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "YOLODetectionOutputLayer.h"
#include "PropMgmt.h"

using namespace std;

template <typename Dtype>
YOLODetectionOutputLayer<Dtype>::YOLODetectionOutputLayer()
: Layer<Dtype>() {
	this->type = Layer<Dtype>::YOLODetectionOutput;
}


template <typename Dtype>
YOLODetectionOutputLayer<Dtype>::~YOLODetectionOutputLayer() {
}

template class YOLODetectionOutputLayer<float>;
