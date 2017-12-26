/**
 * @file LayerFunc.cpp
 * @date 2017-05-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <iostream>

#include "LayerFunc.h"
#include "BaseLayer.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "LayerPropList.h"
#include "MemoryMgmt.h"

using namespace std;

CBLayerFunc* LayerFunc::layerFuncs;

void LayerFunc::init() {
    int layerTypeSize = Layer<float>::LayerTypeMax;

    LayerFunc::layerFuncs = NULL;
    int allocSize = sizeof(CBLayerFunc) * layerTypeSize;
    SMALLOC_ONCE(LayerFunc::layerFuncs, CBLayerFunc, allocSize);
    SASSERT0(LayerFunc::layerFuncs != NULL); 
}

void LayerFunc::destroy() {
    SASSERT0(LayerFunc::layerFuncs != NULL);
    SFREE(LayerFunc::layerFuncs);
}

void LayerFunc::registerLayerFunc(int layerType, CBInitLayer initLayer,
    CBDestroyLayer destroyLayer, CBSetInOutTensor setInOutTensor,
    CBAllocLayerTensors allocLayerTensors, CBForward forward, CBBackward backward,
    CBLearn learn) {    
    SASSERT0(layerType < Layer<float>::LayerTypeMax);
    LayerFunc::layerFuncs[layerType].initLayer = initLayer;
    LayerFunc::layerFuncs[layerType].destroyLayer = destroyLayer;
    LayerFunc::layerFuncs[layerType].setInOutTensor = setInOutTensor;
    LayerFunc::layerFuncs[layerType].allocLayerTensors = allocLayerTensors;
    LayerFunc::layerFuncs[layerType].forward = forward;
    LayerFunc::layerFuncs[layerType].backward = backward;
    LayerFunc::layerFuncs[layerType].learn = learn;
}

void* LayerFunc::initLayer(int layerType) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "init layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    return LayerFunc::layerFuncs[layerType].initLayer();
}

void LayerFunc::destroyLayer(int layerType, void* instancePtr) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "destroy layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    LayerFunc::layerFuncs[layerType].destroyLayer(instancePtr);
}

void LayerFunc::setInOutTensor(int layerType, void* instancePtr, void *tensorPtr,
    bool isInput, int index) {

    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "set in/out layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    LayerFunc::layerFuncs[layerType].setInOutTensor(instancePtr, tensorPtr, isInput, index);
}

bool LayerFunc::allocLayerTensors(int layerType, void* instancePtr) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "alloc layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    return LayerFunc::layerFuncs[layerType].allocLayerTensors(instancePtr);
}

void LayerFunc::runForward(int layerType, void* instancePtr, int miniBatchIdx) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "forward layer(layer=%s). miniBatchIdx=%d",
            LayerPropList::getLayerName(layerType).c_str(), miniBatchIdx);
    LayerFunc::layerFuncs[layerType].forward(instancePtr, miniBatchIdx);
}

void LayerFunc::runBackward(int layerType, void* instancePtr) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "backward layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    LayerFunc::layerFuncs[layerType].backward(instancePtr);
}

void LayerFunc::learn(int layerType, void* instancePtr) {
    SASSUME0(layerType < Layer<float>::LayerTypeMax);
    STDOUT_COND_LOG(SPARAM(PRINT_LAYERFUNC_LOG), "learn layer(layer=%s).",
            LayerPropList::getLayerName(layerType).c_str());
    LayerFunc::layerFuncs[layerType].learn(instancePtr);
}

