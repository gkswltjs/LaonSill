/**
 * @file LayerFunc.cpp
 * @date 2017-05-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "LayerFunc.h"
#include "Layer.h"
#include "SysLog.h"

CBLayerFunc* LayerFunc::layerFuncs;

void LayerFunc::init() {
    int layerTypeSize = Layer<float>::LayerTypeMax;

    LayerFunc::layerFuncs = (CBLayerFunc*)malloc(sizeof(CBLayerFunc));
    SASSERT0(LayerFunc::layerFuncs != NULL); 
}

void LayerFunc::destroy() {
    SASSERT0(LayerFunc::layerFuncs != NULL);
    free(LayerFunc::layerFuncs);
}

void LayerFunc::registerLayerFunc(int layerTypeID, CBAllocInOutTensor allocInOutTensor,
    CBAllocLayerTensors allocLayerTensors, CBForward forward, CBBackward backward,
    CBLearn learn) {    
    SASSERT0(layerTypeID < Layer<float>::LayerTypeMax);
    LayerFunc::layerFuncs[layerTypeID].allocInOutTensor = allocInOutTensor;
    LayerFunc::layerFuncs[layerTypeID].allocLayerTensors = allocLayerTensors;
    LayerFunc::layerFuncs[layerTypeID].forward = forward;
    LayerFunc::layerFuncs[layerTypeID].backward = backward;
    LayerFunc::layerFuncs[layerTypeID].learn = learn;
}


bool LayerFunc::allocInOutTensor(int layerTypeID, void *tensorPtr, bool isInput, int index) {
    SASSUME0(layerTypeID < Layer<float>::LayerTypeMax);
    return LayerFunc::layerFuncs[layerTypeID].allocInOutTensor(tensorPtr, isInput, index);
}

bool LayerFunc::allocLayerTensors(int layerTypeID) {
    SASSUME0(layerTypeID < Layer<float>::LayerTypeMax);
    return LayerFunc::layerFuncs[layerTypeID].allocLayerTensors();
}

bool LayerFunc::runForward(int layerTypeID) {
    SASSUME0(layerTypeID < Layer<float>::LayerTypeMax);
    return LayerFunc::layerFuncs[layerTypeID].forward();
}

bool LayerFunc::runBackward(int layerTypeID) {
    SASSUME0(layerTypeID < Layer<float>::LayerTypeMax);
    return LayerFunc::layerFuncs[layerTypeID].backward();
}

bool LayerFunc::learn(int layerTypeID) {
    SASSUME0(layerTypeID < Layer<float>::LayerTypeMax);
    return LayerFunc::layerFuncs[layerTypeID].learn();
}
