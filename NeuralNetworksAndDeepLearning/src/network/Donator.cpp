/**
 * @file Donator.cpp
 * @date 2017-02-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Donator.h"
#include "Layer.h"
#include "ConvLayer.h"
#include "BatchNormLayer.h"
#include "FullyConnectedLayer.h"
#include "SysLog.h"
#include "ColdLog.h"

using namespace std;

template<typename Dtype>
map<uint32_t, DonatorData>  Donator<Dtype>::donatorMap;
template<typename Dtype>
mutex                       Donator<Dtype>::donatorMutex;

template<typename Dtype>
void Donator<Dtype>::donate(uint32_t donatorID, void* layerPtr) {
    DonatorData newData;
    newData.donatorID = donatorID;
    newData.refCount = 0;
    newData.layerPtr = layerPtr;
    newData.cleanUp = false;

    unique_lock<mutex> donatorLock(Donator::donatorMutex);
    SASSERT((Donator::donatorMap.count(donatorID) == 0), "already layer(ID=%u) donated.",
        donatorID);

    Donator::donatorMap[donatorID] = newData;
}

template<typename Dtype>
void Donator<Dtype>::receive(uint32_t donatorID, void* layerPtr) {
    unique_lock<mutex> donatorLock(Donator::donatorMutex);
    SASSERT((Donator::donatorMap.count(donatorID) == 1), "there is no donator(ID=%u).",
        donatorID);
    DonatorData data = Donator::donatorMap[donatorID];
    data.refCount += 1;
    donatorLock.unlock();

    // FIXME: dangerous casting.. should be fixed in the futre
    Layer<Dtype>* donatorLayer = (Layer<Dtype>*)data.layerPtr;
    Layer<Dtype>* receiverLayer = (Layer<Dtype>*)layerPtr;

    LearnableLayer<Dtype>* donatorLearnableLayer = (LearnableLayer<Dtype>*)data.layerPtr;
    LearnableLayer<Dtype>* receiverLearnableLayer = (LearnableLayer<Dtype>*)layerPtr;
   
    SASSERT(donatorLayer->type == receiverLayer->type,
        "both donator and receiver should have same layer type."
        "donator layer type=%d, receiver layer type=%d",
        (int)donatorLayer->type, (int)receiverLayer->type);

    if (donatorLayer->type == Layer<Dtype>::Conv || 
        donatorLayer->type == Layer<Dtype>::Deconv) {
        ConvLayer<Dtype>* donator = dynamic_cast<ConvLayer<Dtype>*>(donatorLayer);
        ConvLayer<Dtype>* receiver = dynamic_cast<ConvLayer<Dtype>*>(receiverLayer);
        donator->donateParam(receiver);
    } else if (donatorLayer->type == Layer<Dtype>::FullyConnected) {
        FullyConnectedLayer<Dtype>* donator =
            dynamic_cast<FullyConnectedLayer<Dtype>*>(donatorLayer);
        FullyConnectedLayer<Dtype>* receiver = 
            dynamic_cast<FullyConnectedLayer<Dtype>*>(receiverLayer);
        donator->donateParam(receiver);
    } else if (donatorLayer->type == Layer<Dtype>::BatchNorm) {
        BatchNormLayer<Dtype>* donator = dynamic_cast<BatchNormLayer<Dtype>*>(donatorLayer);
        BatchNormLayer<Dtype>* receiver = dynamic_cast<BatchNormLayer<Dtype>*>(receiverLayer);
        donator->donateParam(receiver);
    } else {
        COLD_LOG(ColdLog::WARNING, true, "layer(type=%d) does not support donate function",
            donatorLayer->type);
    }
}

template<typename Dtype>
void Donator<Dtype>::releaseDonator(uint32_t donatorID) {
    unique_lock<mutex> donatorLock(Donator::donatorMutex);
    SASSERT((Donator::donatorMap.count(donatorID) == 1), "there is no donator(ID=%u).",
        donatorID);
    DonatorData data = Donator::donatorMap[donatorID];

    if (data.refCount > 0) {
        data.cleanUp = true;
    } else {
        Layer<Dtype>* layer = (Layer<Dtype>*)data.layerPtr;
        delete layer;
        Donator::donatorMap.erase(donatorID);
    }
}

template<typename Dtype>
void Donator<Dtype>::releaseReceiver(uint32_t donatorID) {
    unique_lock<mutex> donatorLock(Donator::donatorMutex);
    SASSERT((Donator::donatorMap.count(donatorID) == 1), "there is no donator(ID=%u).",
        donatorID);
    DonatorData data = Donator::donatorMap[donatorID];
   
    data.refCount -= 1;
    if ((data.refCount == 0) && data.cleanUp) {
        Layer<Dtype>* layer = (Layer<Dtype>*)data.layerPtr;
        delete layer;
        Donator::donatorMap.erase(donatorID);
    }
}

template class Donator<float>;
