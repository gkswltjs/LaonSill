/*
 * Network.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include <stdlib.h>

#include <vector>
#include <map>
#include <cfloat>

#include "DataSet.h"
#include "BaseLayer.h"
#include "SoftmaxWithLossLayer.h"
#include "LossLayer.h"
#include "Util.h"
#include "Worker.h"
#include "Perf.h"
#include "StdOutLog.h"
#include "Network.h"
#include "SysLog.h"
#include "DebugUtil.h"
#include "WorkContext.h"
#include "PhysicalPlan.h"
#include "PlanOptimizer.h"
#include "PropMgmt.h"
#include "LearnableLayer.h"

using namespace std;

extern const char*  SOOOA_HOME_ENVNAME;

template<typename Dtype>
atomic<int>         Network<Dtype>::networkIDGen;
template<typename Dtype>
map<int, Network<Dtype>*> Network<Dtype>::networkIDMap;
template<typename Dtype>
mutex Network<Dtype>::networkIDMapMutex;

template <typename Dtype>
Network<Dtype>::Network() {
    this->networkID = atomic_fetch_add(&Network<Dtype>::networkIDGen, 1);
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    Network<Dtype>::networkIDMap[this->networkID] = this;
    this->isLoaded = false;
}

template<typename Dtype>
void Network<Dtype>::init() {
    atomic_store(&Network<Dtype>::networkIDGen, 0);
}

template<typename Dtype>
Network<Dtype>* Network<Dtype>::getNetworkFromID(int networkID) {
    Network<Dtype>* network;
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    network = Network<Dtype>::networkIDMap[networkID];
    lock.unlock();
    return network;
}

template <typename Dtype>
Network<Dtype>::~Network() {
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    Network<Dtype>::networkIDMap.erase(this->networkID);
}

template <typename Dtype>
void Network<Dtype>::run_with_timer( bool inference) {
    struct timespec startTime;
    SPERF_START(NETWORK_TRAINING_TESTTIME, &startTime);
	run(inference);

    SPERF_END(NETWORK_TRAINING_TESTTIME, startTime, SNPROP(epochs));
    STDOUT_BLOCK(cout << "Total Training Time : " << SPERF_TIME(NETWORK_TRAINING_TESTTIME)
                    << endl;);
}

template<typename Dtype>
void Network<Dtype>::build(int epochs) {
    SASSERT0(this->isLoaded);
        
    WorkContext::updateNetwork(this->networkID); 
    SNPROP(epochs) = epochs;

    PlanOptimizer::buildPlans(networkID);
}

template<typename Dtype>
void Network<Dtype>::reset() {
    SASSERT0(this->isLoaded);

    WorkContext::updateNetwork(this->networkID); 

    PlanInfo* planInfo = WorkContext::curPlanInfo;
    planInfo->curEpochIndex = 0;
    planInfo->curMiniBatchIndex = -1;
    SNPROP(iterations) = 0;
}

template<typename Dtype>
void Network<Dtype>::run(bool inference) {
    SASSERT0(this->isLoaded);
    PlanOptimizer::runPlan(this->networkID, inference);
}

template<typename Dtype>
void Network<Dtype>::runPlanType(PlanType planType, bool inference) {
    SASSERT0(this->isLoaded);
    PlanOptimizer::runPlanByType(this->networkID, planType, inference);
}

template<typename Dtype>
void Network<Dtype>::runMiniBatch(bool inference, int miniBatchIdx) {
    SASSERT0(this->isLoaded);

    WorkContext::updateNetwork(this->networkID); 
    WorkContext::updatePlan(WorkContext::curDOPID);

    PlanInfo* planInfo = WorkContext::curPlanInfo;

    SASSERT0(miniBatchIdx >= 0);
    SASSERT0(miniBatchIdx < planInfo->miniBatchCount);

    int oldEpochIdx = planInfo->curEpochIndex;
    int oldMiniBatchIdx = planInfo->curMiniBatchIndex;
    int oldEpochCount = planInfo->epochCount;
    int oldMiniBatchCount = planInfo->miniBatchCount;

    planInfo->curMiniBatchIndex = miniBatchIdx - 1;
    planInfo->curEpochIndex = 0;
    planInfo->miniBatchCount = miniBatchIdx + 1;
    planInfo->epochCount = 1;
    SNPROP(iterations) = 0;

    PlanOptimizer::runPlan(this->networkID, inference);

    planInfo->curEpochIndex = oldEpochIdx;
    planInfo->curMiniBatchIndex = oldMiniBatchIdx;
    planInfo->epochCount = oldEpochCount;
    planInfo->miniBatchCount = oldMiniBatchCount;
}

template<typename Dtype>
void Network<Dtype>::save(string path) {
	// save learned params
	ofstream paramOfs(path.c_str(), ios::out | ios::binary);

	uint32_t numParams = 0;
    int oldNetworkID = WorkContext::curNetworkID;
    WorkContext::updateNetwork(this->networkID);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);
        if (!SLPROP_BASE(learnable))
            continue;

        LearnableLayer<Dtype>* ll = (LearnableLayer<Dtype>*)instancePtr;
        numParams += ll->numParams();
    }

	paramOfs.write((char*)&numParams, sizeof(uint32_t));
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);
        if (!SLPROP_BASE(learnable))
            continue;

        LearnableLayer<Dtype>* ll = (LearnableLayer<Dtype>*)instancePtr;
        ll->saveParams(paramOfs);
    }

	paramOfs.close();

    if (SPARAM(PRINT_EDGELOG_AFTER_NETWORKSAVE)) {
        DebugUtil<Dtype>::printNetworkEdges(stderr, "network save result", this->networkID,
            0);
    }

    WorkContext::updateNetwork(oldNetworkID);

}

template <typename Dtype>
void Network<Dtype>::save() {
    string path;
	if (SNPROP(savePathPrefix) == "") {
        path = string(getenv(SOOOA_HOME_ENVNAME)) + "/network/" +
            to_string(SNPROP(iterations)) + ".param";
    } else {
        path = SNPROP(savePathPrefix) + to_string(SNPROP(iterations)) + ".param";
    }

    save(path);
}

template <typename Dtype>
void Network<Dtype>::load(string path) {
    ifstream ifs(path, std::ios::in | std::ios::binary);

    // TODO : 반드시 구현 필요
	// load data list from model file
	map<std::string, Data<float>*> dataMap;


    uint32_t numData;
    ifs.read((char*)&numData, sizeof(uint32_t));

    Data<float>::printConfig = true;
    cout << "Load Pretrained Weights ... ----------" << endl;
    for (uint32_t j = 0; j < numData; j++) {
        Data<float>* data = new Data<float>("", true);
        data->load(ifs);

        if (data)
            data->print();

        string dataName;
        dataName = data->_name;

        map<string, Data<float>*>::iterator it;
        it = dataMap.find(dataName);
        if (it != dataMap.end()) {
            cout << dataName << " overwrites ... " << endl;
            delete it->second;
        }

        dataMap[dataName] = data;
        cout << data->_name << " is set to " << dataName << endl;
    }
    cout << "--------------------------------------" << endl;
    Data<float>::printConfig = false;
    ifs.close();

    int oldNetworkID = WorkContext::curNetworkID;
    WorkContext::updateNetwork(this->networkID);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);
        if (!SLPROP_BASE(learnable))
            continue;

        LearnableLayer<Dtype>* ll = (LearnableLayer<Dtype>*)instancePtr;
        ll->loadParams(dataMap);
    }

	map<std::string, Data<float>*>::iterator it;
	for (it = dataMap.begin(); it != dataMap.end(); it++)
		delete it->second;
	dataMap.clear();

    if (SPARAM(PRINT_EDGELOG_AFTER_NETWORKLOAD)) {
        DebugUtil<Dtype>::printNetworkEdges(stderr, "network load result", this->networkID,
            0);
    }

    WorkContext::updateNetwork(oldNetworkID);

}

template <typename Dtype>
void Network<Dtype>::load() {
    load(SNPROP(loadPath));
}

template <typename Dtype>
Layer<Dtype>* Network<Dtype>::findLayer(const string layerName) {
    int oldNetworkID = WorkContext::curNetworkID;
    WorkContext::updateNetwork(this->networkID);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    Layer<Dtype>* layer;
    bool foundLayer = false;
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);

        layer = (Layer<Dtype>*)instancePtr;

        // FIXME: 현재 linear search. 너무 속도가 느리면 개선하자.
        if (SLPROP_BASE(name) == layerName) {
            foundLayer = true;
            break;
        }
    }

    WorkContext::updateNetwork(oldNetworkID);

    if (foundLayer)
        return layer;
    else
        return NULL;
}

template <typename Dtype>
vector<Layer<Dtype>*> Network<Dtype>::findLayersByType(int layerType) {
    vector<Layer<Dtype>*> result;

    int oldNetworkID = WorkContext::curNetworkID;
    WorkContext::updateNetwork(this->networkID);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    bool foundLayer = false;
    for (map<int, void*>::iterator iter = pp->instanceMap.begin();
        iter != pp->instanceMap.end(); iter++) {
        int layerID = iter->first;
        void* instancePtr = iter->second;

        WorkContext::updateLayer(this->networkID, layerID);

        // FIXME: 현재 linear search. 너무 속도가 느리면 개선하자.
        if (WorkContext::curLayerProp->layerType == layerType) {
            result.push_back((Layer<Dtype>*)instancePtr);
        }
    }

    WorkContext::updateNetwork(oldNetworkID);

    return result;
}

template<typename Dtype>
Data<Dtype>* Network<Dtype>::findTensor(int nodeID, int devID, string tensorName) {
    int oldNetworkID = WorkContext::curNetworkID;
    WorkContext::updateNetwork(this->networkID);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    // XXX: does not consider multi-device, multi-node situation
    Data<Dtype>* result = (Data<Dtype>*)pp->getTensor(nodeID, devID, tensorName);

    WorkContext::updateNetwork(oldNetworkID);

    return result;
}

template class Network<float>;
