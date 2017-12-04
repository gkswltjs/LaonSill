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
#include <string>
#include <iostream>
#include <limits>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>

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
#include "LogicalPlan.h"
#include "MeasureManager.h"
#include "FileMgmt.h"

using namespace std;
using namespace boost::uuids;

extern const char*  LAONSILL_HOME_ENVNAME;

template<typename Dtype>
map<string, Network<Dtype>*>   Network<Dtype>::networkIDMap;
template<typename Dtype>
mutex Network<Dtype>::networkIDMapMutex;

template <typename Dtype>
Network<Dtype>::Network() {
    random_generator gen;
    uuid id = gen();
    this->networkID = to_string(id);

    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    Network<Dtype>::networkIDMap[this->networkID] = this;
    this->isLoaded = false;
    this->isBuilt = false;
    this->isMeasureInserted = false;

    this->bestLoss = numeric_limits<float>::max();
    this->bestSavedParamPath = "";

    // train 정보를 관리하는 파일 포인터를 얻는다.
    string trainFilePath = string(getenv(LAONSILL_HOME_ENVNAME)) + "/param/" +
            this->networkID + ".train";
    this->trainFP = fopen(trainFilePath.c_str(), "w+");
    SASSERT0(this->trainFP != NULL);

}

template <typename Dtype>
Network<Dtype>::~Network() {
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    Network<Dtype>::networkIDMap.erase(this->networkID);

    if (this->isMeasureInserted) {
        MeasureManager::removeEntry(this->networkID);
    }

    if (this->trainFP != NULL)
        fclose(this->trainFP);
}

template<typename Dtype>
void Network<Dtype>::init() {
}

template<typename Dtype>
Network<Dtype>* Network<Dtype>::getNetworkFromID(string networkID) {
    Network<Dtype>* network;
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    network = Network<Dtype>::networkIDMap[networkID];
    lock.unlock();
    return network;
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

    SASSERT0(this->isMeasureInserted == false);

    if (SNPROP(measureLayer).size() > 0) {
        MeasureManager::insertEntry(this->networkID, SNPROP(measureLayer));
        this->isMeasureInserted = true;
    }

    PlanOptimizer::buildPlans(networkID);
}

template<typename Dtype>
void Network<Dtype>::reset() {
    SASSERT0(this->isLoaded);

    WorkContext::updateNetwork(this->networkID); 

    PlanInfo* planInfo = WorkContext::curPlanInfo;
    planInfo->curEpochIndex = 0;
    planInfo->curMiniBatchIndex = 0;
    planInfo->doneCount = 0;
    SNPROP(iterations) = 0;
    
    for (int i = 0; i < planInfo->dopCount; i++) {
        WorkContext::updatePlan(i, true);
        PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
        pp->reset();
    }
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
    WorkContext::updatePlan(WorkContext::curDOPID, true);

    PlanInfo* planInfo = WorkContext::curPlanInfo;

    SASSERT0(miniBatchIdx >= 0);

    planInfo->curMiniBatchIndex = miniBatchIdx;
    planInfo->curEpochIndex = 0;
    planInfo->miniBatchCount = miniBatchIdx + 1;
    planInfo->epochCount = 1;
    planInfo->doneCount = 0;

    for (int i = 0; i < planInfo->dopCount; i++) {
        WorkContext::updatePlan(i, true);
        PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
        pp->reset();
    }

    PlanOptimizer::runPlan(this->networkID, inference);
}

template<typename Dtype>
void Network<Dtype>::save(string path) {
	// save learned params
	ofstream paramOfs(path.c_str(), ios::out | ios::binary);

	uint32_t numParams = 0;
    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(0, true);   // 아무 dopID에서 가져가도 상관없을꺼 같다.
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

    if (SPARAM(PRINT_PARAMLOG_AFTER_NETWORKSAVE)) {
        DebugUtil<Dtype>::printNetworkParams(stderr, "network save result",
            this->networkID, 0);
    }
}

template <typename Dtype>
string Network<Dtype>::save() {
    string path;
	if (SNPROP(savePathPrefix) == "") {
        path = string(getenv(LAONSILL_HOME_ENVNAME)) + "/param/" +
            this->networkID + "_" +
            to_string(SNPROP(iterations)) + ".param";
    } else {
        path = SNPROP(savePathPrefix) + + "_" + to_string(SNPROP(iterations)) + ".param";
    }

    save(path);
    return path;
}

template<typename Dtype>
void Network<Dtype>::handleIntervalSaveParams(int iterNum) {
    if (this->intervalSavedParamPathQueue.size() == SNPROP(keepSaveIntervalModelCount)) {
        string removeParamPath = this->intervalSavedParamPathQueue.front();
        this->intervalSavedParamPathQueue.pop();
        FileMgmt::removeFile(removeParamPath.c_str());
    }

    string newParamPath = this->save();
    this->intervalSavedParamPathQueue.push(newParamPath);

    logTrainFile(to_string(iterNum) + "," + newParamPath);
}

template<typename Dtype>
void Network<Dtype>::handleBestLoss(float loss, int iterNum) {
    if (!SNPROP(keepSaveBestModel))
        return;

    if (SNPROP(keepSaveBestModelStartIterNum) > iterNum)
        return;

    if (loss > this->bestLoss)
        return; 

    this->bestLoss = loss;

    // XXX: remove file 하고 나서 best model을 저장하는 순간에 서버가 죽으면 좀 난감하다.
    //      이 부분에 대한 고려가 필요하다.
    string newParamPath = string(getenv(LAONSILL_HOME_ENVNAME)) + "/param/" +
        this->networkID + "_best_" + to_string(iterNum) + ".param";

    this->save(newParamPath);

    if (this->bestSavedParamPath != "")
        FileMgmt::removeFile(this->bestSavedParamPath.c_str()); 
    this->bestSavedParamPath = newParamPath;

    logTrainFile("best(" + to_string(iterNum) + ")," + newParamPath);
}

template <typename Dtype>
void Network<Dtype>::load(string path) {
    ifstream ifs(path, std::ios::in | std::ios::binary);

    SASSERT0(ifs.is_open());

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

    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(0, true);   // 아무 dopID에서 가져가도 상관없을꺼 같다.
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

    if (SPARAM(PRINT_PARAMLOG_AFTER_NETWORKLOAD)) {
        DebugUtil<Dtype>::printNetworkParams(stderr, "network load result",
            this->networkID, 0);
    }
}

template <typename Dtype>
void Network<Dtype>::load() {
    load(SNPROP(loadPath));
}

template <typename Dtype>
Layer<Dtype>* Network<Dtype>::findLayer(const string layerName) {
    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(WorkContext::curDOPID, true);
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

    if (foundLayer)
        return layer;
    else
        return NULL;
}

template <typename Dtype>
vector<Layer<Dtype>*> Network<Dtype>::findLayersByType(int layerType) {
    vector<Layer<Dtype>*> result;

    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(WorkContext::curDOPID, true);
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

    return result;
}

template<typename Dtype>
Data<Dtype>* Network<Dtype>::findTensor(int nodeID, int devID, string tensorName) {
    WorkContext::updateNetwork(this->networkID);
    WorkContext::updatePlan(WorkContext::curDOPID, true);
    PhysicalPlan* pp = WorkContext::curPhysicalPlan;

    // XXX: does not consider multi-device, multi-node situation
    Data<Dtype>* result = (Data<Dtype>*)pp->getTensor(nodeID, devID, tensorName);

    return result;
}

template<typename Dtype>
bool Network<Dtype>::isInnerLayer(int layerID) {
    if (layerID >= SPARAM(SPLITLAYER_START_LAYERID))
        return false;

    return LogicalPlan::isInnerLayer(this->networkID, layerID);
}

template<typename Dtype>
void Network<Dtype>::logNetworkDefString(string networkDef) {
    SASSERT0(this->trainFP != NULL);
    fprintf(this->trainFP, "%s\n", networkDef.c_str());
    fprintf(this->trainFP, "=========================================================\n\n");
    fflush(this->trainFP);
}

template<typename Dtype>
void Network<Dtype>::logNetworkDefFile(string networkDefFilePath) {
    std::ifstream file(networkDefFilePath);
    std::string content((std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>());
    logNetworkDefString(content);
}

template<typename Dtype>
void Network<Dtype>::logTrainFile(string content) {
    SASSUME0(this->trainFP != NULL);
    fprintf(this->trainFP, "%s\n", content.c_str());
    fflush(this->trainFP);
}

template class Network<float>;
