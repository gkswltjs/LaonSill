/**
 * @file PhysicalPlan.cpp
 * @date 2017-05-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PhysicalPlan.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "WorkContext.h"
#include "Param.h"
#include "Worker.h"
#include "LayerFunc.h"
#include "StdOutLog.h"
#include "LearnableLayer.h"
#include "Donator.h"
#include "Network.h"

using namespace std;

map<int, vector<PhysicalPlan*>>     PhysicalPlan::planGlobalMap;
map<int, PlanInfo*>                 PhysicalPlan::planGlobalInfoMap;
mutex                               PhysicalPlan::planGlobalMutex;

PhysicalPlan::~PhysicalPlan() {
    for (map<int, void*>::iterator iter = instanceMap.begin(); iter != instanceMap.end();
        ++iter) {

        int layerID = iter->first;
        void* instancePtr = iter->second;

        int forwardPlanID = LP_FORWARD_PLANID(layerID);
        if (planMap.find(forwardPlanID) != planMap.end()) {
            int layerType = planMap[forwardPlanID].layerType;
            LayerFunc::destroyLayer(layerType, instancePtr);
        }
    }
}

void* PhysicalPlan::allocTensorMem(int layerType, void* instancePtr, string tensorName,
    PlanAlloc planAlloc, bool isInput, int index) {
    if (planAlloc.nodeID != SPARAM(NODE_ID)) {
        // allocate GPU memory of other nodes.
        // TODO: 구현!!
        SASSERT(false, "not implemented yet");
    }

    int oldGPUIdx = Worker::gpuIdx;
    if (planAlloc.devID != oldGPUIdx) {
        cout << "set device!!" << endl;
        //checkCudaErrors(cudaSetDevice(planAlloc.devID)); 
    }

    // XXX: float형 코딩으로 박지 말고, 설정에 따라서 template date type을 설정하도록 수정해야
    //     한다. 
    Data<float>* tensor = new Data<float>(tensorName);
    SASSERT0(tensor != NULL);

    if (planAlloc.devID != oldGPUIdx) {
        cout << "set device!!" << endl;
        //checkCudaErrors(cudaSetDevice(oldGPUIdx));
    }

    LayerFunc::setInOutTensor(layerType, instancePtr, (void*)tensor, isInput, index);

    return (void*)tensor;
}

void PhysicalPlan::allocateTensorInternal(int networkID) {
    map<TensorAllocKey, void*> tensorAllocMap;
    
    for (map<int, PlanAlloc>::iterator iter = this->allocMap.begin();
        iter !=this->allocMap.end(); iter++) {
        int layerID = iter->first;
        PlanAlloc planAlloc = iter->second;

        WorkContext::updateLayer(networkID, layerID);
        vector<string> inputs = SLPROP_BASE(input);
        vector<string> outputs = SLPROP_BASE(output);

        SASSUME0(planMap.find(LP_FORWARD_PLANID(layerID)) != planMap.end());
        int layerType = planMap[LP_FORWARD_PLANID(layerID)].layerType;
        // When you get the layer type, you can use any plan ID that corresponds to the layer
        // ID

        // (0) initialize layer instance
        SASSUME0(this->instanceMap.find(layerID) == this->instanceMap.end());
        void* instancePtr = LayerFunc::initLayer(layerType);
        SASSUME0(instancePtr != NULL);
        this->instanceMap[layerID] = instancePtr;

        // (1) allocate input/output tensor
        for (int i = 0; i < inputs.size(); i++) {
            TensorAllocKey key;
            key.tensorAlloc = planAlloc;
            key.tensorName = inputs[i];

            if (tensorAllocMap.find(key) == tensorAllocMap.end()) {
                void* allocPtr = PhysicalPlan::allocTensorMem(layerType, instancePtr,
                    key.tensorName, key.tensorAlloc, true, i);
                SASSERT0(allocPtr != NULL);
                tensorAllocMap[key] = allocPtr;
            } else {
                void* tensor = tensorAllocMap[key];
                LayerFunc::setInOutTensor(layerType, instancePtr, tensor, true, i);
            }
        }

        for (int i = 0; i < outputs.size(); i++) {
            TensorAllocKey key;
            key.tensorAlloc = planAlloc;
            key.tensorName = outputs[i];

            if (tensorAllocMap.find(key) == tensorAllocMap.end()) {
                void* allocPtr = PhysicalPlan::allocTensorMem(layerType, instancePtr,
                    key.tensorName, key.tensorAlloc, false, i);
                SASSERT0(allocPtr != NULL);
                tensorAllocMap[key] = allocPtr;
            } else {
                void* tensor = tensorAllocMap[key];
                LayerFunc::setInOutTensor(layerType, instancePtr, tensor, false, i);
            }
        }

        SASSERT0(LayerFunc::allocLayerTensors(layerType, instancePtr) == true);

        if (SLPROP_BASE(learnable)) {
            if (SLPROP_BASE(donate))
                SLPROP_BASE(donatorID) = SLPROP_BASE(id);

            LearnableLayer<float>* learnableLayer = (LearnableLayer<float>*)instancePtr;

            if (SLPROP_BASE(receive)) {
                SASSERT0(!SLPROP_BASE(donate));
                SASSERT0(SLPROP_BASE(donatorID) >= 0);
                Donator<float>::receive(SLPROP_BASE(donatorID), instancePtr);
            }

            if (SLPROP_BASE(donate)) {
                SASSERT0(!SLPROP_BASE(receive));
                SASSERT0(SLPROP_BASE(donatorID) >= 0);
                Donator<float>::donate(SLPROP_BASE(donatorID), instancePtr);
            }
        }
    }
}

void PhysicalPlan::allocateTensor(int networkID) {
    // FIXME: plan lock의 범위가 너무 넓다..
    unique_lock<mutex> planLock(PhysicalPlan::planGlobalMutex);
    SASSUME0(PhysicalPlan::planGlobalInfoMap.find(networkID) !=
            PhysicalPlan::planGlobalInfoMap.end());
    PlanInfo *planInfo = PhysicalPlan::planGlobalInfoMap[networkID];

    SASSUME0(PhysicalPlan::planGlobalMap.find(networkID) !=
        PhysicalPlan::planGlobalMap.end());
    SASSUME0(PhysicalPlan::planGlobalMap[networkID].size() == planInfo->dopCount);

    for (int i = 0; i < planInfo->dopCount; i++) {
        PhysicalPlan* curPP = PhysicalPlan::planGlobalMap[networkID][i];
        curPP->allocateTensorInternal(networkID);
    }
}

void PhysicalPlan::markFinish(int targetPlanID) {
    unique_lock<mutex> planLock(this->planMutex);

    SASSUME(this->depRefMap.find(targetPlanID) != this->depRefMap.end(),
        "There is no ref map for requesting plan ID. targetPlanID=%d", targetPlanID);
    this->depRefMap[targetPlanID] -= 1;
    
    if (this->depRefMap[targetPlanID] == 0) {
        this->readyQueue.push_back(targetPlanID); 
    }

    this->refCount -= 1;
    planLock.unlock();
    SASSUME0(this->depRefMap[targetPlanID] >= 0);
}

void PhysicalPlan::markFinish(int networkID, int dopID, int planID) {
    int oldNetworkID = WorkContext::curNetworkID;
    int oldDOPID = WorkContext::curDOPID;

    WorkContext::updateNetwork(networkID);
    WorkContext::updatePlan(dopID);

    PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
    PlanDef planDef = pp->planMap[planID];
    for (int i = 0; i < planDef.notifyList.size(); i++) {
        int targetPlanID = planDef.notifyList[i];
        pp->markFinish(targetPlanID);
    }

    WorkContext::updateNetwork(oldNetworkID);
    WorkContext::updatePlan(oldDOPID);
}

void PhysicalPlan::saveNetwork(bool checkCond) {
    if (checkCond) {
        bool saveNetwork = false;

        if ((SNPROP(saveInterval) != 0) &&
            ((SNPROP(iterations) % SNPROP(saveInterval)) == 0)) {
            saveNetwork = true;
        } 

        if (!saveNetwork)
            return;
    }

    int networkID = WorkContext::curNetworkID;
    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->save();
}

void PhysicalPlan::loadNetwork() {
    if (SNPROP(loadPath) == "")
        return;
    
    int networkID = WorkContext::curNetworkID;
    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->load();

}

bool PhysicalPlan::generatePlan() {
    // (1) mini batch를 다 돌았는지 확인한다.
    // FIXME: plan lock의 범위가 좀 넓다.. 최적화 고민해보자.
    unique_lock<mutex> planLock(this->planMutex);
    if (this->refCount > 0) {
        planLock.unlock();
        return true;
    }

    // (2) plan info의 curMiniBatchIndex, curEpochIndex를 갱신한다.
    unique_lock<mutex> planInfoLock(WorkContext::curPlanInfo->planMutex);
    this->epochIdx = WorkContext::curPlanInfo->curEpochIndex;
    this->miniBatchIdx = WorkContext::curPlanInfo->curMiniBatchIndex;

    if (WorkContext::curPlanInfo->curMiniBatchIndex == 
            WorkContext::curPlanInfo->miniBatchCount - 1) {
        WorkContext::curPlanInfo->curMiniBatchIndex = 0;
        WorkContext::curPlanInfo->curEpochIndex += 1;
    } else {
        WorkContext::curPlanInfo->curMiniBatchIndex += 1;
    }
    SNPROP(iterations) += 1;

    bool saveNetwork = false;
    if ((SNPROP(saveInterval) != 0) &&
        ((SNPROP(iterations) % SNPROP(saveInterval)) == 0)) {
        saveNetwork = true;
    }

    if (WorkContext::curPlanInfo->curEpochIndex >= WorkContext::curPlanInfo->epochCount) {
        WorkContext::curPlanInfo->curEpochIndex -= 1;
        WorkContext::curPlanInfo->curMiniBatchIndex =
            WorkContext::curPlanInfo->miniBatchCount - 1;

        planInfoLock.unlock();
        planLock.unlock();

        if (saveNetwork)
            PhysicalPlan::saveNetwork(false);
            
        return false;
    }
    planInfoLock.unlock();

    // (3) 초기화를 수행한다.
    this->refCount = 0;
    this->readyQueue = {};
    for (map<int, PlanDef>::iterator it = planMap.begin(); it != planMap.end(); ++it) {
        int key = it->first;
        PlanDef value = it->second;
      
        depRefMap[key] = value.depCount;

        if (value.depCount == 0) {
            readyQueue.push_back(key);
        } else {
            this->refCount += 1;
        }
    }

    planLock.unlock();

    if (saveNetwork)
        PhysicalPlan::saveNetwork(false);

    return true;
}

void PhysicalPlan::runLayer(int planID, bool inference) {
    // (1) set context
    int layerID = LP_PLANID_TO_LAYERID(planID);
    WorkContext::updateLayer(WorkContext::curNetworkID, layerID);
    PlanType planType = LP_PLANID_TO_PLANTYPE(planID);
    
    // FIXME: 나름 핫 코드영역인데 이렇게 자주 맵을 뒤지면 성능에 안좋다. 수정필요!!
    SASSUME0(this->planMap.find(planID) != this->planMap.end());
    int layerType = this->planMap[planID].layerType;

    // (2) run layer
    bool doMarkFinish = true;
    if (!inference || (planType == PLANTYPE_FORWARD)) {
        PlanInfo* planInfo = WorkContext::curPlanInfo;
        STDOUT_COND_BLOCK(SPARAM(PRINT_RUNLAYER_LOG), 
        cout << "Epoch : " << planInfo->curEpochIndex << ", minibatch : " << 
            planInfo->curMiniBatchIndex << " run layer (planID=" << planID << ")" << endl);


        SASSUME0(this->instanceMap.find(layerID) != this->instanceMap.end());
        void* instancePtr = this->instanceMap[layerID];

        SASSUME0(planType < PLANTYPE_MAX);
        if (planType == PLANTYPE_FORWARD) {
            LayerFunc::runForward(layerType, instancePtr, 
                                                 planInfo->curMiniBatchIndex);
        } else if (planType == PLANTYPE_BACKWARD) {
            LayerFunc::runBackward(layerType, instancePtr);
        } else {
            SASSUME0(planType == PLANTYPE_UPDATE);
            LayerFunc::learn(layerType, instancePtr);
            doMarkFinish = false;       // update는 내부적으로 mark finish한다.
        }
    }

    // (3) mark
    if (!doMarkFinish)
        return;

    PlanDef *planDef = &WorkContext::curPhysicalPlan->planMap[planID];
    for (int i = 0; i < planDef->notifyList.size(); i++) {
        int targetPlanID = planDef->notifyList[i];
        markFinish(targetPlanID);
    }
}

bool PhysicalPlan::runPlan(int layerID, PlanType planType, bool inference) {
    unique_lock<mutex> planLock(this->planMutex);

    if (find(this->readyQueue.begin(), this->readyQueue.end(), layerID) != 
        this->readyQueue.end())
        return false;

    planLock.unlock();

    if (planType == PLANTYPE_FORWARD) {
        runLayer(LP_FORWARD_PLANID(layerID), inference);
    } else if (planType == PLANTYPE_BACKWARD) {
        runLayer(LP_BACKWARD_PLANID(layerID), inference);
    } else if (planType == PLANTYPE_UPDATE) {
        runLayer(LP_UPDATE_PLANID(layerID), inference);
    } else {
        SASSERT(false, "invalid plan type. plan type=%d", (int)planType);
    }

    return true;
}

bool PhysicalPlan::runPlan(bool inference) {
    unique_lock<mutex> planLock(this->planMutex);
    
    if (this->readyQueue.size() == 0) {
        return false;
    }

    int planID = this->readyQueue.front();
    this->readyQueue.pop_front();
    planLock.unlock();

    SASSUME0(this->planMap.find(planID) != this->planMap.end());

    runLayer(planID, inference);
    return true;
}

void PhysicalPlan::insertPlan(int networkID, vector<PhysicalPlan*> pMap, PlanInfo *pInfoMap) {
    unique_lock<mutex> planLock(PhysicalPlan::planGlobalMutex);
    SASSERT0(PhysicalPlan::planGlobalMap.find(networkID) == 
            PhysicalPlan::planGlobalMap.end());
    PhysicalPlan::planGlobalMap[networkID] = {};
    for (int i = 0; i < pMap.size(); i++) {
        PhysicalPlan::planGlobalMap[networkID].push_back(pMap[i]);
    }

    SASSERT0(PhysicalPlan::planGlobalInfoMap.find(networkID) ==
            PhysicalPlan::planGlobalInfoMap.end());

    PhysicalPlan::planGlobalInfoMap[networkID] = pInfoMap;
}

void PhysicalPlan::removePlan(int networkID) {
    unique_lock<mutex> planLock(PhysicalPlan::planGlobalMutex);
    SASSERT0(PhysicalPlan::planGlobalMap.find(networkID) != 
            PhysicalPlan::planGlobalMap.end());

    vector<PhysicalPlan*>::iterator iter;
    for (iter = PhysicalPlan::planGlobalMap[networkID].begin(); 
            iter != PhysicalPlan::planGlobalMap[networkID].end(); ) {
        PhysicalPlan* pp = (PhysicalPlan*)(*iter);
        delete pp;

        iter = PhysicalPlan::planGlobalMap[networkID].erase(iter);
    }

    SASSERT0(PhysicalPlan::planGlobalInfoMap.find(networkID) !=
            PhysicalPlan::planGlobalInfoMap.end());

    PlanInfo* deletePlanInfo = PhysicalPlan::planGlobalInfoMap[networkID];
    delete deletePlanInfo;
    PhysicalPlan::planGlobalInfoMap.erase(networkID);
}

PhysicalPlan* PhysicalPlan::getCurPhysicalPlan() {
    return WorkContext::curPhysicalPlan;
}

void PhysicalPlan::setCurPlan(int networkID, int dopID) {
    unique_lock<mutex> planLock(PhysicalPlan::planGlobalMutex);
    SASSERT0(PhysicalPlan::planGlobalMap.find(networkID) != 
            PhysicalPlan::planGlobalMap.end());

    SASSUME0(dopID < PhysicalPlan::planGlobalMap[networkID].size());
    WorkContext::curPhysicalPlan = PhysicalPlan::planGlobalMap[networkID][dopID];

    SASSERT0(PhysicalPlan::planGlobalInfoMap.find(networkID) !=
            PhysicalPlan::planGlobalInfoMap.end());
    WorkContext::curPlanInfo = PhysicalPlan::planGlobalInfoMap[networkID];
}

int PhysicalPlan::getDOPCount(int networkID) {
    unique_lock<mutex> planInfoLock(PhysicalPlan::planGlobalMutex);
    SASSUME0(PhysicalPlan::planGlobalInfoMap.find(networkID) !=
            PhysicalPlan::planGlobalInfoMap.end());
    PlanInfo* planInfo = PhysicalPlan::planGlobalInfoMap[networkID];
    planInfoLock.unlock();
    return planInfo->dopCount;
}
