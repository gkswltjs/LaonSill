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

using namespace std;

map<int, vector<PhysicalPlan*>>     PhysicalPlan::planGlobalMap;
map<int, PlanInfo*>                 PhysicalPlan::planGlobalInfoMap;
mutex                               PhysicalPlan::planGlobalMutex;

PhysicalPlan::~PhysicalPlan() {
    for (map<int, void*>::iterator iter = instanceMap.begin(); iter != instanceMap.end();
        ++iter) {

        int layerID = iter->first;
        void* instancePtr = iter->second;

        SASSUME0(planMap.find(layerID) != planMap.end());
        int layerType = planMap[layerID].layerType;

        LayerFunc::destroyLayer(layerType, instancePtr);
    }
}

void* PhysicalPlan::allocTensorMem(int layerType, void* instancePtr, string tensorName,
    PlanAlloc planAlloc, bool isInput, int index) {
    if (planAlloc.nodeID != SPARAM(NODE_ID)) {
        // allocate GPU memory of other nodes.
        // TODO: 구현!!
        SASSERT(false, "not implemented yet");
    }

    int oldGPUIdx = Worker<float>::gpuIdx;
    if (planAlloc.devID != oldGPUIdx) {
        checkCudaErrors(cudaSetDevice(planAlloc.devID)); 
    }

    // XXX: float형 코딩으로 박지 말고, 설정에 따라서 template date type을 설정하도록 수정해야
    //     한다. 
    Data<float>* tensor = new Data<float>(tensorName);
    SASSERT0(tensor != NULL);

    if (planAlloc.devID != oldGPUIdx) {
        checkCudaErrors(cudaSetDevice(oldGPUIdx));
    }

    LayerFunc::setInOutTensor(layerType, instancePtr, (void*)tensor, isInput, index);
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
            }
        }

        // (2) allocate layer tensor(ex. paramHistory)
        if (SLPROP_BASE(receive)) {
            // TODO: 구현
            SASSERT(false, "not implemented yet");
        }

        SASSERT0(LayerFunc::allocLayerTensors(layerType, instancePtr) == true);

        if (SLPROP_BASE(donate)) {
            // TODO: 구현
            SASSERT(false, "not implemented yet");
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

void PhysicalPlan::markFinish(int planID) {
    unique_lock<mutex> planLock(this->planMutex);

    SASSUME(this->depRefMap.find(planID) != this->depRefMap.end(),
        "There is no ref map for requesting plan ID. planID=%d", planID);
    this->depRefMap[planID] -= 1;
    
    if (this->depRefMap[planID] == 0) {
        this->readyQueue.push_back(planID); 
    }

    this->refCount -= 1;
    planLock.unlock();
    SASSUME0(this->depRefMap[planID] >= 0);
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

    planInfoLock.unlock();

    if (WorkContext::curPlanInfo->curEpochIndex >= WorkContext::curPlanInfo->epochCount) {
        return false;
    }

    // (3) 초기화를 수행한다.
    this->refCount = 0;
    this->readyQueue = {};
    for (map<int, PlanDef>::iterator it = planMap.begin(); it != planMap.end(); ++it) {
        int key = it->first;
        PlanDef value = it->second;
      
        if (value.depCount == 0) {
            readyQueue.push_back(key);
        } else {
            this->refCount += 1;
        }
    
        depRefMap[key] = value.depCount;
    }

    return true;
}

void PhysicalPlan::runLayer(int planID) {
    // (1) set context
    int layerID = LP_PLANID_TO_LAYERID(planID);
    WorkContext::updateLayer(WorkContext::curNetworkID, layerID);

    PlanType planType = LP_PLANID_TO_PLANTYPE(planID);
    
    // (2) run layer
    // TODO:
    PlanInfo* planInfo = WorkContext::curPlanInfo;
    cout << "Epoch : " << planInfo->curEpochIndex << ", minibatch : " << 
        planInfo->curMiniBatchIndex << " run layer (planID=" << planID << ")";

    // FIXME: 나름 핫 코드영역인데 이렇게 자주 맵을 뒤지면 성능에 안좋다. 수정필요!!
    SASSUME0(this->planMap.find(planID) != this->planMap.end());
    int layerType = this->planMap[planID].layerType;

    SASSUME0(this->instanceMap.find(layerID) != this->instanceMap.end());
    void* instancePtr = this->instanceMap[layerID];

    SASSUME0(planType < PLANTYPE_MAX);
    if (planType == PLANTYPE_FORWARD) {
        LayerFunc::runForward(layerType, instancePtr, planInfo->curMiniBatchIndex);
    } else if (planType == PLANTYPE_BACKWARD) {
        LayerFunc::runBackward(layerType, instancePtr);
    } else {
        SASSUME0(planType == PLANTYPE_UPDATE);
        LayerFunc::learn(layerType, instancePtr);
    }

    // (3) mark
    PlanDef *planDef = &WorkContext::curPhysicalPlan->planMap[planID];
    for (int i = 0; i < planDef->notifyList.size(); i++) {
        int targetPlanID = planDef->notifyList[i];
        markFinish(targetPlanID);
    }
}

bool PhysicalPlan::runPlan(int layerID, PlanType planType) {
    unique_lock<mutex> planLock(this->planMutex);

    if (find(this->readyQueue.begin(), this->readyQueue.end(), layerID) != 
        this->readyQueue.end())
        return false;

    planLock.unlock();

    if (planType == PLANTYPE_FORWARD) {
        runLayer(LP_FORWARD_PLANID(layerID));
    } else if (planType == PLANTYPE_BACKWARD) {
        runLayer(LP_BACKWARD_PLANID(layerID));
    } else if (planType == PLANTYPE_UPDATE) {
        runLayer(LP_UPDATE_PLANID(layerID));
    } else {
        SASSERT(false, "invalid plan type. plan type=%d", (int)planType);
    }

    return true;
}

bool PhysicalPlan::runPlan() {
    unique_lock<mutex> planLock(this->planMutex);
    
    if (this->readyQueue.size() == 0) {
        return false;
    }

    int planID = this->readyQueue.front();
    this->readyQueue.pop_front();
    planLock.unlock();

    SASSUME0(this->planMap.find(planID) != this->planMap.end());

    runLayer(planID);
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
