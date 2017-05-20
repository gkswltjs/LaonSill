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

using namespace std;

map<int, vector<PhysicalPlan*>>     PhysicalPlan::planGlobalMap;
map<int, PlanInfo*>                  PhysicalPlan::planGlobalInfoMap;
mutex                               PhysicalPlan::planGlobalMutex;
thread_local int                    PhysicalPlan::curDOPID;
thread_local PhysicalPlan*          PhysicalPlan::curPhysicalPlan; 
thread_local PlanInfo*              PhysicalPlan::curPlanInfo;

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
    unique_lock<mutex> planInfoLock(PhysicalPlan::curPlanInfo->planMutex);
    this->epochIdx = PhysicalPlan::curPlanInfo->curEpochIndex;
    this->miniBatchIdx = PhysicalPlan::curPlanInfo->curMiniBatchIndex;

    if (PhysicalPlan::curPlanInfo->curMiniBatchIndex == 
            PhysicalPlan::curPlanInfo->miniBatchCount - 1) {
        PhysicalPlan::curPlanInfo->curMiniBatchIndex = 0;
        PhysicalPlan::curPlanInfo->curEpochIndex += 1;
    } else {
        PhysicalPlan::curPlanInfo->curMiniBatchIndex += 1;
    }

    planInfoLock.unlock();

    if (PhysicalPlan::curPlanInfo->curEpochIndex >= PhysicalPlan::curPlanInfo->epochCount) {
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
    PropMgmt::updateContext(PropMgmt::curNetworkID, layerID);
    
    // (2) run layer
    // TODO:
    PlanInfo* planInfo = PhysicalPlan::curPlanInfo;
    cout << "Epoch : " << planInfo->curEpochIndex << ", minibatch : " << 
        planInfo->curMiniBatchIndex << " run layer (planID=" << planID << ")";

    // (3) mark
    PlanDef *planDef = &PhysicalPlan::curPhysicalPlan->planMap[planID];
    for (int i = 0; i < planDef->notifyList.size(); i++) {
        int targetPlanID = planDef->notifyList[i];
        markFinish(targetPlanID);
    }

    if (planDef->layerType == Layer<float>::Conv) {
        cout << " [";
        for (int i = 0; i < SLPROP(Conv, input).size(); i++) {
            cout << SLPROP(Conv, input)[i] << " ";
        }
        cout << "] => [";
        for (int i = 0; i < SLPROP(Conv, output).size(); i++) {
            cout << SLPROP(Conv, output)[i] << " ";
        }
        cout << "]" << endl;
    } else if (planDef->layerType == Layer<float>::FullyConnected) {
        cout << "[";
        for (int i = 0; i < SLPROP(FullyConnected, input).size(); i++) {
            cout << SLPROP(FullyConnected, input)[i] << " ";
        }
        cout << "] => [";
        for (int i = 0; i < SLPROP(FullyConnected, output).size(); i++) {
            cout << SLPROP(FullyConnected, output)[i] << " ";
        }
        cout << "]" << endl;
    } else {
        cout << endl;
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

void PhysicalPlan::setCurPlan(int dopID) {
    PhysicalPlan::curDOPID = dopID;
    int networkID = PropMgmt::curNetworkID;

    unique_lock<mutex> planLock(PhysicalPlan::planGlobalMutex);
    SASSERT0(PhysicalPlan::planGlobalMap.find(networkID) != 
            PhysicalPlan::planGlobalMap.end());

    SASSUME0(dopID < PhysicalPlan::planGlobalMap[networkID].size());
    PhysicalPlan::curPhysicalPlan = PhysicalPlan::planGlobalMap[networkID][dopID];

    SASSERT0(PhysicalPlan::planGlobalInfoMap.find(networkID) !=
            PhysicalPlan::planGlobalInfoMap.end());
    PhysicalPlan::curPlanInfo = PhysicalPlan::planGlobalInfoMap[networkID];
}

PhysicalPlan* PhysicalPlan::getCurPhysicalPlan() {
    return PhysicalPlan::curPhysicalPlan;
}
