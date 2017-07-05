/**
 * @file WorkContext.cpp
 * @date 2017-05-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "WorkContext.h"
#include "PropMgmt.h"

using namespace std;

thread_local int            WorkContext::curDOPID;
thread_local PhysicalPlan*  WorkContext::curPhysicalPlan;
thread_local PlanInfo*      WorkContext::curPlanInfo;
thread_local int            WorkContext::curThreadID;

thread_local LayerProp*     WorkContext::curLayerProp;
thread_local _NetworkProp*  WorkContext::curNetworkProp;
thread_local int            WorkContext::curNetworkID = -1;
BootMode                    WorkContext::curBootMode;

void WorkContext::updateNetwork(int networkID) {
    if (networkID < 0)
        return;

    WorkContext::curNetworkID = networkID;
    WorkContext::curNetworkProp = PropMgmt::getNetworkProp(networkID);
    PhysicalPlan::setCurPlanInfo(networkID);
}

void WorkContext::updateLayer(int networkID, int layerID) {
    if (networkID != WorkContext::curNetworkID) {
        WorkContext::updateNetwork(networkID);
    }

    WorkContext::curLayerProp = PropMgmt::getLayerProp(networkID, layerID);
}

void WorkContext::updatePlan(int dopID, bool acquireLock) {
    WorkContext::curDOPID = dopID;
    int networkID = WorkContext::curNetworkID;

    PhysicalPlan::setCurPlan(networkID, dopID, acquireLock);
}

void WorkContext::printContext(FILE *fp) {
    fprintf(fp,
        "networkID = %d, dopID = %d, PhysicalPlan : %p, LayerProp : %p, NetworkProp : %p\n",
        WorkContext::curNetworkID, WorkContext::curDOPID, WorkContext::curPhysicalPlan,
        WorkContext::curLayerProp, WorkContext::curNetworkProp);
}