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

thread_local LayerProp*     WorkContext::curLayerProp;
thread_local _NetworkProp*  WorkContext::curNetworkProp;
thread_local int            WorkContext::curNetworkID;

void WorkContext::updateNetwork(int networkID) {
    WorkContext::curNetworkID = networkID;
    WorkContext::curNetworkProp = PropMgmt::getNetworkProp(networkID);
}

void WorkContext::updateLayer(int networkID, int layerID) {
    if (networkID != WorkContext::curNetworkID) {
        WorkContext::updateNetwork(networkID);
    }

    WorkContext::curLayerProp = PropMgmt::getLayerProp(networkID, layerID);
}

void WorkContext::updatePlan(int dopID) {
    WorkContext::curDOPID = dopID;
    int networkID = WorkContext::curNetworkID;

    PhysicalPlan::setCurPlan(networkID, dopID);
}
