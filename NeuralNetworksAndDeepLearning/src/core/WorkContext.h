/**
 * @file WorkContext.h
 * @date 2017-05-22
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef WORKCONTEXT_H
#define WORKCONTEXT_H 

#include <atomic>

#include "PhysicalPlan.h"
#include "LayerProp.h"
#include "NetworkProp.h"

class WorkContext {
public: 
    WorkContext() {}
    virtual ~WorkContext() {}

    // FIXME: volatile 필요한지 고민해보자..
    static thread_local int                     curDOPID;
    static thread_local PhysicalPlan*           curPhysicalPlan;
    static thread_local PlanInfo*               curPlanInfo;

    static thread_local LayerProp*              curLayerProp;
    static thread_local _NetworkProp*           curNetworkProp;
    static thread_local int                     curNetworkID;

    static void updateNetwork(int networkID);
    static void updateLayer(int networkID, int layerID);
    static void updatePlan(int dopID);
};
#endif /* WORKCONTEXT_H */
