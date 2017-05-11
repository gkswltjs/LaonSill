/**
 * @file PropMgmt.cpp
 * @date 2017-04-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PropMgmt.h"
#include "ColdLog.h"
#include "SysLog.h"

using namespace std;

map<unsigned long, LayerProp*> PropMgmt::layerPropMap;
map<int, vector<int>> PropMgmt::net2LayerIDMap;
map<int, _NetworkProp*> PropMgmt::networkPropMap;
thread_local volatile LayerProp* PropMgmt::curLayerProp;
thread_local volatile _NetworkProp* PropMgmt::curNetworkProp;
thread_local volatile int PropMgmt::curNetworkID = -1;

void PropMgmt::update(int networkID, int layerID) {
    PropMgmt::curLayerProp = getLayerProp(networkID, layerID);

    if (networkID != curNetworkID) {
        PropMgmt::curNetworkProp = getNetworkProp(networkID);
    }
}

LayerProp* PropMgmt::getLayerProp(int networkID, int layerID) {
    LayerPropKey key = MAKE_LAYER_PROP_KEY(networkID, layerID);

    map<LayerPropKey, LayerProp*>::iterator iter = layerPropMap.find(key);
    SASSERT0(iter != layerPropMap.end());

    LayerProp* lp = iter->second;

    SASSERT0(lp->networkID == networkID);
    SASSERT0(lp->layerID == layerID);
    
    return lp;
}

_NetworkProp* PropMgmt::getNetworkProp(int networkID) {
    map<int, _NetworkProp*>::iterator iter = networkPropMap.find(networkID);
    SASSERT0(iter != networkPropMap.end());

    _NetworkProp* np = iter->second;

    return np;
}

void PropMgmt::insertLayerProp(LayerProp* layerProp) {
    int networkID = layerProp->networkID;
    int layerID = layerProp->layerID;
    map<int, vector<int>>::iterator layerIDIter = net2LayerIDMap.find(networkID);
    if (layerIDIter == net2LayerIDMap.end())
        net2LayerIDMap[networkID] = {};
    net2LayerIDMap[networkID].push_back(layerID);

    LayerPropKey key = MAKE_LAYER_PROP_KEY(networkID, layerID);
    map<LayerPropKey, LayerProp*>::iterator iter = layerPropMap.find(key);
    SASSERT0(iter == layerPropMap.end());

    layerPropMap[key] = layerProp;
}

void PropMgmt::removeLayerProp(int networkID) {
    // XXX: 메모리 잘 해제되는지 확인해야 한다.
    map<int, vector<int>>::iterator iter = net2LayerIDMap.find(networkID);
    if (iter != net2LayerIDMap.end()) {
        COLD_LOG(ColdLog::WARNING, true,
            "specific networkID is not registered yet. network ID=%d", networkID);
    }

    vector<int> layerIDList = iter->second;
    vector<int>::iterator layerIDIter;
    for (layerIDIter = layerIDList.begin(); layerIDIter != layerIDList.end(); ) {
        int layerID = (*layerIDIter);
        LayerProp* lpp = getLayerProp(networkID, layerID);
        LayerPropKey key = MAKE_LAYER_PROP_KEY(networkID, layerID);
        layerPropMap.erase(key);
        delete lpp;

        layerIDIter = layerIDList.erase(layerIDIter);
    }

    net2LayerIDMap.erase(networkID);
}

void PropMgmt::insertNetworkProp(int networkID, _NetworkProp* networkProp) {
    map<int, _NetworkProp*>::iterator iter = networkPropMap.find(networkID);
    SASSERT0(iter == networkPropMap.end());

    networkPropMap[networkID] = networkProp;
}

void PropMgmt::removeNetworkProp(int networkID) {
    _NetworkProp* networkProp = getNetworkProp(networkID);
    networkPropMap.erase(networkID);
    delete networkProp;
}
