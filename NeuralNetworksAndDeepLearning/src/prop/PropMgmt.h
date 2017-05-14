/**
 * @file PropMgmt.h
 * @date 2017-04-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PROPMGMT_H
#define PROPMGMT_H 

#include "LayerProp.h"
#include "LayerPropList.h"
#include "NetworkProp.h"

typedef unsigned long   LayerPropKey;

#define MAKE_LAYER_PROP_KEY(networkID, layerID)                                              \
    (LayerPropKey)((unsigned long)networkID << 16 | (unsigned long)layerID)

#define SLPROP(layer, var)                                                                   \
    (((_##layer##PropLayer*)(PropMgmt::curLayerProp->prop))->_##var##_)

#define SNPROP(var)                                                                          \
    (PropMgmt::curNetworkProp->_##var##_)


class PropMgmt {
public: 
    PropMgmt() {}
    virtual ~PropMgmt() {}

    static void update(int networkID, int layerID);
    static void insertLayerProp(LayerProp* layerProp);
    static void removeLayerProp(int networkID);
    static void insertNetworkProp(int networkID, _NetworkProp* networkProp);
    static void removeNetworkProp(int networkID);

    static thread_local volatile LayerProp* curLayerProp;
    static thread_local volatile int curNetworkID;
    static thread_local volatile _NetworkProp* curNetworkProp;
private:
    // FIXME: 맵으로 접근하면 아무래도 느릴 수 밖에 없다. 
    //        쓰레드에서 처리할 job이 변경될때 마다 1번씩 접근하기 때문에 비용이 아주 크지는
    //        않다. 더 좋은 방법이 없을지 고민해보자.
    static std::map<LayerPropKey, LayerProp*> layerPropMap;
    static std::map<int, std::vector<int>> net2LayerIDMap;
    static std::map<int, _NetworkProp*> networkPropMap;
    static LayerProp* getLayerProp(int networkID, int layerID);
    static _NetworkProp* getNetworkProp(int networkID);
};

#endif /* PROPMGMT_H */
