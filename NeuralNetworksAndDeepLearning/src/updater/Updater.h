/**
 * @file Updater.h
 * @date 2017-06-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef UPDATER_H
#define UPDATER_H 

#include <map>
#include <mutex>

#include "Update.h"

typedef struct UpdaterKey_s {
    int networkID;
    int layerID;
    int paramType;

    bool operator < (const struct UpdaterKey_s &x) const {
        if (networkID == x.networkID) {
            if (layerID == x.layerID) {
                return paramType < x.paramType;
            } else {
                return layerID < x.layerID;
            }
        } else {
            return networkID < x.networkID;
        }
    }
} UpdaterKey;

typedef struct UpdaterValue_s {
    int         nodeID;
    int         devID;
    void*       tensorDataPtr;
    void*       tensorDataHis1Ptr;
    void*       tensorDataHis2Ptr;
    bool        reshape;
    bool        access;
    std::mutex  mutex;
} UpdaterValue;

class Updater {
public: 
    Updater() {}
    virtual ~Updater() {}

    static void addUpdater(int networkID, int layerID, int paramType, int nodeID, int devID);

    static void unsetReshape(int networkID, int layerId, int paramType);

    static bool updateParam(int networkID, int layerID, int paramType, int planID,
                            int dopID, UpdateContext context, void* tensorParamPtr,
                            void *tensorParamHis1Ptr, void *tensorParamHis2Ptr,
                            bool needSyncGrad);

    static bool updateParam(int paramType, UpdateContext context, void *tensorParamPtr,
                            void *tensorParamHis1Ptr, void *tensorParamHis2Ptr);

private:
    static std::map<UpdaterKey, UpdaterValue*>  updaterMap;
    static std::mutex                           updaterMutex;
};

#endif /* UPDATER_H */
