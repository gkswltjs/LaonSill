/**
 * @file Updater.cpp
 * @date 2017-06-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Updater.h"
#include "SysLog.h"
#include "Worker.h"
#include "PhysicalPlan.h"
#include "WorkContext.h"

using namespace std;

map<UpdaterKey, UpdaterValue*>    Updater::updaterMap; 
mutex                           Updater::updaterMutex;

void Updater::addUpdater(int networkID, int layerID, int paramType, int nodeID, int devID) {
   
    UpdaterKey key;
    key.networkID = networkID;
    key.layerID = layerID;
    key.paramType = paramType;

    UpdaterValue *value = new UpdaterValue();
    SASSERT0(value != NULL);
    value->nodeID = nodeID;
    value->devID = devID;
    value->tensorDataPtr = new Data<float>(string("update_Data_") + to_string(networkID) + 
                                       string("_") + to_string(layerID) +
                                       string("_") + to_string(paramType) +
                                       string("_") + to_string(nodeID) +
                                       string("_") + to_string(devID));
    value->tensorDataHis1Ptr = 
                        new Data<float>(string("update_DataHis1_") + to_string(networkID) + 
                                       string("_") + to_string(layerID) +
                                       string("_") + to_string(paramType) +
                                       string("_") + to_string(nodeID) +
                                       string("_") + to_string(devID));
    value->tensorDataHis2Ptr = 
                        new Data<float>(string("update_DataHis2_") + to_string(networkID) + 
                                       string("_") + to_string(layerID) +
                                       string("_") + to_string(paramType) +
                                       string("_") + to_string(nodeID) +
                                       string("_") + to_string(devID));
    value->reshape = false;
    value->access = true;

    unique_lock<mutex> lock(updaterMutex);
    SASSERT0(updaterMap.find(key) == updaterMap.end()); 
    updaterMap[key] = value;
}

void Updater::unsetReshape(int networkID, int layerID, int paramType) {
    UpdaterKey key;
    key.networkID = networkID;
    key.layerID = layerID;
    key.paramType = paramType;

    unique_lock<mutex> lock(updaterMutex);
    SASSUME0(updaterMap.find(key) != updaterMap.end());
    UpdaterValue* value = updaterMap[key];
    unique_lock<mutex> accessLock(value->mutex);
    value->reshape = false;
    accessLock.unlock();
}

// @return  false : cannot access tensor (locked)
//          true : can access tensor but it could not be done
//                 (for cluster or multi-device scenario)
bool Updater::updateParam(int networkID, int layerID, int paramType, int planID,
    int dopID, UpdateContext context, void* tensorParamPtr, void* tensorParamHis1Ptr, 
    void* tensorParamHis2Ptr, bool needSyncGrad) {

    Data<float>* tensorSourceParam = (Data<float>*)tensorParamPtr;
    Data<float>* tensorTargetParam;
    Data<float>* tensorDataHis1;
    Data<float>* tensorDataHis2;

    int dopCount;
    if (!needSyncGrad) {
        // needSyncGrad가 false인 경우는 UpdaterTaskDefs를 Task Consumer가 실행하는
        // 경우이다. updateTaskDefs는 curPlanInfo가 아닐 수 있기 때문에 
        // 직접 planGlobalInfoMap으로 부터 dopCount를 가져와야 한다.
        dopCount = PhysicalPlan::getDOPCount(networkID);
    } else {
        dopCount = WorkContext::curPlanInfo->dopCount;
    }

    if (dopCount == 1) {
        tensorDataHis1 = (Data<float>*)tensorParamHis1Ptr;
        tensorDataHis2 = (Data<float>*)tensorParamHis2Ptr;

        Update<float>::updateParam(context, tensorDataHis1, tensorDataHis2,
            tensorTargetParam);
        PhysicalPlan::markFinish(networkID, dopID, planID, 1);
        return true;
    }


    UpdaterKey key;
    key.networkID = networkID;
    key.layerID = layerID;
    key.paramType = paramType;

    unique_lock<mutex> updaterMapLock(updaterMutex);
    SASSUME0(updaterMap.find(key) != updaterMap.end());
    UpdaterValue *value = updaterMap[key];
    updaterMapLock.unlock();

    tensorTargetParam = (Data<float>*)value->tensorDataPtr;
    tensorDataHis1 = (Data<float>*)value->tensorDataHis1Ptr;
    tensorDataHis2 = (Data<float>*)value->tensorDataHis2Ptr;

    int bufSize = tensorTargetParam->getCount() * sizeof(float);

    // (1) 상황에 맞게 param의 gradient를 동기화한다.
    if (needSyncGrad) {
        if (value->nodeID != SPARAM(NODE_ID)) {
            // cluster mode
            SASSERT(false, "Not implemented yet");
        }

        bool needReshape;
        unique_lock<mutex> accessLock(value->mutex);
        if (!value->access) {
            return false;
        }
        value->access = false;
        needReshape = value->reshape;
        value->reshape = false;
        accessLock.unlock();
     
        if (value->reshape) {
            tensorTargetParam->reshape(tensorSourceParam->getShape());
            tensorDataHis1->reshape(tensorSourceParam->getShape());
            tensorDataHis2->reshape(tensorSourceParam->getShape());
        }

        // XXX: should I use async copy to increase performance..?
        //      if I use async copy there can be timing issue.
        //        example>
        //          TaskConsumer A : memcpyAsync tensor
        //          TaskConsumer B : update tensor
        //          A should be done before B!! but I'm not sure
        checkCudaErrors(cudaMemcpy((void*)tensorSourceParam->device_grad(),
                                   tensorTargetParam->mutable_device_grad(), bufSize, 
                                   cudaMemcpyDeviceToDevice));

        if (value->devID != Worker::gpuIdx) {
            // makes an updater task and inserts it into updater task queue
            int consumerIdx;    // TODO: convert gpuIdx to consumerIdx
            Worker::addUpdaterTask(consumerIdx, networkID, dopID, layerID, paramType, planID,
                context, tensorParamPtr);
            return false;
        }
    }

    // (2) update param을 수행
    Update<float>::updateParam(context, tensorDataHis1, tensorDataHis2, tensorTargetParam);

    // (3) 상황에 맞게 param의 data를 동기화한다.
    if (value->nodeID != SPARAM(NODE_ID)) {
        // cluster mode
        SASSERT(false, "Not implemented yet");
    }

    checkCudaErrors(cudaMemcpy(tensorSourceParam->mutable_device_data(),
                               (void*)tensorTargetParam->device_data(), bufSize, 
                               cudaMemcpyDeviceToDevice));

    unique_lock<mutex> accessLock(value->mutex);
    SASSUME0(value->access == false);
    value->access = true;
    accessLock.unlock();

    PhysicalPlan::markFinish(networkID, dopID, planID, 1);

    return true;
}

// 각 레이어의 업데이트 함수는 이 함수를 호출해야 한다.
bool Updater::updateParam(int paramType, UpdateContext context, void* tensorParamPtr,
    void* tensorParamHis1Ptr, void* tensorParamHis2Ptr) {

    int networkID = WorkContext::curNetworkID;
    int layerID = WorkContext::curLayerProp->layerID;
    int planID = LP_UPDATE_PLANID(layerID);
    int dopID = WorkContext::curDOPID;

    return Updater::updateParam(networkID, layerID, paramType, planID, dopID, context,
        tensorParamPtr, tensorParamHis1Ptr, tensorParamHis2Ptr, true);
}
