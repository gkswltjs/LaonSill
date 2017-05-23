/**
 * @file PhysicalPlan.h
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string.h>

#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <list>
#include <string>

#include "common.h"
#include "LogicalPlan.h"

#ifndef PHYSICALPLAN_H
#define PHYSICALPLAN_H 

typedef struct PlanInfo_t {
    int         networkID;
    int         dopCount;
    int         epochCount;
    int         miniBatchCount;     // per epoch

    int         curEpochIndex;
    int         curMiniBatchIndex;
    std::mutex  planMutex;      // mutex for curEpochIndex, curMiniBatchIndex
} PlanInfo;

typedef struct TensorAllocKey_t {
    std::string     tensorName;
    PlanAlloc       tensorAlloc;

    bool operator < (const struct TensorAllocKey_t &x) const {
        if (strcmp(tensorName.c_str(), x.tensorName.c_str()) == 0) {
            if (tensorAlloc.nodeID == x.tensorAlloc.nodeID) {
                return tensorAlloc.devID < x.tensorAlloc.devID;
            } else {
                return tensorAlloc.nodeID < x.tensorAlloc.nodeID;
            }
        } else {
            return tensorName < x.tensorName;
        }
    }
} TensorAllocKey;

class PhysicalPlan {
public: 
    PhysicalPlan() {}
    virtual ~PhysicalPlan();

    int                         networkID;
    std::map<int, PlanAlloc>    allocMap;       // 특정 레이어를 어느곳에 배치할 것인가
                                                // key : layer ID, value : PlanAlloc
    std::map<int, PlanDef>      planMap;        // plan ID별 PlanDef 맵
                                                // key : planID, value : PlanDef
    std::map<int, int>          depRefMap;      // 각 plan들의 dependency를 관리한다.
                                                // key : planID, value : remain dependecy
                                                // count
    std::map<int, void*>        instanceMap;    // key : layer ID, value : instance pointer

    int                         dopID;          // degree of parallelism ID
    int                         refCount;   // 이 값이 0이 되면 해당 mini batch에 대한 plan은
                                            // 다 수행한 것으로 판단하면 된다.

    int epochIdx;
    int miniBatchIdx;       // current mini-batch (per epoch) index

    bool generatePlan();    // 현 minibatch에 해당하는 작업이 완료되면 그다음 mini batch에
                            // 대한 플랜을 생성한다.
                            // 만약 모든 batch를 다 돌았을 경우에는 false를 반환한다.

    bool runPlan(int layerID, PlanType planType);
    bool runPlan();

    std::list<int>      readyQueue; // dependency 문제가 해결이 되어 실행이 될 수 있는
                                    // planID들의 리스트
    std::mutex          planMutex;  // refCount, readyQueue, depRefMap을 보호한다.
                                    // XXX: 락을 효율적으로 사용하도록 추후에 수정하자.

    static void insertPlan(int networkID, std::vector<PhysicalPlan*> pMap,
        PlanInfo *pInfoMap);
    static void removePlan(int networkID);

    static PhysicalPlan* getCurPhysicalPlan();

    static void allocateTensor(int networkID);

    static void setCurPlan(int networkID, int dopID);

private:
    static std::map<int, std::vector<PhysicalPlan*>>    planGlobalMap;    // key = networkID,
                                                                // value = Physical Plans
    static std::map<int, PlanInfo*>  planGlobalInfoMap; // 하나의 네트워크에 대한 plan 정보를 
                                                // 담고 있는 구조체. 
                                                // key = networkID, value = plan info 
    static std::mutex               planGlobalMutex;    // planMap, planInfoMap을 보호

    void runLayer(int planID);
    void markFinish(int planID);    // 해당 planID에게 dependency가 있는 planID가 완료가
                                    // 되었음을 알린다.

    void allocateTensorInternal(int networkID);
    static void* allocTensorMem(int layerType, void* instancePtr, std::string tensorName,
                                PlanAlloc planAlloc, bool isInput, int index);
};
#endif /* PHYSICALPLAN_H */
