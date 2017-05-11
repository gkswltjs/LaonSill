/**
 * @file LogicalPlan.h
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef LOGICALPLAN_H
#define LOGICALPLAN_H 

#include <vector>
#include <map>
#include <atomic>

typedef struct PlanAlloc_s {
    int nodeID;
    int devID;
} PlanAlloc;

typedef struct PlanDef_s {
    int id;
    int layerType;

    std::vector<int> depList;
    int depCount;
    std::vector<int> notifyList;
    std::vector<int> gcList;
    int gcCount;

    std::vector<PlanAlloc> allocList;   // 어떤 노드의 어떤 dev(GPU) ID에서 동작할지를 정의
} PlanDef;

class LogicalPlan {
public: 
    LogicalPlan() {}
    virtual ~LogicalPlan() {}

    int networkID;
    static void cleanupLogicalPlan(int lpID);

private:
    std::vector<PlanDef>                ppDef;  // physical plan Definition
    static std::map<int, LogicalPlan*>  lpMap;  // logical plan map
                                                // key : network ID, value : plan def list
};

#endif /* LOGICALPLAN_H */
