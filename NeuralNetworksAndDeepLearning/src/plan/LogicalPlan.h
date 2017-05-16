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

typedef enum PlanType_e {
    PLANTYPE_FORWARD = 0,
    PLANTYPE_BACKWARD,
    PLANTYPE_UPDATE,
    PLANTYPE_MAX
} PlanType;

typedef struct PlanBuildDef_s {
    int layerID;
    int layerType;

    std::vector<std::string> inputs;    // PlanDef의 빌드를 위해 임시로 사용.
    std::vector<std::string> outputs;   // PlanDef의 빌드를 위해 임시로 사용.
    std::vector<bool> propDowns;       // PlanDef의 빌드를 위해 임시로 사용.

    bool isDonator;
    bool isReceiver;
    int donatorID;

    bool learnable;
} PlanBuildDef;

typedef struct PlanDef_s {
    int planID;
    PlanType planType;

    int layerID;
    int layerType;

    int depCount;
    std::vector<int> notifyList;

    std::vector<PlanAlloc> allocList;   // 어떤 노드의 어떤 dev(GPU) ID에서 동작할지를 정의
} PlanDef;

class LogicalPlan {
public: 
    LogicalPlan(int networkID) {
        this->networkID = networkID; 
    }
    virtual ~LogicalPlan() {}

    int networkID;
    static void cleanup(int networkID);
    static void build(int networkID, std::map<int, PlanBuildDef> planDefMap);
    std::vector<PlanDef>                ppDefs;  // physical plan Definition
    static void printPlanDef(int networkID);

private:
    static std::map<int, LogicalPlan*>  lpMap;  // logical plan map
                                                // key : network ID, value : plan def list
};

#endif /* LOGICALPLAN_H */
