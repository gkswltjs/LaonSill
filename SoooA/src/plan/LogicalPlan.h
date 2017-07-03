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
#include <mutex>

#define LP_FORWARD_PLANID(id)               (id * 3 + 0)
#define LP_BACKWARD_PLANID(id)              (id * 3 + 1)
#define LP_UPDATE_PLANID(id)                (id * 3 + 2)

#define LP_PLANID_TO_LAYERID(planid)        ((int)((int)planid / 3))

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

#define LP_PLANID_TO_PLANTYPE(planid)       ((PlanType)((int)planid % 3))

typedef struct PlanBuildDef_s {
    int layerID;
    int layerType;

    std::vector<std::string> inputs;    // PlanDef의 빌드를 위해 임시로 사용.
    std::vector<std::string> outputs;   // PlanDef의 빌드를 위해 임시로 사용.
    std::vector<bool> propDowns;       // PlanDef의 빌드를 위해 임시로 사용.

    bool learnable;

    // 아래 3개의 변수는 안쓰는거 같은데.. 좀 생각해보고 지우자..
    bool isDonator;
    bool isReceiver;
    int donatorID;
} PlanBuildDef;

typedef struct PlanDef_s {
    int planID;
    PlanType planType;

    int layerID;
    int layerType;

    int depCount;
    std::vector<int> notifyList;
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
    static LogicalPlan* getLogicalPlan(int networkID);

    static bool isInnerLayer(int networkID, int layerID);
    static void setLayerType(int networkID, int layerID, bool isInner);

private:
    static std::map<int, LogicalPlan*>  lpMap;  // logical plan map
                                                // key : network ID, value : plan def list
    static std::mutex                   lpMapMutex;
    static PlanDef* findPlanDef(LogicalPlan* lp, int planID);

    std::map<int, bool>                 layerTypeMap;   // key : layerID, 
                                                        // value : true(innerLayer)
    std::mutex                          layerTypeMutex;
};

#endif /* LOGICALPLAN_H */
