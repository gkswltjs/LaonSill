/**
 * @file PlanOptimizer.h
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PLANOPTIMIZER_H
#define PLANOPTIMIZER_H 

#include <vector>
#include <map>
#include <string>

#include "ResourceManager.h"
#include "LogicalPlan.h"

// bit exclusive
#define PLAN_OPT_NONE               0
#define PLAN_OPT_SINGLE_GPU         1
#define PLAN_OPT_MULTI_GPU          2 
#define PLAN_OPT_MULTI_NODE         4
#define PLAN_OPT_VERTICAL_SPLIT     8
#define PLAN_OPT_HORIZONTAL_SPLIT   16

//#define PLAN_OPT_DEFAULT            (PLAN_OPT_SINGLE_GPU|PLAN_OPT_MULTI_GPU)
#define PLAN_OPT_DEFAULT            (PLAN_OPT_SINGLE_GPU)

typedef enum PlanOptPolicy_e {
    PLAN_OPT_POLICY_USE_FIRST_AVAILABLE_OPTION = 0,
    PLAN_OPT_POLICY_USE_LAST_AVAILABLE_OPTION,
    PLAN_OPT_POLICY_USE_BEST_OPTION,
    PLAN_OPT_POLICY_MAX
} PlanOptPolicy;

#define PLAN_OPT_POLICY_DEFAULT     (PLAN_OPT_POLICY_USE_BEST_OPTION)

class PlanOptimizer {
public: 
    PlanOptimizer() {}
    virtual ~PlanOptimizer() {}

    static bool buildPlans(int networkID, int option, PlanOptPolicy policy);
    static bool buildPlans(int networkID);

    static void removePlans(int networkID);
    static void init();
    static double testPlan();

private:
    static void setPlanContext(int networkID, int option, bool isTest);
    static void setSingleGPUPlanContext(int networkID, bool isTest);
    static void setMultiGPUPlanContext(int networkID, bool isTest);
    static void setMultiNodePlanContext(int networkID, bool isTest);
    static void setVerticalSplitPlanContext(int networkID, bool isTest);
    static void setHorizontalSplitPlanContext(int networkID, bool isTest);
    static void unsetPlanContext(int networkID);
    static std::vector<int> options;
};

#endif /* PLANOPTIMIZER_H */
