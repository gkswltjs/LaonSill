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
#define PLAN_OPT_SINGLE_GPU         1
#define PLAN_OPT_MULTI_GPU          2 
#define PLAN_OPT_MULTI_NODE         4
#define PLAN_OPT_VERTICAL_SPLIT     8
#define PLAN_OPT_HORIZONTAL_SPLIT   16

#define PLAN_OPT_DEFAULT            (PLAN_OPT_SINGLE_GPU|PLAN_OPT_MULTI_GPU)

typedef struct GenPlanDef_t {
    bool                    use;
    float                   time;       // cost
    std::vector<PlanDef>    planDefList;
} GenPlanDef;

typedef struct GenPlanKey_t {
    bool operator<(const struct GenPlanKey_t& value) const {
        if (lpID == value.lpID) {
            return option < value.option;
        } else {
            return lpID < value.lpID;
        }
    }

    int lpID;
    int option;
} GenPlanKey;

class PlanOptimizer {
public: 
    PlanOptimizer() {}
    virtual ~PlanOptimizer() {}

    static int generatePlans(int lpID, int option);
    static int generatePlans(int lpID);

    static void cleanupGenPlans(int lpID);

private:
    static std::map<GenPlanKey, std::vector<GenPlanDef>> genPlanMap;
};

#endif /* PLANOPTIMIZER_H */
