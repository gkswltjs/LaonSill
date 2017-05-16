/**
 * @file PlanOptimizer.cpp
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PlanOptimizer.h"
#include "ResourceManager.h"

using namespace std;

map<GenPlanKey, vector<GenPlanDef>> PlanOptimizer::genPlanMap;

int PlanOptimizer::generatePlans(int networkID, int option) {
    return 0;
}

int PlanOptimizer::generatePlans(int networkID) {
    return generatePlans(networkID, PLAN_OPT_DEFAULT);
}

void PlanOptimizer::cleanupGenPlans(int networkID) {

}
