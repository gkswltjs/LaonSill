/**
 * @file PlanOptimizer.cpp
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PlanOptimizer.h"

using namespace std;

map<GenPlanKey, vector<GenPlanDef>> PlanOptimizer::genPlanMap;

int PlanOptimizer::generatePlans(int lpID, int option) {
    return 0;
}

int PlanOptimizer::generatePlans(int lpID) {
    generatePlans(lpID, PLAN_OPT_DEFAULT);
}

void PlanOptimizer::cleanupGenPlans(int lpID) {

}
