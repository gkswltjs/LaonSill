/**
 * @file PlanOptimizer.cpp
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <time.h>

#include "PlanOptimizer.h"
#include "ResourceManager.h"

using namespace std;

vector<int> PlanOptimizer::options;

void PlanOptimizer::init() {
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_SINGLE_GPU))
        options.push_back(PLAN_OPT_SINGLE_GPU);
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_MULTI_GPU))
        options.push_back(PLAN_OPT_MULTI_GPU);
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_MULTI_NODE))
        options.push_back(PLAN_OPT_MULTI_NODE);
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_VERTICAL_SPLIT))
        options.push_back(PLAN_OPT_VERTICAL_SPLIT);
    if (ResourceManager::isVaildPlanOption(PLAN_OPT_HORIZONTAL_SPLIT))
        options.push_back(PLAN_OPT_HORIZONTAL_SPLIT);
}

bool PlanOptimizer::buildPlans(int networkID, int option, PlanOptPolicy policy) {
    vector<int> availableOptions;
    for (int i = 0; i < PlanOptimizer::options.size(); i++) {
        if (option & PlanOptimizer::options[i])
            availableOptions.push_back(PlanOptimizer::options[i]);
    }

    if (availableOptions.size() == 0)
        return false;

    if (availableOptions.size() == 1) {
        setPlan(networkID, availableOptions[0], false);
        return true;
    }
   
    if (policy == PLAN_OPT_POLICY_USE_FIRST_AVAILABLE_OPTION) {
        setPlan(networkID, availableOptions[0], false);
        return true;
    }

    if (policy == PLAN_OPT_POLICY_USE_LAST_AVAILABLE_OPTION) {
        setPlan(networkID, availableOptions[availableOptions.size() - 1], false);
        return true;
    }

    double bestElapsedTime;
    bool isFirst = true;
    int bestOption;

    for (int i = 0; i < availableOptions.size(); i++) {
        setPlan(networkID, availableOptions[i], true);
        double curElapsedTime = testPlan();

        if (isFirst) {
            bestOption = availableOptions[i];
            isFirst = false;
            bestElapsedTime = curElapsedTime;
        } else if (curElapsedTime < bestElapsedTime) {
            bestElapsedTime = curElapsedTime; 
            bestOption = availableOptions[i];
        }
        unsetPlan(networkID);
    }

    return true;
}

bool PlanOptimizer::buildPlans(int networkID) {
    return buildPlans(networkID, PLAN_OPT_DEFAULT, PLAN_OPT_POLICY_DEFAULT);
}

double PlanOptimizer::testPlan() {
    struct timespec startTime, endTime;
    clock_gettime(CLOCK_REALTIME, &startTime);
   
    // TODO: run test


    clock_gettime(CLOCK_REALTIME, &endTime);
    double elapsed = (endTime.tv_sec - startTime.tv_sec) +
        + (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

    return elapsed;
}

void PlanOptimizer::setPlan(int networkID, int option, bool isTest) {
    // TODO:

}

void PlanOptimizer::unsetPlan(int networkID) {
    // TODO:

}
