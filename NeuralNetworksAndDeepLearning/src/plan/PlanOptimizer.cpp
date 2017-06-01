/**
 * @file PlanOptimizer.cpp
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <time.h>

#include <algorithm>

#include "PhysicalPlan.h"
#include "PlanOptimizer.h"
#include "ResourceManager.h"
#include "common.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "WorkContext.h"

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
        setPlanContext(networkID, availableOptions[0], false);
        return true;
    }
   
    if (policy == PLAN_OPT_POLICY_USE_FIRST_AVAILABLE_OPTION) {
        setPlanContext(networkID, availableOptions[0], false);
        return true;
    }

    if (policy == PLAN_OPT_POLICY_USE_LAST_AVAILABLE_OPTION) {
        setPlanContext(networkID, availableOptions[availableOptions.size() - 1], false);
        return true;
    }

    double bestElapsedTime;
    bool isFirst = true;
    int bestOption;

    for (int i = 0; i < availableOptions.size(); i++) {
        setPlanContext(networkID, availableOptions[i], true);
        double curElapsedTime = runPlan(true);

        if (isFirst) {
            bestOption = availableOptions[i];
            isFirst = false;
            bestElapsedTime = curElapsedTime;
        } else if (curElapsedTime < bestElapsedTime) {
            bestElapsedTime = curElapsedTime; 
            bestOption = availableOptions[i];
        }
        unsetPlanContext(networkID);
    }

    return true;
}

bool PlanOptimizer::buildPlans(int networkID) {
    return buildPlans(networkID, PLAN_OPT_DEFAULT, PLAN_OPT_POLICY_DEFAULT);
}

double PlanOptimizer::runPlan(bool inference) {
    struct timespec startTime, endTime;
    clock_gettime(CLOCK_REALTIME, &startTime);
   
    // TODO: run test
    PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();

    bool jobFinish = true;
    while (jobFinish) {
        bool canRunPlan = true;
        while (canRunPlan) {
            canRunPlan = pp->runPlan(inference);
        }
        jobFinish = pp->generatePlan();
    }

    clock_gettime(CLOCK_REALTIME, &endTime);
    double elapsed = (endTime.tv_sec - startTime.tv_sec) +
        + (endTime.tv_nsec - startTime.tv_nsec) / 1000000000.0;

    return elapsed;
}

void PlanOptimizer::setSingleGPUPlanContext(int networkID, bool isTest) {
    // (1) make physical plan list
    vector<PhysicalPlan*> ppList;
    GPUDevInfo devInfo = ResourceManager::getSingleGPUInfo();

    LogicalPlan* lp = LogicalPlan::getLogicalPlan(networkID);
    PhysicalPlan* pp = new PhysicalPlan();

    pp->networkID = networkID;
    pp->refCount = 0;

    for (int i = 0; i < lp->ppDefs.size(); i++) {
        PlanDef planDef = lp->ppDefs[i];
        PlanAlloc planAlloc;
        planAlloc.nodeID = devInfo.nodeID;
        planAlloc.devID = devInfo.devID;
       
        if (pp->allocMap.find(planDef.layerID) == pp->allocMap.end()) {
            pp->allocMap[planDef.layerID] = planAlloc;
        }

        SASSERT0(pp->planMap.find(planDef.planID) == pp->planMap.end());
        pp->planMap[planDef.planID] = planDef;

        SASSERT0(pp->depRefMap.find(planDef.planID) == pp->depRefMap.end());
        pp->depRefMap[planDef.planID] = planDef.depCount;

        if (planDef.depCount == 0) {
            pp->readyQueue.push_back(planDef.planID);
        } else {
            pp->refCount += 1;
        }
    }

    pp->dopID = 0;
    pp->epochIdx = 0;
    pp->miniBatchIdx = 0;

    ppList.push_back(pp);

    // (2) make PlanInfo
    PlanInfo *planInfo = new PlanInfo();
    planInfo->networkID = networkID;
    planInfo->dopCount = 1;

    planInfo->epochCount = SNPROP(epochs);
    planInfo->miniBatchCount = SNPROP(miniBatch);

    if (isTest) {
        planInfo->epochCount =
            min((int)planInfo->epochCount, (int)SPARAM(PLAN_OPT_TEST_MAX_EPOCH_COUNT));
        planInfo->miniBatchCount =
            min((int)planInfo->miniBatchCount, (int)SPARAM(PLAN_OPT_TEST_MAX_MINIBATCH_COUNT));
    }

    planInfo->curEpochIndex = 0;
    planInfo->curMiniBatchIndex = 0;

    // (3) insert plan
    PhysicalPlan::insertPlan(networkID, ppList, planInfo);

    // (4) set context
    WorkContext::updatePlan(0);
}

void PlanOptimizer::setMultiGPUPlanContext(int networkID, bool isTest) { 
    SASSERT0(false);
}

void PlanOptimizer::setMultiNodePlanContext(int networkID, bool isTest) { 
    SASSERT0(false);
}

void PlanOptimizer::setVerticalSplitPlanContext(int networkID, bool isTest) { 
    SASSERT0(false);
}

void PlanOptimizer::setHorizontalSplitPlanContext(int networkID, bool isTest) { 
    SASSERT0(false);
}

void PlanOptimizer::setPlanContext(int networkID, int option, bool isTest) {
    PlanInfo planInfoMap;
    PhysicalPlan* physicalPlan;

    switch (option) {
        case PLAN_OPT_SINGLE_GPU:
            setSingleGPUPlanContext(networkID, isTest);
            break;

        case PLAN_OPT_MULTI_GPU:
            setMultiGPUPlanContext(networkID, isTest);
            break;

        case PLAN_OPT_MULTI_NODE:
            setMultiNodePlanContext(networkID, isTest);
            break;

        case PLAN_OPT_VERTICAL_SPLIT:
            setVerticalSplitPlanContext(networkID, isTest);
            break;

        case PLAN_OPT_HORIZONTAL_SPLIT:
            setHorizontalSplitPlanContext(networkID, isTest);
            break;

        default:
            SASSERT(false, "invalid plan option. option=%d", option);
            break;
    }
    PhysicalPlan::allocateTensor(networkID);
    PhysicalPlan::loadNetwork();
}

void PlanOptimizer::unsetPlanContext(int networkID) {
    PhysicalPlan::removePlan(networkID);
}
