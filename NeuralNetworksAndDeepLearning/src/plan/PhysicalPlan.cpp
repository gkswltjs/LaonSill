/**
 * @file PhysicalPlan.cpp
 * @date 2017-05-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PhysicalPlan.h"

using namespace std;

atomic<uint64_t> PhysicalPlan::ppIDGen;
map<uint64_t, PhysicalPlan*> PhysicalPlan::ppMap;
vector<PhysicalPlan*> PhysicalPlan::ppList;
mutex PhysicalPlan::ppMutex;

vector<uint64_t> PhysicalPlan::popReadyPPIDList() {
    vector<uint64_t> result;
    return result;
}

void PhysicalPlan::markFinish(uint64_t ppID) {

}

void PhysicalPlan::genPPList(int networkID, int epochIdx, int miniBatchIdx) {

}
