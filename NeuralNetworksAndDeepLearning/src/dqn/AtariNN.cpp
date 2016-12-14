/**
 * @file AtariNN.cpp
 * @date 2016-12-14
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "common.h"
#include "AtariNN.h"
#include "Worker.h"
#include "Job.h"
#include "SysLog.h"

using namespace std;

void AtariNN::createNetwork() {
    this->networkId = Worker<float>::createNetwork();
    this->network = Worker<float>::getNetwork(this->networkId); 
    SASSUME0(this->network);
}

void AtariNN::buildDQNLayer() {
    Job* newJob = new Job(Job::BuildDQNLayer, this->network, this->networkId);
    Worker<float>::pushJob(newJob);

}

