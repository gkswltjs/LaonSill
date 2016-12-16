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
    Job* newJob = new Job(Job::BuildDQNNetwork);
    Worker<float>::pushJob(newJob);
}

void AtariNN::feedForward(int batchSize) {
    Job* newJob = new Job(Job::FeedForwardDQNNetwork);
    Worker<float>::pushJob(newJob);
}

void AtariNN::fillInputData(int imgCount, float* img, int action, float reward, bool term) {
    //Job* newJob = new Job(Job::InsertFrameInfoDQNNetwork, this->network, 
}
