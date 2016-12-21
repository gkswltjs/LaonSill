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
    // XXX: 이것도 Worker에서 수행하도록 변경하자.
    this->networkId = Worker<float>::createNetwork();
}

void AtariNN::buildDQNLayer() {
    Job* newJob = new Job(Job::BuildDQNNetwork);
    Worker<float>::pushJob(newJob);
}

void AtariNN::feedForward(int batchSize) {
    Job* newJob = new Job(Job::FeedForwardDQNNetwork);
    Worker<float>::pushJob(newJob);
}

void AtariNN::pushData(float lastReward, int lastAction, int lastTerm, float* state) {
    Job* newJob = new Job(Job::PushDQNInput);
    newJob->addJobElem(Job::IntType, 1, (void*)&this->networkId);
    newJob->addJobElem(Job::FloatType, 1, (void*)&lastReward);
    newJob->addJobElem(Job::IntType, 1, (void*)&lastAction);
    newJob->addJobElem(Job::IntType, 1, (void*)&lastTerm);
    newJob->addJobElem(Job::FloatArrayType, 4 * 84 * 84, (void*)state);

    Worker<float>::pushJob(newJob);
}
