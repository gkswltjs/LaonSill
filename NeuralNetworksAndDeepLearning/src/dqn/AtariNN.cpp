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
#include "Broker.h"

using namespace std;

AtariNN::AtariNN(int rowCount, int colCount, int channelCount) {
    this->rowCount = rowCount;
    this->colCount = colCount;
    this->channelCount = channelCount;
}

void AtariNN::createDQNImageLearner() {
    // (1) create job
    Job* job = new Job(Job::CreateDQNImageLearner);
    job->addJobElem(Job::IntType, 1, (void*)&this->rowCount);
    job->addJobElem(Job::IntType, 1, (void*)&this->colCount);
    job->addJobElem(Job::IntType, 1, (void*)&this->channelCount);

    // (2) push job & get pub-job(job result) ID
    int pubJobID = Worker<float>::pushJob(job);

    // (3) subscribe pub-job(job result)
    SASSERT0(pubJobID != -1);
    Job *pubJob;
    Broker::subscribe(pubJobID, &pubJob, Broker::Blocking);

    // (4) handle job result
    this->dqnImageLearnerID     = pubJob->getIntValue(0);
    this->networkQID            = pubJob->getIntValue(1);
    this->networkQHeadID        = pubJob->getIntValue(2);

    // (5) cleanup resource
    delete pubJob;
}

void AtariNN::buildDQNNetworks() {
    // (1) create job
    Job* job = new Job(Job::BuildDQNNetworks);
    job->addJobElem(Job::IntType, 1, (void*)&this->networkQID);
    job->addJobElem(Job::IntType, 1, (void*)&this->networkQHeadID);

    // (2) push job
    Worker<float>::pushJob(job);
}

void AtariNN::feedForward(int batchSize) {
    Job* newJob = new Job(Job::FeedForwardDQNNetwork);
    Worker<float>::pushJob(newJob);
}

void AtariNN::pushData(float lastReward, int lastAction, int lastTerm, float* state) {
    Job* newJob = new Job(Job::PushDQNImageInput);
    newJob->addJobElem(Job::IntType, 1, (void*)&this->dqnImageLearnerID);
    newJob->addJobElem(Job::FloatType, 1, (void*)&lastReward);
    newJob->addJobElem(Job::IntType, 1, (void*)&lastAction);
    newJob->addJobElem(Job::IntType, 1, (void*)&lastTerm);
    newJob->addJobElem(Job::FloatArrayType, 4 * 84 * 84, (void*)state);

    Worker<float>::pushJob(newJob);
}
