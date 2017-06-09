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

AtariNN::AtariNN(int rowCount, int colCount, int channelCount, int actionCount) {
    this->rowCount      = rowCount;
    this->colCount      = colCount;
    this->channelCount  = channelCount;
    this->actionCount   = actionCount;
}

void AtariNN::createDQNImageLearner() {
    // (1) create job
    Job* job = new Job(Job::CreateDQNImageLearner);
    job->addJobElem(Job::IntType, 1, (void*)&this->rowCount);
    job->addJobElem(Job::IntType, 1, (void*)&this->colCount);
    job->addJobElem(Job::IntType, 1, (void*)&this->channelCount);
    job->addJobElem(Job::IntType, 1, (void*)&this->actionCount);

    // (2) push job & get pub-job(job result) ID
    int pubJobID = Worker::pushJob(job);

    // (3) subscribe pub-job(job result)
    SASSERT0(pubJobID != -1);
    Job *pubJob;
    Broker::subscribe(pubJobID, &pubJob, Broker::Blocking);

    // (4) handle job result
    this->dqnImageLearnerID     = pubJob->getIntValue(0);
    this->networkQID            = pubJob->getIntValue(1);
    this->networkQHeadID        = pubJob->getIntValue(2);

    // (5) cleanup pubJob resource
    delete pubJob;

}

void AtariNN::buildDQNNetworks() {
    // (1) create job
    Job* job = new Job(Job::BuildDQNNetworks);
    job->addJobElem(Job::IntType, 1, (void*)&this->dqnImageLearnerID);
    job->addJobElem(Job::IntType, 1, (void*)&this->networkQID);
    job->addJobElem(Job::IntType, 1, (void*)&this->networkQHeadID);

    // (2) push job
    Worker::pushJob(job);
}

int AtariNN::stepNetwork(float lastReward, int lastAction, int lastTerm, float* state) {
    // (1) push StepDQNImageLearner Job
    Job* job = new Job(Job::StepDQNImageLearner);
    job->addJobElem(Job::IntType, 1, (void*)&this->dqnImageLearnerID);
    job->addJobElem(Job::IntType, 1, (void*)&this->networkQID);
    job->addJobElem(Job::IntType, 1, (void*)&this->networkQHeadID);
    job->addJobElem(Job::FloatType, 1, (void*)&lastReward);
    job->addJobElem(Job::IntType, 1, (void*)&lastAction);
    job->addJobElem(Job::IntType, 1, (void*)&lastTerm);
    int stateCount = this->rowCount * this->colCount * this->channelCount;
    job->addJobElem(Job::FloatArrayType, stateCount, (void*)state);

    // (2) push job & get pub-job(job result) ID
    int pubJobID = Worker::pushJob(job);

    // (3) subscribe pub-job(job result)
    SASSERT0(pubJobID != -1);
    Job *pubJob;
    Broker::subscribe(pubJobID, &pubJob, Broker::Blocking);

    // (4) handle job result
    int action = pubJob->getIntValue(0);

    // (5) cleanup resource
    delete pubJob;

    return action;
}

void AtariNN::cleanupDQNImageLearner() {
    // (1) cleanup networks
    Job *delQNetJob = new Job(Job::CleanupNetwork);
    delQNetJob->addJobElem(Job::IntType, 1, (void*)&this->networkQID);
    Worker::pushJob(delQNetJob);

    Job *delQHeadNetJob = new Job(Job::CleanupNetwork);
    delQNetJob->addJobElem(Job::IntType, 1, (void*)&this->networkQHeadID);
    Worker::pushJob(delQHeadNetJob);

    // (2) cleanup learner
    Job *delJob = new Job(Job::CleanupDQNImageLearner);
    delJob->addJobElem(Job::IntType, 1, (void*)&this->dqnImageLearnerID);
    Worker::pushJob(delJob);
}
