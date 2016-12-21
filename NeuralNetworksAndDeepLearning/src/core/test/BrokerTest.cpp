/**
 * @file BrokerTest.cpp
 * @date 2016-12-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "BrokerTest.h"
#include "common.h"
#include "Broker.h"
#include "Job.h"
#include "StdOutLog.h"

bool BrokerTest::runSimplePubSubTest() {
    // (1) create job
    Job *testJob = new Job(Job::TestJob);
    int A = 10;
    float B = 3.77;
    float C[100];
    for (int i = 0; i < 100; i++)
        C[i] = (float)i;
    int D = 15;

    testJob->addJobElem(Job::IntType, 1, (void*)&A);
    testJob->addJobElem(Job::FloatType, 1, (void*)&B);
    testJob->addJobElem(Job::FloatArrayType, 100, (void*)&C);
    testJob->addJobElem(Job::IntType, 1, (void*)&D);

    // (2) publish & subscribe
    Broker::BrokerRetType retType;
    retType = Broker::publish(testJob->getJobID(), 0, testJob);
    if (retType != Broker::Success) {
        STDOUT_LOG("publish failed. ret type=%d", (int)retType);
        return false;
    }

    // (3) subscribe
    Job *subscribedJob = NULL;
    retType = Broker::subscribe(testJob->getJobID(), 0, &subscribedJob, Broker::Blocking);
    if (retType != Broker::Success) {
        STDOUT_LOG("subscribe failed. ret type=%d", (int)retType);
        goto fail_ret;
    }

    if (subscribedJob->getType() != Job::TestJob) {
        STDOUT_LOG("invalid type of subscribed job. job type=%d",
            (int)subscribedJob->getType());
        goto fail_ret;
    }

    if ((subscribedJob->getIntValue(0) != A) ||
        (subscribedJob->getFloatValue(1) != B) ||
        (subscribedJob->getIntValue(3) != D)) {
        STDOUT_LOG("invalid value of subscribed job. A=%d, B=%f, D=%d",
            subscribedJob->getIntValue(0), subscribedJob->getFloatValue(1),
            subscribedJob->getIntValue(3));
        goto fail_ret;
    }

    for (int i = 0; i < 100; i++) {
        if (subscribedJob->getFloatArrayValue(2, i) != C[i]) {
            STDOUT_LOG("invalid value of subscribed job. i=%d, C[i]=%f but %f",
                i, C[i], subscribedJob->getFloatArrayValue(2, i));
            goto fail_ret;
        }
    }

    delete subscribedJob;

    return true;

fail_ret:
    if (subscribedJob != NULL)
        delete subscribedJob;

    return true;
}

bool BrokerTest::runTest() {
    bool result = runSimplePubSubTest();
    if (result) {
        STDOUT_LOG(" - simple pub-sub test is success");
    } else {
        STDOUT_LOG(" - simple pub-sub test is failed");
        return false;
    }

    return true;
}
