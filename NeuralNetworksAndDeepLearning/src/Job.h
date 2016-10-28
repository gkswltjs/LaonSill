/**
 * @file Job.h
 * @date 2016-10-14
 * @author mhlee
 * @brief 서버가 수행해야 할 작업을 명시한다.
 * @details
 */

#ifndef JOB_H
#define JOB_H 

#include <atomic>

using namespace std;

class Job {
public:
    enum JobType : int {
        BuildLayer = 0,
        TrainNetwork,
        CleanupLayer,
        HaltMachine
    };

    Job (JobType jobType, void* network, int arg1) {
        this->jobType = jobType;
        this->network = network;
        this->arg1 = arg1;
    };
    virtual        ~Job() {};

    JobType         getType() const { return jobType; }
    void*           getNetwork() const { return network; }
    int             getArg1() const { return arg1; }

    atomic<int>     refCnt;

private:
    JobType         jobType;
    void*           network;
    int             arg1;
    
};

#endif /* JOB_H */
