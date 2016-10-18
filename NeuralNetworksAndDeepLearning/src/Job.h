/**
 * @file Job.h
 * @date 2016-10-14
 * @author mhlee
 * @brief 서버가 수행해야 할 작업을 명시한다.
 * @details
 */

#ifndef JOB_H
#define JOB_H 

#include "network/Network.h"

template<typename Dtype> class Network;

template <typename Dtype>
class Job {
public:
    enum Type {
        BuildLayer = 0,
        TrainNetwork,
        CleanupLayer,
        HaltMachine
    };

    Job (Type type, Network<Dtype>* network, int arg1) {
        this->type = type;
        this->network = network;
        this->arg1 = arg1;
    };
    virtual ~Job() {};

    Type getType() const { return type; }
    Network<Dtype>* getNetwork() const { return network; }
    int getArg1() const { return arg1; }

private:
    Type type;
    Network<Dtype>* network;
    int arg1;
    
};

template class Job<float>;

#endif /* JOB_H */
