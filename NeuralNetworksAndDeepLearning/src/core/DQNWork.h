/**
 * @file DQNWork.h
 * @date 2016-12-23
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DQNWORK_H
#define DQNWORK_H 

#include "Job.h"
#include "Network.h"
#include "DQNImageLearner.h"

template <typename Dtype>
class DQNWork {
public: 
                    DQNWork() {}
    virtual        ~DQNWork() {}

    static void     buildDQNNetworks(Job* job);
    static void     createDQNImageLearner(Job* job);
    static void     feedForwardDQNNetwork(Job* job);
    static void     pushDQNImageInput(Job* job);

private:
    static void     buildDQNNetwork(DQNImageLearner<Dtype>* learner, Network<Dtype>* network);
};
#endif /* DQNWORK_H */
