/**
 * @file AtariNN.h
 * @date 2016-12-14
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef ATARINN_H
#define ATARINN_H 

#include "ALEInputLayer.h"
#include "Network.h"

template <typename Dtype> class Network;

class AtariNN {
public: 
                AtariNN(int rowCount, int colCount, int channelCount, int actionCount);
    virtual    ~AtariNN() {}

    int         dqnImageLearnerID;
    int         networkQID;
    int         networkQHeadID;

    void        buildDQNNetworks();
    void        createDQNImageLearner();
    void        feedForward(int batchSize);
    int         stepNetwork(float lastReward, int lastAction, int lastTerm, float* state);

private:
    int         rowCount;
    int         colCount;
    int         channelCount;
    int         actionCount;
};
#endif /* ATARINN_H */
