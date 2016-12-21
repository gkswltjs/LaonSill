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
                                AtariNN() {}
    virtual                    ~AtariNN() {}

    int                         networkId;
    void                        buildDQNLayer();
    void                        createNetwork();
    void                        feedForward(int batchSize);
    void                        pushData(float lastReward, int lastAction, int lastTerm,
                                    float* state);
};
#endif /* ATARINN_H */
