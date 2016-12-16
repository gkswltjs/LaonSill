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
    Network<float>             *network;
    void                        buildDQNLayer();
    void                        createNetwork();
    void                        feedForward(int batchSize);
    void                        fillInputData(int imgCount, float* img, int action,
                                    float reward, bool term);

    ALEInputLayer<float>       *inputLayer;
    FullyConnectedLayer<float> *outputLayer;
};
#endif /* ATARINN_H */
