/**
 * @file AtariNN.h
 * @date 2016-12-14
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef ATARINN_H
#define ATARINN_H 

template <typename Dtype> class Network;

#include "ALEInputLayer.h"

class AtariNN {
public: 
                            AtariNN() {}
    virtual                ~AtariNN() {}

    int                     networkId;
    Network<float>         *network;
    void                    buildDQNLayer();
    void                    createNetwork();

    ALEInputLayer<float>   *inputLayer;
};
#endif /* ATARINN_H */
