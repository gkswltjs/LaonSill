/**
 * @file LayerProp.h
 * @date 2017-04-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef LAYERPROP_H
#define LAYERPROP_H 

#include "BaseLayer.h"

class LayerProp {
public: 
    LayerProp() {
        this->networkID = -1;
        this->layerID = Layer<float>::None;
        this->layerType = -1;
        this->prop = NULL;
    }

    LayerProp(int networkID, int layerID, int layerType, void* prop) {
        this->networkID = networkID;
        this->layerID = layerID;
        this->layerType = layerType;
        this->prop = prop;
    }

    virtual ~LayerProp() {
        if (prop != NULL)
            free(prop);
    }

    int networkID;
    int layerID;
    int layerType;
    void *prop;
};

#endif /* LAYERPROP_H */
