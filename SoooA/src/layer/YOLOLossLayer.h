/**
 * @file YOLOLossLayer.h
 * @date 2017-04-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef YOLOLOSSLAYER_H
#define YOLOLOSSLAYER_H 

#include "common.h"
#include "LossLayer.h"
#include "LayerConfig.h"

template<typename Dtype>
class YOLOLossLayer : public LossLayer<Dtype> {
public: 
    YOLOLossLayer();
    virtual ~YOLOLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

public:
    /****************************************************************************
     * layer callback functions 
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
};

#endif /* YOLOLOSSLAYER_H */
