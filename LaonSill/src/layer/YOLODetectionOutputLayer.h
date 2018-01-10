/**
 * @file YOLODetectionOutputLayer.h
 * @date 2018-01-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef YOLODETECTIONOUTPUTLAYER_H
#define YOLODETECTIONOUTPUTLAYER_H 

#include "common.h"
#include "BaseLayer.h"

template<typename Dtype>
class YOLODetectionOutputLayer : public Layer<Dtype> {
public: 
    YOLODetectionOutputLayer();
    virtual ~YOLODetectionOutputLayer();
	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

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
#endif /* YOLODETECTIONOUTPUTLAYER_H */
