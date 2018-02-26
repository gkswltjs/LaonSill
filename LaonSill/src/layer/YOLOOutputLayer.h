/*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
 | @ file      YOLOOutputLayer.h
 * @ date      2018-02-06
 | @ author    SUN
 * @ brief
 | @ details
 *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*/

#include "common.h"
#include "BaseLayer.h"

#ifndef YOLOOUTPUTLAYER_H
#define YOLOOUTPUTLAYER_H

template<typename Dtype>
class YOLOOutputLayer : public Layer<Dtype> {
public:
    YOLOOutputLayer();
    virtual ~YOLOOutputLayer();

	virtual void reshape();
	virtual void feedforward();

private:
    void YOLOOutputForward(const Dtype* inputData);

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

#endif /* YOLOOUTPUTLAYER_H */
