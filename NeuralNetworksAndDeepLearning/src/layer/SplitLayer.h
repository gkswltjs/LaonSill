/*
 * SplitLayer.h
 *
 *  Created on: Nov 8, 2016
 *      Author: jkim
 */

#ifndef SPLITLAYER_H_
#define SPLITLAYER_H_



#include "Layer.h"

template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
public:
	SplitLayer();
	virtual ~SplitLayer();

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

#endif /* SPLITLAYER_H_ */
