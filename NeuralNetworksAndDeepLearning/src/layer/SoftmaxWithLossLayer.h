/*
 * SoftmaxWithLoss.h
 *
 *  Created on: Nov 23, 2016
 *      Author: jkim
 */

#ifndef SOFTMAXWITHLOSSLAYER_H_
#define SOFTMAXWITHLOSSLAYER_H_

#if 1
#include "common.h"
#include "LossLayer.h"
#include "SoftmaxLayer.h"
#include "Cuda.h"

template <typename Dtype>
class SoftmaxWithLossLayer : public LossLayer<Dtype> {
public:
    SoftmaxWithLossLayer();
	virtual ~SoftmaxWithLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

private:
	Dtype getNormalizer(int validCount);

public:
	Data<Dtype> prob;

private:
	uint32_t outerNum;
	uint32_t innerNum;

	SoftmaxLayer<Dtype>* softmaxLayer;

	cudnnTensorDescriptor_t inputTensorDesc;
	cudnnTensorDescriptor_t probTensorDesc;

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
#endif

#endif /* SOFTMAXWITHLOSSLAYER_H_ */
