/*
 * SoftmaxLayer.h
 *
 *  Created on: Nov 29, 2016
 *      Author: jkim
 */

#ifndef SOFTMAXLAYER_H_
#define SOFTMAXLAYER_H_

#if 1

#include "common.h"
#include "Layer.h"
//#include "Activation.h"
#include "Cuda.h"

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
public:
	SoftmaxLayer();
	virtual ~SoftmaxLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	uint32_t outerNum;
	uint32_t innerNum;

	// used to carry out sum using BLAS
	Data<Dtype> sumMultiplier;
	// intermediate data to hold temporary results.
	Data<Dtype> scale;

	cudnnTensorDescriptor_t inputTensorDesc;
	cudnnTensorDescriptor_t outputTensorDesc;


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

#endif /* SOFTMAXLAYER_H_ */
