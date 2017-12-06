/*
 * AccuracyLayer.h
 *
 *  Created on: Apr 25, 2017
 *      Author: jkim
 */

#ifndef ACCURACYLAYER_H_
#define ACCURACYLAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "MeasureLayer.h"

template <typename Dtype>
class AccuracyLayer : public MeasureLayer<Dtype> {
public:
	AccuracyLayer();
	virtual ~AccuracyLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

	Dtype getAccuracy();
    Dtype measure();


private:
	//uint32_t topK;
	//int labelAxis;

	bool hasIgnoreLabel;
	//int ignoreLabel;

	int outerNum;
	int innerNum;

	int numCorrect;
	int numIterations;


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

#endif /* ACCURACYLAYER_H_ */
