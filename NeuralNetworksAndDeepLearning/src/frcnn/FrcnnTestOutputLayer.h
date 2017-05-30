/*
 * FrcnnTestOutputLayer.h
 *
 *  Created on: Dec 16, 2016
 *      Author: jkim
 */

#ifndef FRCNNTESTOUTPUTLAYER_H_
#define FRCNNTESTOUTPUTLAYER_H_

#include "BaseLayer.h"
#include "frcnn_common.h"

template <typename Dtype>
class FrcnnTestOutputLayer : public Layer<Dtype> {
public:
	FrcnnTestOutputLayer();
	virtual ~FrcnnTestOutputLayer();

	virtual void reshape();
	virtual void feedforward();

public:
	std::vector<cv::Scalar> boxColors;


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

#endif /* FRCNNTESTOUTPUTLAYER_H_ */
