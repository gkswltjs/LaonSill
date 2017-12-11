/*
 * AnnotatedLiveDataLayer.h
 *
 *  Created on: Sep 4, 2017
 *      Author: jkim
 */

#ifndef ANNOTATEDLIVEDATALAYER_H_
#define ANNOTATEDLIVEDATALAYER_H_

#include "InputLayer.h"
#include "Datum.h"
#include "InputDataProvider.h"
#include "DataTransformer.h"
#include "LayerPropList.h"

template <typename Dtype>
class AnnotatedLiveDataLayer : public InputLayer<Dtype> {
public:
	AnnotatedLiveDataLayer();
	virtual ~AnnotatedLiveDataLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();
	virtual void reshape();

private:
	void load_batch();

private:
	DataTransformer<Dtype> dataTransformer;

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

#endif /* ANNOTATEDLIVEDATALAYER_H_ */
