/*
 * DataInputLayer.h
 *
 *  Created on: Jun 30, 2017
 *      Author: jkim
 */

#ifndef DATAINPUTLAYER_H_
#define DATAINPUTLAYER_H_

#include "InputLayer.h"
#include "DataReader.h"
#include "Datum.h"


template <typename Dtype>
class DataInputLayer : public InputLayer<Dtype> {
public:
	DataInputLayer();
	virtual ~DataInputLayer();

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
	DataReader<Datum> dataReader;


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

#endif /* DATAINPUTLAYER_H_ */
