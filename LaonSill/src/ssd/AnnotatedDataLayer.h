/*
 * AnnotatedDataLayer.h
 *
 *  Created on: Sep 4, 2017
 *      Author: jkim
 */

#ifndef ANNOTATEDDATALAYER_H_
#define ANNOTATEDDATALAYER_H_

#include "InputLayer.h"
#include "DataReader.h"
#include "Datum.h"
#include "InputDataProvider.h"
#include "DataTransformer.h"
#include "LayerPropList.h"

template <typename Dtype>
class AnnotatedDataLayer : public InputLayer<Dtype> {
public:
	AnnotatedDataLayer();
	AnnotatedDataLayer(_AnnotatedDataPropLayer* prop);
	virtual ~AnnotatedDataLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	virtual void reshape();
    InputPool* inputPool;

private:
	void load_batch();

private:
	DataReader<class AnnotatedDatum> dataReader;
	std::vector<BatchSampler> batchSamplers;
	std::string labelMapFile;
	DataTransformer<Dtype> dataTransformer;
	bool outputLabels;
	bool hasAnnoType;

	_AnnotatedDataPropLayer* prop;
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

#endif /* ANNOTATEDDATALAYER_H_ */
