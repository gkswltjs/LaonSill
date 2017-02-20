/*
 * FullyConnectedLayerTest.h
 *
 *  Created on: Feb 16, 2017
 *      Author: jkim
 */

#ifndef FULLYCONNECTEDLAYERTEST_H_
#define FULLYCONNECTEDLAYERTEST_H_


#include "LayerTestInterface.h"
#include "TestUtil.h"
#include "FullyConnectedLayer.h"

using namespace std;


template <typename Dtype>
class FullyConnectedLayerTest : public LayerTestInterface<Dtype> {
public:
	FullyConnectedLayerTest(int gpuid, typename FullyConnectedLayer<Dtype>::Builder* builder)
	: gpuid(gpuid), builder(builder), layer(0) {	}

	virtual ~FullyConnectedLayerTest() {
		if (this->layer)
			delete this->layer;

		if (this->builder)
			delete builder;

		typename map<string, Data<Dtype>*>::iterator itr;
		for (itr = this->nameDataMap.begin(); itr != this->nameDataMap.end(); itr++) {
			if (itr->second)
				delete itr->second;
		}
	}

	virtual void setUp() {
		//setUpCuda(this->gpuid);

		buildNameDataMapFromNpzFile(NPZ_PATH, this->builder->_name, this->nameDataMap);
		printNameDataMap(this->nameDataMap, true);

		// 최소 설정만 전달받고 나머지는 npz로부터 추론하는 것이 좋겠다.
		this->layer = dynamic_cast<FullyConnectedLayer<Dtype>*>(this->builder->build());
		fillLayerDataVec(this->layer->_inputs, this->layer->_inputData);
		fillLayerDataVec(this->layer->_outputs, this->layer->_outputData);
	}

	virtual void cleanUp() {
		//cleanUpCuda();
	}

	virtual void forwardTest() {
		fillData(this->nameDataMap, this->layer->name + SIG_BOTTOM, this->layer->_inputData);
		fillParam(this->nameDataMap, this->layer->name + SIG_PARAMS, this->layer->_params);
		this->layer->_paramsInitialized[0] = true;
		this->layer->_paramsInitialized[1] = true;

		//printData(this->layer->_inputData);
		//printData(this->layer->_params);

		this->layer->feedforward();

		compareData(this->nameDataMap, this->layer->name + SIG_TOP, this->layer->_outputData,
				0);
	}

	virtual void backwardTest() {
		fillData(this->nameDataMap, this->layer->name + SIG_BOTTOM, this->layer->_inputData);
		fillData(this->nameDataMap, this->layer->name + SIG_TOP, this->layer->_outputData);
		fillParam(this->nameDataMap, this->layer->name + SIG_PARAMS, this->layer->_params);

		this->layer->backpropagation();

		compareData(this->nameDataMap, this->layer->name + SIG_BOTTOM,
				this->layer->_inputData, 1);
	}


private:
	const int gpuid;
	typename FullyConnectedLayer<Dtype>::Builder* builder;
	FullyConnectedLayer<Dtype>* layer;

	map<string, Data<Dtype>*> nameDataMap;
};




#endif /* FULLYCONNECTEDLAYERTEST_H_ */
