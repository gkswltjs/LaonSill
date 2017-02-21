/*
 * LearnableLayerTest.h
 *
 *  Created on: Feb 21, 2017
 *      Author: jkim
 */

#ifndef LEARNABLELAYERTEST_H_
#define LEARNABLELAYERTEST_H_

#include "LayerTestInterface.h"
#include "TestUtil.h"
#include "LearnableLayer.h"

using namespace std;


template <typename Dtype>
class LearnableLayerTest : public LayerTestInterface<Dtype> {
public:
	LearnableLayerTest(typename LearnableLayer<Dtype>::Builder* builder)
	: builder(builder), layer(0) {}

	virtual ~LearnableLayerTest() {
		cleanUpObject(this->layer);
		cleanUpObject(this->builder);
		cleanUpMap(this->nameDataMap);
	}

	virtual void setUp() {
		buildNameDataMapFromNpzFile(NPZ_PATH, this->builder->_name, this->nameDataMap);
		printNameDataMap(this->nameDataMap, false);

		// 최소 설정만 전달받고 나머지는 npz로부터 추론하는 것이 좋겠다.
		this->layer = dynamic_cast<LearnableLayer<Dtype>*>(this->builder->build());
		assert(this->layer != 0);

		fillLayerDataVec(this->layer->_inputs, this->layer->_inputData);
		fillLayerDataVec(this->layer->_outputs, this->layer->_outputData);
	}

	virtual void cleanUp() {}

	virtual void forwardTest() {
		fillData(this->nameDataMap, this->layer->name + SIG_BOTTOM, this->layer->_inputData);
		fillParam(this->nameDataMap, this->layer->name + SIG_PARAMS, this->layer->_params);
		for (uint32_t i = 0; i < this->layer->_params.size(); i++)
			this->layer->_paramsInitialized[i] = true;

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
	typename LearnableLayer<Dtype>::Builder* builder;
	LearnableLayer<Dtype>* layer;

	map<string, Data<Dtype>*> nameDataMap;
};


#endif /* LEARNABLELAYERTEST_H_ */
