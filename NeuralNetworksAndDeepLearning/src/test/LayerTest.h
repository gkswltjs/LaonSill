/*
 * LayerTest.h
 *
 *  Created on: Feb 20, 2017
 *      Author: jkim
 */

#ifndef LAYERTEST_H_
#define LAYERTEST_H_

#include "LayerTestInterface.h"
#include "TestUtil.h"
#include "Layer.h"

using namespace std;


template <typename Dtype>
class LayerTest : public LayerTestInterface<Dtype> {
public:
	LayerTest(typename Layer<Dtype>::Builder* builder)
	: builder(builder), layer(0) {}

	virtual ~LayerTest() {
		cleanUpObject(this->layer);
		cleanUpObject(this->builder);
		cleanUpMap(this->nameDataMap);
	}

	virtual void setUp() {
		// setUpCuda(this->gpuid);

		buildNameDataMapFromNpzFile(NPZ_PATH, this->builder->_name, this->nameDataMap);
		printNameDataMap(this->nameDataMap, false);

		// 최소 설정만 전달받고 나머지는 npz로부터 추론하는 것이 좋겠다.
		this->layer = this->builder->build();
		fillLayerDataVec(this->layer->_inputs, this->layer->_inputData);
		fillLayerDataVec(this->layer->_outputs, this->layer->_outputData);
	}

	virtual void cleanUp() {
		//cleanUpCuda();
	}

	virtual void forwardTest() {
		fillData(this->nameDataMap, this->layer->name + SIG_BOTTOM,
				this->layer->_inputData);

		this->layer->feedforward();

		compareData(this->nameDataMap, this->layer->name + SIG_TOP,
			this->layer->_outputData, 0);
	}

	virtual void backwardTest() {
		fillData(this->nameDataMap, this->layer->name + SIG_BOTTOM,
				this->layer->_inputData);
		fillData(this->nameDataMap, this->layer->name + SIG_TOP,
				this->layer->_outputData);

		this->layer->backpropagation();

		compareData(this->nameDataMap, this->layer->name + SIG_BOTTOM,
			this->layer->_inputData, 1);
	}


private:
	typename Layer<Dtype>::Builder* builder;
	Layer<Dtype>* layer;

	map<string, Data<Dtype>*> nameDataMap;
};


#endif /* LAYERTEST_H_ */
