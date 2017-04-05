/*
 * LayerInputTest.h
 *
 *  Created on: Mar 27, 2017
 *      Author: jkim
 */

#ifndef LAYERINPUTTEST_H_
#define LAYERINPUTTEST_H_

#include "LayerTestInterface.h"
#include "TestUtil.h"
#include "Layer.h"

using namespace std;


template <typename Dtype>
class LayerInputTest : public LayerTestInterface<Dtype> {
public:
	LayerInputTest(typename Layer<Dtype>::Builder* builder)
	: builder(builder), layer(0) {
		//assert(dynamic_cast<typename InputLayer<Dtype>::Builder*>(builder));
	}

	virtual ~LayerInputTest() {
		cleanUpObject(this->layer);
		cleanUpObject(this->builder);
		cleanUpMap(this->nameDataMap);
	}

	virtual void setUp() {
		// setUpCuda(this->gpuid);

		buildNameDataMapFromNpzFile(NPZ_PATH, this->builder->_name, this->nameDataMap);
		printNameDataMap("nameDataMap", this->nameDataMap, false);

		// 최소 설정만 전달받고 나머지는 npz로부터 추론하는 것이 좋겠다.
		this->layer = this->builder->build();
		fillLayerDataVec(this->layer->_inputs, this->layer->_inputData);
		fillLayerDataVec(this->layer->_outputs, this->layer->_outputData);
	}

	virtual void cleanUp() {
		//cleanUpCuda();
	}

	virtual void forwardTest() {
		this->layer->feedforward();

		compareData(this->nameDataMap, this->layer->name + SIG_TOP,
			this->layer->_outputData, 0);
	}

	virtual void backwardTest() {
		// no backward test for input layer ...
	}


private:
	typename Layer<Dtype>::Builder* builder;
	Layer<Dtype>* layer;

	map<string, Data<Dtype>*> nameDataMap;
};


#endif /* LAYERINPUTTEST_H_ */
