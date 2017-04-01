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
	LayerTest(typename Layer<Dtype>::Builder* builder,
			NetworkConfig<Dtype>* networkConfig = 0)
	: builder(builder), layer(0), networkConfig(networkConfig) {}

	virtual ~LayerTest() {
		cleanUpObject(this->layer);
		cleanUpObject(this->builder);
		cleanUpMap(this->nameDataMap);
	}

	virtual void setUp() {
		buildNameDataMapFromNpzFile(NPZ_PATH, this->builder->_name, this->nameDataMap);
		printNameDataMap(this->nameDataMap, false);

		// 최소 설정만 전달받고 나머지는 npz로부터 추론하는 것이 좋겠다.
		this->layer = this->builder->build();
		if (this->networkConfig != 0) {
			this->layer->setNetworkConfig(this->networkConfig);
		}
		fillLayerDataVec(this->layer->_inputs, this->layer->_inputData);
		fillLayerDataVec(this->layer->_outputs, this->layer->_outputData);
	}

	virtual void cleanUp() {
	}

	virtual void forwardTest() {
		fillData(this->nameDataMap, this->layer->name + SIG_BOTTOM,
				this->layer->_inputData);

		//printDataList(this->layer->_inputData, 0);
		this->layer->feedforward();
		//printDataList(this->layer->_outputData, 0);

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


	void printDataList(const std::vector<Data<Dtype>*>& dataList, int type = 0) {
		Data<Dtype>::printConfig = 1;
		SyncMem<Dtype>::printConfig = 1;

		if (type == 0) {
			for (int j = 0; j < dataList.size(); j++) {
				dataList[j]->print_data({}, false);
			}
		} else if (type == 1) {
			for (int j = 0; j < dataList.size(); j++) {
				dataList[j]->print_grad({}, false);
			}
		}

		Data<Dtype>::printConfig = 0;
		SyncMem<Dtype>::printConfig = 0;
	}


private:
	NetworkConfig<Dtype>* networkConfig;
	typename Layer<Dtype>::Builder* builder;
	Layer<Dtype>* layer;

	map<string, Data<Dtype>*> nameDataMap;
};


#endif /* LAYERTEST_H_ */
