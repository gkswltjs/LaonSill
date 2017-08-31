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
#include "BaseLayer.h"
#include "SysLog.h"
#include "Network.h"

using namespace std;






template <typename Dtype>
class LayerTest : public LayerTestInterface<Dtype> {
public:
	LayerTest(const std::string& networkFilePath, const std::string& networkName,
			const std::string& targetLayerName, const int numAfterSteps = 10,
			const NetworkStatus status = NetworkStatus::Train)
	: networkFilePath(networkFilePath), networkName(networkName),
	  targetLayerName(targetLayerName), numAfterSteps(numAfterSteps),
	  status(status) {
		SASSERT0(!this->networkFilePath.empty());
		SASSERT0(!this->networkName.empty());
		SASSERT0(!this->targetLayerName.empty());
		SASSERT0(this->numAfterSteps > 0);	// save network시 주석 처리해야 함.
	}
	virtual void setUp() {
		this->networkID = PrepareContext<Dtype>(this->networkFilePath, 1);
		this->network = Network<Dtype>::getNetworkFromID(this->networkID);

		RetrieveLayers<float>(this->networkID, NULL, &this->layers, NULL, NULL, &this->learnableLayers);
		BuildNameLayerMap(this->networkID, this->layers, this->nameLayerMap);
		PrintLayerList(this->networkID, &this->layers, &this->learnableLayers);
		PrintLayerDataConfig(this->networkID, this->layers);

		LoadParams(this->networkName, 1, status, this->nameParamsMapList);
		PrintNameDataMapList("nameParamsMap", this->nameParamsMapList);

		LoadBlobs(this->networkName, 1, this->nameBlobsMapList);
		PrintNameDataMapList("nameBlobsMap", this->nameBlobsMapList);
	}

	virtual void cleanUp() {
		cleanUpMap(this->nameParamsMapList[0]);
		cleanUpMap(this->nameBlobsMapList[0]);
	}

	virtual void forwardTest() {
		auto itr = this->nameLayerMap.find(this->targetLayerName);
		SASSERT(itr != this->nameLayerMap.end(), "[ERROR] INVALID LAYER: %s",
				this->targetLayerName.c_str());

		int layerID = itr->second.first;
		Layer<Dtype>* layer = itr->second.second;

		FillDatum(this->networkID, this->nameBlobsMapList[0], itr->second, DataEndType::INPUT);
		LearnableLayer<Dtype>* learnableLayer = dynamic_cast<LearnableLayer<Dtype>*>(layer);
		if (learnableLayer) {
			std::pair<int, LearnableLayer<Dtype>*> learnableLayerPair =
					std::make_pair(layerID, learnableLayer);
			FillParam(this->networkID, this->nameParamsMapList[0], learnableLayerPair);

			printConfigOn();
			learnableLayer->_params[0]->print_data({}, false);
			printConfigOff();
		}

		/*
		cout << "-----------------before feed forward ... " << endl;
		printConfigOn();
		for (int i = 0; i < layer->_inputData.size(); i++)
			layer->_inputData[i]->print_data({}, false);
		for (int i = 0; i < layer->_outputData.size(); i++)
			layer->_outputData[i]->print_data({}, false);
		printConfigOff();
		*/

		layer->feedforward();

		/*
		cout << "-----------------after feed forward ... " << endl;
		printConfigOn();
		for (int i = 0; i < layer->_inputData.size(); i++)
			layer->_inputData[i]->print_data({}, false);
		for (int i = 0; i < layer->_outputData.size(); i++)
			layer->_outputData[i]->print_data({}, false);
		printConfigOff();
		*/

		CompareData(this->networkID, this->nameBlobsMapList[0], itr->second, DataEndType::OUTPUT);
	}

	virtual void backwardTest() {
		auto itr = this->nameLayerMap.find(this->targetLayerName);
		SASSERT(itr != this->nameLayerMap.end(), "[ERROR] INVALID LAYER: %s",
				this->targetLayerName.c_str());

		int layerID = itr->second.first;
		Layer<Dtype>* layer = itr->second.second;

		FillDatum(this->networkID, this->nameBlobsMapList[0], itr->second, DataEndType::OUTPUT);
		LearnableLayer<Dtype>* learnableLayer = dynamic_cast<LearnableLayer<Dtype>*>(layer);
		if (learnableLayer) {
			std::pair<int, LearnableLayer<Dtype>*> learnableLayerPair =
					std::make_pair(layerID, learnableLayer);
			FillParam(this->networkID, this->nameParamsMapList[0], learnableLayerPair);
		}

		/*
		cout << "-----------------before back propagation ... " << endl;
		printConfigOn();
		for (int i = 0; i < layer->_inputData.size(); i++)
			layer->_inputData[i]->print_grad({}, false);
		for (int i = 0; i < layer->_outputData.size(); i++)
			layer->_outputData[i]->print_grad({}, false);
		printConfigOff();
		*/

		layer->backpropagation();


		/*
		cout << "-----------------after back propagation ... " << endl;
		printConfigOn();
		for (int i = 0; i < layer->_inputData.size(); i++)
			layer->_inputData[i]->print_grad({}, false);
		for (int i = 0; i < layer->_outputData.size(); i++)
			layer->_outputData[i]->print_grad({}, false);
		printConfigOff();
		*/

		CompareData(this->networkID, this->nameBlobsMapList[0], itr->second, DataEndType::INPUT);
	}

public:
	Network<Dtype>* network;


private:
	const std::string networkFilePath;
	const std::string networkName;
	const std::string targetLayerName;
	const int numAfterSteps;
	const NetworkStatus status;

	int networkID;


	std::vector<std::pair<int, Layer<Dtype>*>> layers;
	std::vector<std::pair<int, LearnableLayer<Dtype>*>> learnableLayers;

	std::vector<std::map<std::string, Data<Dtype>*>> nameParamsMapList;
	std::vector<std::map<std::string, Data<Dtype>*>> nameBlobsMapList;

	std::map<std::string, std::pair<int, Layer<Dtype>*>> nameLayerMap;
};


#if 0
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
		printNameDataMap("nameDataMap", this->nameDataMap, false);

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
		//printDataList(this->layer->_inputData, 1);

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
#endif


#endif /* LAYERTEST_H_ */
