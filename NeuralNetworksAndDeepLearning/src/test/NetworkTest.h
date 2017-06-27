/*
 * NetworkTest.h
 *
 *  Created on: Feb 21, 2017
 *      Author: jkim
 */

#ifndef NETWORKTEST_H_
#define NETWORKTEST_H_

#include <set>

#include "NetworkTestInterface.h"
#include "TestUtil.h"
#include "Util.h"
#include "Network.h"
#include "Data.h"
#include "SysLog.h"
#include "PlanParser.h"
#include "WorkContext.h"
#include "PropMgmt.h"
#include "InputLayer.h"
#include "PlanOptimizer.h"
#include "SplitLayer.h"


#define COPY_INPUT		1




template <typename Dtype>
class NetworkTest : public NetworkTestInterface<Dtype> {
public:
	NetworkTest(const std::string& networkFilePath, const std::string& networkName,
			const int numSteps)
	: networkFilePath(networkFilePath), networkName(networkName), numSteps(numSteps) {
		SASSERT0(this->networkFilePath.empty() != true);
		SASSERT0(this->networkName.empty() != true);
		SASSERT0(this->numSteps > 0);
	}

	virtual ~NetworkTest() {}

	virtual void setUp() {
		prepareContext();
		retrieveLayers();
		printLayerList();
		printLayerDataConfig();
		loadParams();
		loadBlobs();
		fillParams();
	}

	virtual void cleanUp() {
		for (int i = 0; i <= this->numSteps; i++) {
			cleanUpMap(this->nameParamsMapList[i]);
		}
		for (int i = 0; i < this->numSteps; i++) {
			cleanUpMap(this->nameBlobsMapList[i]);
		}
	}

	virtual void updateTest() {
		for (int i = 0; i < this->numSteps; i++) {
			cout << "::::::::::STEP " << i << "::::::::::" << endl;

			// feedforward
			logStartTest("FEED FORWARD");
			forward(i);
			dataTest(i);
			logEndTest("FEED FORWARD");

			// backpropagation
			//replaceDataWithGroundTruth(i);
			logStartTest("BACK PROPAGATION");
			backward();
			gradTest(i);
			logEndTest("BACK PROPAGATION");

#if 0
			for (int i = 0; i < this->learnableLayers.size(); i++) {
				int layerID = this->learnableLayers[i].first;
				LearnableLayer<Dtype>* learnableLayer = this->learnableLayers[i].second;
				printDataList(learnableLayer->_params, 1);

			}
#endif

			// update & compare result
			//replaceGradWithGroundTruth(i);
			logStartTest("UPDATE");
			update();

#if 0
			for (int i = 0; i < this->learnableLayers.size(); i++) {
				int layerID = this->learnableLayers[i].first;
				LearnableLayer<Dtype>* learnableLayer = this->learnableLayers[i].second;
				printDataList(learnableLayer->_params, 1);

			}
			exit(1);
#endif
			paramTest(i);
			logEndTest("UPDATE");
			//replaceParamWithGroundTruth(i+1, 0);
		}
	}


private:
	void prepareContext() {
		this->networkID = PlanParser::loadNetwork(this->networkFilePath);
		this->network = Network<Dtype>::getNetworkFromID(this->networkID);
		this->network->build(this->numSteps);

		WorkContext::updateNetwork(this->networkID);
		//PlanOptimizer::buildPlans(this->networkID);
		WorkContext::updatePlan(0);
	}

	void retrieveLayers() {
		PhysicalPlan* pp = WorkContext::curPhysicalPlan;
		for (map<int, void*>::iterator iter = pp->instanceMap.begin();
			iter != pp->instanceMap.end(); iter++) {
			int layerID = iter->first;
			void* instancePtr = iter->second;


			Layer<Dtype>* layer = (Layer<Dtype>*)instancePtr;


			if (!dynamic_cast<SplitLayer<Dtype>*>(layer) && 

				// skip inner layer
				this->network->isInnerLayer(layerID)) {
					continue;
			}
			
			this->outerLayers.push_back(std::make_pair(layerID, layer));

			if (dynamic_cast<SplitLayer<Dtype>*>(layer)) {
				continue;
			}

			WorkContext::updateLayer(this->networkID, layerID);

			this->layers.push_back(std::make_pair(layerID, layer));

			if (dynamic_cast<InputLayer<Dtype>*>(layer)) {
				this->inputLayer = std::make_pair(layerID, (InputLayer<Dtype>*)instancePtr);
			}

			if (dynamic_cast<LossLayer<Dtype>*>(layer)) {
				this->lossLayers.push_back(
						std::make_pair(layerID, (LossLayer<Dtype>*)instancePtr));
			}

			if (SLPROP_BASE(learnable)) {
				this->learnableLayers.push_back(
						std::make_pair(layerID, (LearnableLayer<Dtype>*)instancePtr));
			}
		}
		SASSERT0(this->inputLayer.second);
	}

	void printLayerList() {
		cout << "Layers: " << endl;
		for (int i = 0; i < this->layers.size(); i++) {
			int layerID = this->layers[i].first;
			Layer<Dtype>* layer = this->layers[i].second;

			WorkContext::updateLayer(this->networkID, layerID);
			cout << layer->getName() << endl;
		}

		cout << endl << "LearnableLayers: " << endl;
		for (int i = 0; i < this->learnableLayers.size(); i++) {
			int layerID = this->learnableLayers[i].first;
			LearnableLayer<Dtype>* learnableLayer = this->learnableLayers[i].second;

			WorkContext::updateLayer(this->networkID, layerID);
			cout << learnableLayer->getName() << endl;
		}
	}

	void printLayerDataConfig() {
		cout << "::: LAYER DATA CONFIGURATION :::" << endl;
		for (int i = 0; i < this->layers.size(); i++) {
			//layers[i]->reshape();
			WorkContext::updateLayer(this->networkID, this->layers[i].first);
			this->layers[i].second->printDataConfig();
		}
	}

	void loadParams() {
		// XXX: inference test를 위해 = 제거,
		// 일반 테스트시 '<' --> '<='로 복구해야 함!!!
		for (int i = 0; i <= this->numSteps; i++) {
			const string strIdx = to_string(i);
			map<string, Data<Dtype>*> nameParamsMap;
			buildNameDataMapFromNpzFile(NPZ_PATH + this->networkName + "/",
					this->networkName + this->params + strIdx, nameParamsMap);
			this->nameParamsMapList.push_back(nameParamsMap);
			printNameDataMap("nameParamsMap" + strIdx, nameParamsMap, false);
		}
	}

	void loadBlobs() {
		for (int i = 0; i < this->numSteps; i++) {
			const string strIdx = to_string(i);
			map<string, Data<Dtype>*> nameBlobsMap;
			buildNameDataMapFromNpzFile(NPZ_PATH + this->networkName + "/",
					this->networkName + this->blobs + strIdx, nameBlobsMap);
			this->nameBlobsMapList.push_back(nameBlobsMap);
			printNameDataMap("nameBlobsMap" + strIdx, nameBlobsMap, false);
		}
	}

	void fillParams() {
		for (int i = 0; i < this->learnableLayers.size(); i++) {
			WorkContext::updateLayer(this->networkID, this->learnableLayers[i].first);
			LearnableLayer<Dtype>* learnableLayer = this->learnableLayers[i].second;
			fillParam(this->nameParamsMapList[0], learnableLayer->getName() + SIG_PARAMS,
					learnableLayer->_params);
			// 반드시 외부에서 params init되었음을 설정해야 한다.
			for (int j = 0; j < learnableLayer->_params.size(); j++) {
				learnableLayer->_paramsInitialized[j] = true;
			}
		}
		/*
		std::vector<LearnableLayer<Dtype>*>& learnableLayers =
				this->layersConfig->_learnableLayers;

		cout << "fill params ... ---------------------------------------------------" << endl;
		for (int i = 0; i < learnableLayers.size(); i++) {
			fillParam(this->nameParamsMapList[0], learnableLayers[i]->name + SIG_PARAMS,
					learnableLayers[i]->_params);
			// 반드시 외부에서 params init되었음을 설정해야 한다.
			for (int j = 0; j < learnableLayers[i]->_params.size(); j++) {
				learnableLayers[i]->_paramsInitialized[j] = true;
			}
		}
		cout << "-------------------------------------------------------------------" << endl;
		*/
	}









	void forward(const int nthStep) {
#if COPY_INPUT
		feedInputLayerData(nthStep);
#endif
		this->network->runPlanType(PlanType::PLANTYPE_FORWARD, false);
	}

	void feedInputLayerData(const int nthStep) {
		SASSERT0(nthStep < this->numSteps);

		InputLayer<Dtype>* inputLayer = this->inputLayer.second;
		for (int i = 0; i < inputLayer->_outputData.size(); i++) {
			const string inputDataName = BLOBS_PREFIX + inputLayer->_outputData[i]->_name;
			Data<Dtype>* data = retrieveValueFromMap(this->nameBlobsMapList[nthStep], inputDataName);

			// 
			//inputLayer->_outputData[i]->set_host_data(data, 0, false);

			// XXX: for frcnn only ...
			inputLayer->_outputData[i]->set_host_data(data, 0, true);
			if (i == 1) {
				const vector<uint32_t> shape = inputLayer->_outputData[1]->getShape();
				inputLayer->_outputData[1]->reshape({1, 1, 1, shape[1]});
				inputLayer->_outputData[1]->print_shape();
			} else if (i == 2) {
				const vector<uint32_t> shape = inputLayer->_outputData[2]->getShape();
				inputLayer->_outputData[2]->reshape({1, 1, shape[0], shape[1]});
				inputLayer->_outputData[2]->print_shape();
			}

		}
		//printDataList(inputLayer->_outputData, 0);
	}

	void dataTest(const int nthStep) {
		// split layer issue와 관련하여 ...
		// data test에서 split layer의 data들은 input과 output이 동일한 data를 share하므로
		// 굳이 따로 비교를 하지 않아도 될 것으로 보임.
		SASSERT0(nthStep < this->numSteps);
		for (int i = 0; i < this->layers.size(); i++) {
			int layerID = this->layers[i].first;
			Layer<Dtype>* layer = this->layers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			cout << "-----------------------------data test at layer " << layer->getName() << endl;
			if (!compareData(this->nameBlobsMapList[nthStep], BLOBS_PREFIX, layer->_outputData, 0)) {
				std::cout << "[ERROR] data feedforward failed at layer " << layer->getName() <<
						std::endl;
			} else {
				std::cout << "data feedforward succeed at layer " << layer->getName() << std::endl;
			}
		}
	}

	void backward() {
		this->network->runPlanType(PlanType::PLANTYPE_BACKWARD, false);
	}

	void gradTest(const int nthStep) {
		SASSERT0(nthStep < this->numSteps);
		// caffe의 backward 과정에서 input layer와
		// input layer의 다음 레이어 input data에 대해 backward 진행하지 않기 때문에
		// 적용된 diff가 없으므로 해당 data에 대해서는 체크하지 않는다.
		for (int i = this->layers.size() - 1; i > 1; i--) {
			int layerID = this->layers[i].first;
			Layer<Dtype>* layer = this->layers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			// test blobs except input layer and second layer
			if (!compareData(this->nameBlobsMapList[nthStep], BLOBS_PREFIX, layer->_inputData, 1)) {
				std::cout << "[ERROR] data backpropagation failed at layer " << layer->getName()
						<< std::endl;
			} else {
				std::cout << "data backpropagation succeed at layer " << layer->getName() <<
						std::endl;
			}
		}
	}

	void update() {
		std::cout.precision(15);
		this->network->runPlanType(PlanType::PLANTYPE_UPDATE, false);
	}

	void paramTest(int nthStep) {
		SASSERT0(nthStep < this->numSteps);

		for (int i = 0; i < this->learnableLayers.size(); i++) {
			int layerID = this->learnableLayers[i].first;
			LearnableLayer<Dtype>* learnableLayer = this->learnableLayers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			// test final delta
			// params grad는 update 과정에서 오염됨.
			/*
			if (!compareParam(this->nameParamsNewMap,
					learnableLayer->name + SIG_PARAMS, learnableLayer->_params, 1)) {
				std::cout << "[ERROR] param backpropagation failed at layer " <<
						learnableLayer->name << std::endl;
				//exit(1);
			} else {
				std::cout << "param backpropagation succeed at layer " <<
						learnableLayer->name << std::endl;
			}
			*/

			// test final params
			bool result = false;
			result = compareParam(this->nameParamsMapList[nthStep + 1],
					learnableLayer->getName() + SIG_PARAMS, learnableLayer->_params, 0);
			result = compareParam(this->nameParamsMapList[nthStep + 1],
					learnableLayer->getName() + SIG_PARAMS, learnableLayer->_params, 1);

			if (!result) {
				std::cout << "[ERROR] update failed at layer " <<
						learnableLayer->getName() << std::endl;
			} else {
				std::cout << "update succeed at layer " << learnableLayer->getName() <<
						std::endl;
			}
		}
	}


	void replaceDataWithGroundTruth(int stepIdx) {

		for (int i = 0; i < this->layers.size(); i++) {
			int layerID = this->layers[i].first;
			Layer<Dtype>* layer = this->layers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			for (int j = 0; j < layer->_inputData.size(); j++) {
				const string dataName = BLOBS_PREFIX + layer->_inputData[j]->_name;
				Data<Dtype>* data =
						retrieveValueFromMap(this->nameBlobsMapList[stepIdx], dataName);
				layer->_inputData[j]->set_host_data(data, 0, false);
			}
		}

		// for loss layer output data ...
		for (int i = 0; i < this->lossLayers.size(); i++) {
			int layerID = this->lossLayers[i].first;
			LossLayer<Dtype>* lossLayer = this->lossLayers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			const std::string lossDataName = lossLayer->_outputData[0]->_name;
			const string dataName = BLOBS_PREFIX + lossDataName;
			Data<Dtype>* data =
					retrieveValueFromMap(this->nameBlobsMapList[stepIdx], dataName);

			lossLayer->_outputData[0]->set_host_data(data, 0, false);
			lossLayer->_outputData[0]->set_host_grad(data, 0, false);
		}
	}


/*
	void replaceDataWithGroundTruth(int stepIdx) {
		for (int i = 0; i < this->layers.size(); i++) {
			int layerID = this->layers[i].first;
			Layer<Dtype>* layer = this->layers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			for (int j = 0; j < layer->_inputData.size(); j++) {
				const string dataName = BLOBS_PREFIX + layer->_inputData[j]->_name;
				Data<Dtype>* data =
						retrieveValueFromMap(this->nameBlobsMapList[stepIdx], dataName);
				layer->_inputData[j]->set_host_data(data, 0, false);
			}
			//printDataList(layer->_inputData, 0);
		}

		for (int i = 0; i < this->lossLayers.size(); i++) {
			int layerID = this->lossLayers[i].first;
			LossLayer<Dtype>* lossLayer = this->lossLayers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			const std::string lossDataName = lossLayer->_outputData[0]->_name;
			const string dataName = BLOBS_PREFIX + lossDataName;
			Data<Dtype>* data =
					retrieveValueFromMap(this->nameBlobsMapList[stepIdx], dataName);

			lossLayer->_outputData[0]->set_host_data(data, 0, false);
			lossLayer->_outputData[0]->set_host_grad(data, 0, false);

			//printDataList(lossLayer->_outputData, 0);
			//printDataList(lossLayer->_outputData, 1);
		}
	}
	i*/

	void replaceGradWithGroundTruth(int stepIdx) {
		for (int i = 0; i < this->layers.size(); i++) {
			Layer<Dtype>* layer = this->layers[i].second;
			for (int j = 0; j < layer->_inputData.size(); j++) {
				const string dataName = BLOBS_PREFIX + layer->_inputData[j]->_name;
				Data<Dtype>* data =
						retrieveValueFromMap(this->nameBlobsMapList[stepIdx], dataName);
				layer->_inputData[j]->set_host_grad(data, 0, false);
			}
		}
	}

	void replaceParamWithGroundTruth(int stepIdx, int type) {
		SASSERT0(stepIdx >= 0);
		SASSERT0(type == 0 || type == 1);

		//std::vector<LearnableLayer<Dtype>*>& learnableLayers = this->layersConfig->_learnableLayers;
		for (int i = 0; i < this->learnableLayers.size(); i++) {
			int layerID = this->learnableLayers[i].first;
			LearnableLayer<Dtype>* learnableLayer = this->learnableLayers[i].second;
			WorkContext::updateLayer(this->networkID, layerID);

			for (int j = 0; j < learnableLayer->_params.size(); j++) {
				const string key = learnableLayer->getName() + SIG_PARAMS + to_string(j);
				Data<Dtype>* param =
						retrieveValueFromMap(this->nameParamsMapList[stepIdx], key);
				SASSERT0(param != 0);
				switch(type) {
				case 0:
					learnableLayer->_params[j]->set_host_data(param, 0, false);
					break;
				case 1:
					learnableLayer->_params[j]->set_host_grad(param, 0, false);
					break;
				}
			}
		}
	}

	void logStartTest(const std::string& testName) {
		cout << "<<< " + testName + " TEST ... -------------------------------" << endl;
	}

	void logEndTest(const std::string& testName) {
		cout << ">>> " + testName + " TEST DONE ... --------------------------" << endl;
	}

	void printDataList(const std::vector<Data<Dtype>*>& dataList, int type = 0, int summary = 6) {
		Data<Dtype>::printConfig = 1;
		SyncMem<Dtype>::printConfig = 1;

		if (type == 0) {
			for (int j = 0; j < dataList.size(); j++) {
				dataList[j]->print_data({}, false, summary);
			}
		} else if (type == 1) {
			for (int j = 0; j < dataList.size(); j++) {
				dataList[j]->print_grad({}, false, summary);
			}
		}

		Data<Dtype>::printConfig = 0;
		SyncMem<Dtype>::printConfig = 0;
	}



private:
	const std::string networkFilePath;
	const std::string networkName;
	const int numSteps;

	int networkID;
	Network<Dtype>* network;

	vector<map<string, Data<Dtype>*>> nameParamsMapList;
	vector<map<string, Data<Dtype>*>> nameBlobsMapList;

	const std::string params = "_params_";
	const std::string blobs = "_blobs_";

	std::pair<int, InputLayer<Dtype>*> inputLayer;
	std::vector<std::pair<int, Layer<Dtype>*>> layers;
	std::vector<std::pair<int, Layer<Dtype>*>> outerLayers;
	std::vector<std::pair<int, LearnableLayer<Dtype>*>> learnableLayers;
	std::vector<std::pair<int, LossLayer<Dtype>*>> lossLayers;
};



#endif /* NETWORKTEST_H_ */
