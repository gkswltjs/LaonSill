/*
 * NetworkTest.h
 *
 *  Created on: Feb 21, 2017
 *      Author: jkim
 */

#ifndef NETWORKTEST_H_
#define NETWORKTEST_H_

#include "NetworkTestInterface.h"
#include "TestUtil.h"
#include "Util.h"
#include "Network.h"
#include "NetworkConfig.h"
#include "Data.h"
#include "SysLog.h"


#define COPY_INPUT		1




template <typename Dtype>
class NetworkTest : public NetworkTestInterface<Dtype> {
public:
	NetworkTest(LayersConfig<Dtype>* layersConfig, const std::string& networkName,
			const int numSteps)
	: layersConfig(layersConfig), networkName(networkName), numSteps(numSteps) {
		SASSERT0(this->numSteps > 0);
	}

	virtual ~NetworkTest() {
		cleanUpObject(this->layersConfig);
		//cleanUpMap(this->nameParamsOldMap);
		//cleanUpMap(this->nameParamsNewMap);
		for (int i = 0; i <= this->numSteps; i++) {
			cleanUpMap(this->nameParamsMapList[i]);
		}
		//cleanUpMap(this->nameBlobsMap);
		for (int i = 0; i < this->numSteps; i++) {
			cleanUpMap(this->nameBlobsMapList[i]);
		}
	}

	virtual void setUp() {
		std::vector<Layer<Dtype>*>& layers = this->layersConfig->_layers;
		cout << "::: LAYER DATA CONFIGURATION :::" << endl;
		for (int i = 0; i < layers.size(); i++) {
			layers[i]->reshape();
			layers[i]->printDataConfig();
		}


		//buildNameDataMapFromNpzFile(NPZ_PATH + this->networkName + "/",
		//		this->networkName + this->paramsOld, this->nameParamsOldMap);
		//buildNameDataMapFromNpzFile(NPZ_PATH + this->networkName + "/",
		//		this->networkName + this->paramsNew, this->nameParamsNewMap);

		for (int i = 0; i <= this->numSteps; i++) {
			const string strIdx = to_string(i);
			map<string, Data<Dtype>*> nameParamsMap;
			buildNameDataMapFromNpzFile(NPZ_PATH + this->networkName + "/",
					this->networkName + this->params + strIdx, nameParamsMap);
			this->nameParamsMapList.push_back(nameParamsMap);
			printNameDataMap("nameParamsMap" + strIdx, nameParamsMap, false);
		}

		for (int i = 0; i < this->numSteps; i++) {
			const string strIdx = to_string(i);
			map<string, Data<Dtype>*> nameBlobsMap;
			buildNameDataMapFromNpzFile(NPZ_PATH + this->networkName + "/",
					this->networkName + this->blobs + strIdx, nameBlobsMap);
			this->nameBlobsMapList.push_back(nameBlobsMap);
			printNameDataMap("nameBlobsMap" + strIdx, nameBlobsMap, false);
		}

		cout << "build name data map done ... " << endl;


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

	}

	virtual void cleanUp() {

	}

	virtual void updateTest() {
		for (int i = 0; i < this->numSteps; i++) {
			cout << "::::::::::STEP " << i << "::::::::::" << endl;

			// feedforward
			logStartTest("FEED FORWARD");
			forward(i);
			dataTest(i);
			logEndTest("FEED FORWARD");

			/*
			typename std::map<std::string, Layer<Dtype>*>::iterator itr =
					this->layersConfig->_nameLayerMap.find("mbox_loss");
			printDataList(itr->second->_outputData);
			exit(1);
			*/

			// backpropagation
			replaceDataWithGroundTruth(i);
			logStartTest("BACK PROPAGATION");
			backward();
			gradTest(i);
			logEndTest("BACK PROPAGATION");

			// update & compare result
			replaceGradWithGroundTruth(i);

			logStartTest("UPDATE");
			update();
			paramTest(i);
			logEndTest("UPDATE");

			replaceParamWithGroundTruth(i+1, 0);
		}
	}

	void feedInputLayerData(const int nthStep) {
		SASSERT0(nthStep < this->numSteps);
		InputLayer<Dtype>* inputLayer = this->layersConfig->_inputLayer;

		for (int i = 0; i < inputLayer->_outputs.size(); i++) {
			const string inputDataName = BLOBS_PREFIX + inputLayer->_outputData[i]->_name;
			Data<Dtype>* data = retrieveValueFromMap(this->nameBlobsMapList[nthStep], inputDataName);
			inputLayer->_outputData[i]->set_host_data(data, 0, true);
		}
	}




private:
	void forward(const int nthStep) {

#if COPY_INPUT
		feedInputLayerData(nthStep);
#endif
		std::set<std::string> targetLayerSet;
		//targetLayerSet.insert("relu5");

#if COPY_INPUT
		for (int i = 1; i < this->layersConfig->_layers.size(); i++) {
#else
		for (int i = 0; i < this->layersConfig->_layers.size(); i++) {
#endif
			Layer<Dtype>* layer = this->layersConfig->_layers[i];

			if (targetLayerSet.find(layer->name) != targetLayerSet.end()) {
				printDataList(layer->_inputData, 0);
			}

			layer->feedforward();

			if (targetLayerSet.find(layer->name) != targetLayerSet.end()) {
				LearnableLayer<Dtype>* learnableLayer = dynamic_cast<LearnableLayer<Dtype>*>(layer);
				if (learnableLayer) printDataList(learnableLayer->_params, 0);
				printDataList(layer->_outputData, 0);
				//printDataList(layer->_outputData, 1);
				//exit(1);
			}
		}
	}

	void dataTest(const int nthStep) {
		SASSERT0(nthStep < this->numSteps);
		for (int i = 0; i < this->layersConfig->_layers.size(); i++) {
			Layer<Dtype>* layer = this->layersConfig->_layers[i];
			cout << "-----------------------------data test at layer " << layer->name << endl;

			if (!compareData(this->nameBlobsMapList[nthStep], BLOBS_PREFIX, layer->_outputData, 0)) {
				std::cout << "[ERROR] data feedforward failed at layer " << layer->name <<
						std::endl;
			} else {
				std::cout << "data feedforward succeed at layer " << layer->name << std::endl;
			}
		}
	}

	void backward() {
		std::set<std::string> targetLayerSet;
		//targetLayerSet.insert("conv5_relu5_0_split");
		//targetLayerSet.insert("relu5");


		for (int i = this->layersConfig->_layers.size() - 1; i >= 1; i--) {
			Layer<Dtype>* layer = this->layersConfig->_layers[i];

			if (targetLayerSet.find(layer->name) != targetLayerSet.end()) {
				//LearnableLayer<Dtype>* learnableLayer = dynamic_cast<LearnableLayer<Dtype>*>(layer);
				//if (learnableLayer) printDataList(learnableLayer->_params, 0);
				printDataList(layer->_outputData, 1, -1);
			}
			layer->backpropagation();

			if (targetLayerSet.find(layer->name) != targetLayerSet.end()) {
				printDataList(layer->_inputData, 0, -1);
				printDataList(layer->_inputData, 1, -1);
			}
		}
		//exit(1);
	}

	void gradTest(const int nthStep) {
		SASSERT0(nthStep < this->numSteps);
		// caffe의 backward 과정에서 input layer와
		// input layer의 다음 레이어 input data에 대해 backward 진행하지 않기 때문에
		// 적용된 diff가 없으므로 해당 data에 대해서는 체크하지 않는다.
		for (int i = this->layersConfig->_layers.size() - 1; i > 1; i--) {
			Layer<Dtype>* layer = this->layersConfig->_layers[i];

			// test blobs except input layer and second layer
			if (!compareData(this->nameBlobsMapList[nthStep], BLOBS_PREFIX, layer->_inputData, 1)) {
				std::cout << "[ERROR] data backpropagation failed at layer " << layer->name
						<< std::endl;
				//exit(1);
			} else {
				std::cout << "data backpropagation succeed at layer " << layer->name
						<< std::endl;
			}
		}
	}

	void update() {
		std::set<std::string> targetLayerSet;
		//targetLayerSet.insert("conv2");
		//targetLayerSet.insert("rpn_conv/3x3");

		std::cout.precision(15);
		for (int i = 0; i < this->layersConfig->_learnableLayers.size(); i++) {
			LearnableLayer<Dtype>* learnableLayer = this->layersConfig->_learnableLayers[i];

			if (targetLayerSet.find(learnableLayer->name) != targetLayerSet.end()) {
				printDataList(learnableLayer->_params, 0);
				printDataList(learnableLayer->_params, 1);
			}

			learnableLayer->update();

			if (targetLayerSet.find(learnableLayer->name) != targetLayerSet.end()) {
				printDataList(learnableLayer->_params, 0);
				printDataList(learnableLayer->_params, 1);
				exit(1);
			}
		}
	}

	void paramTest(int nthStep) {
		SASSERT0(nthStep < this->numSteps);

		for (int i = 0; i < this->layersConfig->_learnableLayers.size(); i++) {
			LearnableLayer<Dtype>* learnableLayer = this->layersConfig->_learnableLayers[i];

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
					learnableLayer->name + SIG_PARAMS, learnableLayer->_params, 0);
			result = compareParam(this->nameParamsMapList[nthStep + 1],
					learnableLayer->name + SIG_PARAMS, learnableLayer->_params, 1);

			if (!result) {
				std::cout << "[ERROR] update failed at layer " << learnableLayer->name <<
						std::endl;
			} else {
				std::cout << "update succeed at layer " << learnableLayer->name << std::endl;
			}
		}
	}

	void replaceDataWithGroundTruth(int stepIdx) {
		std::vector<Layer<Dtype>*>& layers = this->layersConfig->_layers;
		for (int i = 0; i < layers.size(); i++) {
			Layer<Dtype>* layer = layers[i];
			for (int j = 0; j < layer->_inputData.size(); j++) {
				const string dataName = BLOBS_PREFIX + layer->_inputData[j]->_name;
				Data<Dtype>* data =
						retrieveValueFromMap(this->nameBlobsMapList[stepIdx], dataName);
				layer->_inputData[j]->set_host_data(data, 0, false);
			}
		}

		const string dataName = BLOBS_PREFIX + "mbox_loss";
		Data<Dtype>* data =
				retrieveValueFromMap(this->nameBlobsMapList[stepIdx], dataName);
		Layer<Dtype>* layer = this->layersConfig->_nameLayerMap["mbox_loss"];
		layer->_outputData[0]->set_host_data(data, 0, false);
		layer->_outputData[0]->mutable_host_grad()[0] = 1;


		/*
		typename std::map<std::string, Layer<Dtype>*>::iterator itr =
				this->layersConfig->_nameLayerMap.find("mbox_loss");
		//itr->second->_outputData[0]->mutable_host_data()[0] = 3.01521993f;
		itr->second->_outputData[0]->mutable_host_data()[0] = 25.01589775f;
		itr->second->_outputData[0]->mutable_host_grad()[0] = 1;
		*/
	}

	void replaceGradWithGroundTruth(int stepIdx) {
		std::vector<Layer<Dtype>*>& layers = this->layersConfig->_layers;
		for (int i = 0; i < layers.size(); i++) {
			Layer<Dtype>* layer = layers[i];
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

		std::vector<LearnableLayer<Dtype>*>& learnableLayers =
				this->layersConfig->_learnableLayers;
		for (int i = 0; i < learnableLayers.size(); i++) {
			LearnableLayer<Dtype>* learnableLayer = learnableLayers[i];
			for (int j = 0; j < learnableLayer->_params.size(); j++) {
				const string key = learnableLayer->name + SIG_PARAMS + to_string(j);
				Data<float>* param =
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
	const int numSteps;

	//Network<Dtype>* network;
	LayersConfig<Dtype>* layersConfig;

	//map<string, Data<Dtype>*> nameParamsOldMap;
	//map<string, Data<Dtype>*> nameParamsNewMap;
	vector<map<string, Data<Dtype>*>> nameParamsMapList;
	//map<string, Data<Dtype>*> nameBlobsMap;
	vector<map<string, Data<Dtype>*>> nameBlobsMapList;

	const std::string networkName;
	const std::string params = "_params_";
	//const std::string paramsOld = "_params_old";
	//const std::string paramsNew = "_params_new";
	const std::string blobs = "_blobs_";
};



#endif /* NETWORKTEST_H_ */
