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

template <typename Dtype>
class NetworkTest : public NetworkTestInterface<Dtype> {
public:
	NetworkTest(LayersConfig<Dtype>* layersConfig, const std::string& networkName)
	: layersConfig(layersConfig), networkName(networkName) {}

	virtual ~NetworkTest() {
		cleanUpObject(this->layersConfig);
		cleanUpMap(this->nameParamsOldMap);
		cleanUpMap(this->nameParamsNewMap);
		cleanUpMap(this->nameBlobsMap);
	}

	virtual void setUp() {
		buildNameDataMapFromNpzFile(NPZ_PATH + this->networkName + "/",
				this->networkName + this->paramsOld, this->nameParamsOldMap);
		buildNameDataMapFromNpzFile(NPZ_PATH + this->networkName + "/",
				this->networkName + this->paramsNew, this->nameParamsNewMap);
		buildNameDataMapFromNpzFile(NPZ_PATH + this->networkName + "/",
				this->networkName + this->blobs, this->nameBlobsMap);

		printNameDataMap("nameParamsOldMap", this->nameParamsOldMap, false);
		printNameDataMap("nameParamsNewMap", this->nameParamsNewMap, false);
		printNameDataMap("nameBlobsMap", this->nameBlobsMap, false);
		cout << "build name data map done ... " << endl;

		std::vector<LearnableLayer<Dtype>*>& learnableLayers =
				this->layersConfig->_learnableLayers;

		cout << "fill params ... ---------------------------------------------------" << endl;
		for (int i = 0; i < learnableLayers.size(); i++) {
			fillParam(this->nameParamsOldMap, learnableLayers[i]->name + SIG_PARAMS,
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
		// feedforward
		logStartTest("FEED FORWARD");
		forward();
		dataTest();
		logEndTest("FEED FORWARD");






		std::vector<Layer<Dtype>*>& layers = this->layersConfig->_layers;
		for (int i = 0; i < layers.size(); i++) {
			Layer<Dtype>* layer = layers[i];
			for (int j = 0; j < layer->_inputData.size(); j++) {
				const string dataName = BLOBS_PREFIX + layer->_inputData[j]->_name;
				Data<Dtype>* data = retrieveValueFromMap(this->nameBlobsMap, dataName);
				layer->_inputData[j]->set_host_data(data, 0, false);
			}
		}

		typename std::map<std::string, Layer<Dtype>*>::iterator itr =
				this->layersConfig->_nameLayerMap.find("loss_cls");
		itr->second->_outputData[0]->mutable_host_data()[0] = 3.1143229008f;
		itr->second->_outputData[0]->mutable_host_grad()[0] = 1;

		itr = this->layersConfig->_nameLayerMap.find("loss_bbox");
		itr->second->_outputData[0]->mutable_host_data()[0] = 0.1780370474f;
		itr->second->_outputData[0]->mutable_host_grad()[0] = 1;





		// backpropagation
		logStartTest("BACK PROPAGATION");
		backward();

		/*
		std::vector<Layer<Dtype>*>& layers = this->layersConfig->_layers;
		for (int i = 0; i < layers.size(); i++) {
			Layer<Dtype>* layer = layers[i];
			for (int j = 0; j < layer->_inputData.size(); j++) {
				const string dataName = BLOBS_PREFIX + layer->_inputData[j]->_name;
				Data<Dtype>* data = retrieveValueFromMap(this->nameBlobsMap, dataName);
				layer->_inputData[j]->set_host_grad(data, 0, false);
			}
		}
		*/
		gradTest();
		logEndTest("BACK PROPAGATION");

		/*
		std::vector<LearnableLayer<Dtype>*>& learnableLayers = this->layersConfig->_learnableLayers;
		for (int i = 0; i < learnableLayers.size(); i++) {
			LearnableLayer<Dtype>* learnableLayer = learnableLayers[i];
			for (int j = 0; j < learnableLayer->_params.size(); j++) {
				const string key = learnableLayer->name + SIG_PARAMS + to_string(j);
				Data<float>* param = retrieveValueFromMap(this->nameParamsNewMap, key);
				assert(param != 0);
				learnableLayer->_params[j]->set_host_grad(param, 0, false);
			}
		}
		*/

		// update & compare result
		logStartTest("UPDATE");
		update();
		paramTest();
		logEndTest("UPDATE");
	}

	void feedInputLayerData() {
		InputLayer<Dtype>* inputLayer = this->layersConfig->_inputLayer;

		// 'data'
		const string inputDataName = BLOBS_PREFIX + inputLayer->_outputData[0]->_name;
		Data<Dtype>* data = retrieveValueFromMap(this->nameBlobsMap, inputDataName);
		inputLayer->_outputData[0]->set_host_data(data, 0, true);

		// 'label'
		const string inputLabelName = BLOBS_PREFIX + inputLayer->_outputData[1]->_name;
		Data<Dtype>* label = retrieveValueFromMap(this->nameBlobsMap, inputLabelName);
		inputLayer->_outputData[1]->set_host_data(label, 0, true);
	}


private:
	void forward() {
		//feedInputLayerData();
		std::set<std::string> targetLayerSet;
		//targetLayerSet.insert("relu5");

		for (int i = 0; i < this->layersConfig->_layers.size(); i++) {
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

	void dataTest() {
		for (int i = 0; i < this->layersConfig->_layers.size(); i++) {
			Layer<Dtype>* layer = this->layersConfig->_layers[i];
			cout << "-----------------------------data test at layer " << layer->name << endl;

			if (!compareData(this->nameBlobsMap, BLOBS_PREFIX, layer->_outputData, 0)) {
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

	void gradTest() {
		// caffe의 backward 과정에서 input layer와
		// input layer의 다음 레이어 input data에 대해 backward 진행하지 않기 때문에
		// 적용된 diff가 없으므로 해당 data에 대해서는 체크하지 않는다.
		for (int i = this->layersConfig->_layers.size() - 1; i > 1; i--) {
			Layer<Dtype>* layer = this->layersConfig->_layers[i];

			// test blobs except input layer and second layer
			if (!compareData(this->nameBlobsMap, BLOBS_PREFIX, layer->_inputData, 1)) {
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

	void paramTest() {
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
			if (!compareParam(this->nameParamsNewMap, learnableLayer->name + SIG_PARAMS,
					learnableLayer->_params, 0)) {
				std::cout << "[ERROR] update failed at layer " << learnableLayer->name <<
						std::endl;
				//exit(1);
			} else {
				std::cout << "update succeed at layer " << learnableLayer->name << std::endl;
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
	//Network<Dtype>* network;
	LayersConfig<Dtype>* layersConfig;

	map<string, Data<Dtype>*> nameParamsOldMap;
	map<string, Data<Dtype>*> nameParamsNewMap;
	map<string, Data<Dtype>*> nameBlobsMap;

	const std::string networkName;
	const std::string paramsOld = "_params_old";
	const std::string paramsNew = "_params_new";
	const std::string blobs = "_blobs";
};



#endif /* NETWORKTEST_H_ */
