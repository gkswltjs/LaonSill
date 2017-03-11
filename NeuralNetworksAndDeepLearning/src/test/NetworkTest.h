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

		printNameDataMap(this->nameParamsOldMap, false);
		printNameDataMap(this->nameParamsNewMap, false);
		printNameDataMap(this->nameBlobsMap, false);

		std::vector<LearnableLayer<Dtype>*>& learnableLayers =
				this->layersConfig->_learnableLayers;

		for (int i = 0; i < learnableLayers.size(); i++) {
			fillParam(this->nameParamsOldMap, learnableLayers[i]->name + SIG_PARAMS,
					learnableLayers[i]->_params);
			// 반드시 외부에서 params init되었음을 설정해야 한다.
			for (int j = 0; j < learnableLayers[i]->_params.size(); j++) {
				learnableLayers[i]->_paramsInitialized[j] = true;
			}
		}
	}

	virtual void cleanUp() {

	}

	virtual void updateTest() {
		// feedforward
		logStartTest("FEED FORWARD");
		forward();
		dataTest();
		logEndTest("FEED FORWARD");

		// backpropagation
		logStartTest("BACK PROPAGATION");
		backward();
		gradTest();
		logEndTest("BACK PROPAGATION");

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

		/*
		printConfigOn();
		inputLayer->_outputData[0]->print_data({}, false);
		inputLayer->_outputData[1]->print_data({}, false);
		printConfigOff();
		*/

	}


private:
	void forward() {
		feedInputLayerData();

		for (int i = 0; i < this->layersConfig->_layers.size(); i++) {
			Layer<Dtype>* layer = this->layersConfig->_layers[i];
			layer->feedforward();
		}
	}

	void dataTest() {
		for (int i = 0; i < this->layersConfig->_layers.size(); i++) {
			Layer<Dtype>* layer = this->layersConfig->_layers[i];

			if (!compareData(this->nameBlobsMap, BLOBS_PREFIX, layer->_outputData, 0)) {
				std::cout << "[ERROR] data feedforward failed at layer " << layer->name <<
						std::endl;
			} else {
				std::cout << "data feedforward succeed at layer " << layer->name << std::endl;
			}
		}
	}

	void backward() {
		for (int i = this->layersConfig->_layers.size() - 1; i >= 1; i--) {
			Layer<Dtype>* layer = this->layersConfig->_layers[i];
			layer->backpropagation();
		}
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
		for (int i = 0; i < this->layersConfig->_learnableLayers.size(); i++) {
			LearnableLayer<Dtype>* learnableLayer = this->layersConfig->_learnableLayers[i];
			learnableLayer->update();
		}
	}

	void paramTest() {
		for (int i = 0; i < this->layersConfig->_learnableLayers.size(); i++) {
			LearnableLayer<Dtype>* learnableLayer = this->layersConfig->_learnableLayers[i];

			// test final delta
			if (!compareParam(this->nameParamsNewMap,
					learnableLayer->name + SIG_PARAMS, learnableLayer->_params, 1)) {
				std::cout << "[ERROR] param backpropagation failed at layer " <<
						learnableLayer->name << std::endl;
				//exit(1);
			} else {
				std::cout << "param backpropagation succeed at layer " <<
						learnableLayer->name << std::endl;
			}

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
