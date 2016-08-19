/*
 * NetworkParam.h
 *
 *  Created on: 2016. 8. 12.
 *      Author: jhkim
 */

#ifndef NETWORKPARAM_H_
#define NETWORKPARAM_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../evaluation/Evaluation.h"
#include "../layer/Layer.h"
#include "../layer/InputLayer.h"
#include "../layer/OutputLayer.h"
#include "../monitor/NetworkListener.h"

class DataSet;

using namespace std;

class LayersConfig {
public:
	class Builder {
	public:
		map<uint32_t, Layer::Builder*> _layerWise;

		Builder* layer(/*uint32_t layerIndex, */Layer::Builder* layerBuilder) {
			uint32_t layerIndex = layerBuilder->_id;
			map<uint32_t, Layer::Builder*>::iterator it = _layerWise.find(layerIndex);
			if(it != _layerWise.end()) {
				cout << "already contained layer index " << layerIndex << endl;
				exit(1);
			} else {
				_layerWise[layerIndex] = layerBuilder;
			}
			return this;
		}
		LayersConfig* build() {
			cout << "LayersConfig::build() ... " << endl;

			vector<Layer*> firstLayers;
			vector<Layer*> lastLayers;
			vector<Layer*> layers;
			map<uint32_t, Layer*> idLayerMap;

			uint32_t layerSize = _layerWise.size();

			//for(uint32_t i = 0; i < layerSize; i++) {
			for(map<uint32_t, Layer::Builder*>::iterator it = _layerWise.begin(); it != _layerWise.end(); it++) {
				//map<uint32_t, Layer::Builder*>::iterator it = _layerWise.find(i);
				//if(it == _layerWise.end()) {
				//	cout << "no layer found for layer index " << i << endl;
				//	exit(1);
				//} else {
					Layer* currentLayer = it->second->build();
					const int numNextLayers = currentLayer->getNextLayerSize();
					const int numPrevLayers = currentLayer->getPrevLayerSize();
					if(numNextLayers < 1 && numPrevLayers < 1) {
						cout << "layer " << currentLayer->getName() << " has no layer relations ... " << endl;
						exit(1);
					}

					if(numPrevLayers < 1) {
						cout << "firstLayer: " << currentLayer->getName().c_str() << endl;
						firstLayers.push_back(currentLayer);
					} else if(numNextLayers < 1) {
						cout << "lastLayer: " << currentLayer->getName().c_str() << endl;
						lastLayers.push_back(currentLayer);
					}
					layers.push_back(currentLayer);
					idLayerMap[it->first] = currentLayer;
				//}
			}

			if(firstLayers.size() < 1) {
				cout << "no input layer ... " << endl;
				exit(1);
			}
			if(lastLayers.size() < 1) {
				cout << "no output layer ... " << endl;
				exit(1);
			}

			// 생성된 레이어들의 prev, next 관계를 설정한다.
			for(uint32_t i = 0; i < layers.size(); i++) {
				Layer* currentLayer = layers[i];
				for(uint32_t j = 0; j < currentLayer->getNextLayers().size(); j++) {
					//currentLayer->getNextLayers()[j] = layers[(size_t)currentLayer->getNextLayers()[j]];
					map<uint32_t, Layer*>::iterator it = idLayerMap.find((size_t)currentLayer->getNextLayers()[j]);
					if(it != idLayerMap.end()) {
						currentLayer->getNextLayers()[j] = it->second;
					}
				}
				for(uint32_t j = 0; j < currentLayer->getPrevLayers().size(); j++) {
					//currentLayer->getPrevLayers()[j] = layers[(size_t)currentLayer->getPrevLayers()[j]];
					map<uint32_t, Layer*>::iterator it = idLayerMap.find((size_t)currentLayer->getPrevLayers()[j]);
					if(it != idLayerMap.end()) {
						currentLayer->getPrevLayers()[j] = it->second;
					}
				}
			}

			return (new LayersConfig(this))
				->firstLayers(firstLayers)
				->lastLayers(lastLayers)
				->layers(layers);
		}
	};




	vector<Layer*> _firstLayers;
	vector<Layer*> _lastLayers;
	vector<Layer*> _layers;

	LayersConfig(Builder* builder) {}
	LayersConfig* firstLayers(vector<Layer*> firstLayers) {
		this->_firstLayers = firstLayers;
		return this;
	}
	LayersConfig* lastLayers(vector<Layer*> lastLayers) {
		this->_lastLayers = lastLayers;
		return this;
	}
	LayersConfig* layers(vector<Layer*> layers) {
		this->_layers = layers;
		return this;
	}
};














class NetworkConfig {
public:
	class Builder {
	public:
		DataSet* _dataSet;
		vector<Evaluation*> _evaluations;
		vector<NetworkListener*> _networkListeners;
		LayersConfig* _layersConfig;

		uint32_t _batchSize;
		uint32_t _epochs;
		float _baseLearningRate;
		float _momentum;
		float _weightDecay;

		string _savePathPrefix;
		float _clipGradientsLevel;

		Builder() {
			this->_dataSet = NULL;
			this->_batchSize = 1;
			this->_epochs = 1;
			this->_clipGradientsLevel = 35.0f;
		}
		Builder* evaluations(const vector<Evaluation*> evaluations) {
			this->_evaluations = evaluations;
			return this;
		}
		Builder* networkListeners(const vector<NetworkListener*> networkListeners) {
			this->_networkListeners = networkListeners;
			return this;
		}
		Builder* layersConfig(LayersConfig* layersConfig) {
			this->_layersConfig = layersConfig;
			return this;
		}
		Builder* batchSize(uint32_t batchSize) {
			this->_batchSize = batchSize;
			return this;
		}
		Builder* epochs(uint32_t epochs) {
			this->_epochs = epochs;
			return this;
		}
		Builder* savePathPrefix(string savePathPrefix) {
			this->_savePathPrefix = savePathPrefix;
			return this;
		}
		Builder* clipGradientsLevel(float clipGradientsLevel) {
			this->_clipGradientsLevel = clipGradientsLevel;
			return this;
		}
		Builder* dataSet(DataSet* dataSet) {
			this->_dataSet = dataSet;
			return this;
		}
		Builder* baseLearningRate(float baseLearningRate) {
			this->_baseLearningRate = baseLearningRate;
			return this;
		}
		Builder* momentum(float momentum) {
			this->_momentum = momentum;
			return this;
		}
		Builder* weightDecay(float weightDecay) {
			this->_weightDecay = weightDecay;
			return this;
		}
		NetworkConfig* build() {
			map<string, Layer*> nameLayerMap;
			for(uint32_t i = 0; i < _layersConfig->_layers.size(); i++) {
				const string& layerName = _layersConfig->_layers[i]->getName();
				map<string, Layer*>::iterator it = nameLayerMap.find(layerName);
				if(it != nameLayerMap.end()) {
					cout << "layer name used more than once ... : " << layerName << endl;
					exit(1);
				}
				nameLayerMap[layerName] = _layersConfig->_layers[i];
			}

			vector<Layer*>& firstLayers = _layersConfig->_firstLayers;
			if(firstLayers.size() != 1) {
				cout << "too many first layers ... " << endl;
				exit(1);
			}

			InputLayer* inputLayer = dynamic_cast<InputLayer*>(firstLayers[0]);
			if(!inputLayer) {
				cout << "no input layer ... " << endl;
				exit(1);
			}

			vector<Layer*>& lastLayers = _layersConfig->_lastLayers;
			if(lastLayers.size() < 1) {
				cout << "no output layer ... " << endl;
			}
			vector<OutputLayer*> outputLayers;
			for(uint32_t i = 0; i < lastLayers.size(); i++) {
				OutputLayer* outputLayer = dynamic_cast<OutputLayer*>(lastLayers[i]);
				if(!outputLayer) {
					cout << "invalid output layer ... " << endl;
					exit(1);
				}
				outputLayers.push_back(outputLayer);
			}

			NetworkConfig* networkConfig = (new NetworkConfig())
					->evaluations(_evaluations)
					->networkListeners(_networkListeners)
					->batchSize(_batchSize)
					->epochs(_epochs)
					->savePathPrefix(_savePathPrefix)
					->clipGradientsLevel(_clipGradientsLevel)
					->dataSet(_dataSet)
					->baseLearningRate(_baseLearningRate)
					->momentum(_momentum)
					->weightDecay(_weightDecay)
					->inputLayer(inputLayer)
					->outputLayers(outputLayers)
					->layers(_layersConfig->_layers)
					->nameLayerMap(nameLayerMap);

			for(uint32_t i = 0; i < _layersConfig->_layers.size(); i++) {
				_layersConfig->_layers[i]->setNetworkConfig(networkConfig);
			}

			return networkConfig;
		}
	};


	InputLayer* _inputLayer;
	vector<OutputLayer*> _outputLayers;
	vector<Layer*> _layers;
	map<string, Layer*> _nameLayerMap;

	DataSet* _dataSet;
	vector<Evaluation*> _evaluations;
	vector<NetworkListener*> _networkListeners;
	LayersConfig* _layersConfig;

	uint32_t _batchSize;
	uint32_t _epochs;
	float _baseLearningRate;
	float _momentum;
	float _weightDecay;

	string _savePathPrefix;
	float _clipGradientsLevel;

	NetworkConfig() {}

	NetworkConfig* evaluations(const vector<Evaluation*> evaluations) {
		this->_evaluations = evaluations;
		return this;
	}
	NetworkConfig* networkListeners(const vector<NetworkListener*> networkListeners) {
		this->_networkListeners = networkListeners;
		return this;
	}
	NetworkConfig* batchSize(uint32_t batchSize) {
		this->_batchSize = batchSize;
		return this;
	}
	NetworkConfig* epochs(uint32_t epochs) {
		this->_epochs = epochs;
		return this;
	}
	NetworkConfig* savePathPrefix(string savePathPrefix) {
		this->_savePathPrefix = savePathPrefix;
		return this;
	}
	NetworkConfig* clipGradientsLevel(float clipGradientsLevel) {
		this->_clipGradientsLevel = clipGradientsLevel;
		return this;
	}
	NetworkConfig* dataSet(DataSet* dataSet) {
		this->_dataSet = dataSet;
		return this;
	}
	NetworkConfig* baseLearningRate(float baseLearningRate) {
		this->_baseLearningRate = baseLearningRate;
		return this;
	}
	NetworkConfig* momentum(float momentum) {
		this->_momentum = momentum;
		return this;
	}
	NetworkConfig* weightDecay(float weightDecay) {
		this->_weightDecay = weightDecay;
		return this;
	}
	NetworkConfig* inputLayer(InputLayer* inputLayer) {
		this->_inputLayer = inputLayer;
		return this;
	}
	NetworkConfig* outputLayers(vector<OutputLayer*> outputLayers) {
		this->_outputLayers = outputLayers;
		return this;
	}
	NetworkConfig* layers(vector<Layer*> layers) {
		this->_layers = layers;
		return this;
	}
	NetworkConfig* nameLayerMap(map<string, Layer*> nameLayerMap) {
		this->_nameLayerMap = nameLayerMap;
		return this;
	}

};




#endif /* NETWORKPARAM_H_ */
