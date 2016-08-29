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
#include "../layer/LearnableLayer.h"
#include "../monitor/NetworkListener.h"

class DataSet;

using namespace std;



template <typename Dtype>
class LayersConfig {
public:
	class Builder {
	public:
		map<uint32_t, typename Layer<Dtype>::Builder*> _layerWise;

		Builder* layer(typename Layer<Dtype>::Builder* layerBuilder) {
			uint32_t layerIndex = layerBuilder->_id;
			typename map<uint32_t, typename Layer<Dtype>::Builder*>::iterator it = _layerWise.find(layerIndex);
			if(it != _layerWise.end()) {
				cout << "already contained layer index " << layerIndex << endl;
				exit(1);
			} else {
				_layerWise[layerIndex] = layerBuilder;
			}
			return this;
		}
		LayersConfig<Dtype>* build() {
			cout << "LayersConfig::build() ... " << endl;

			vector<Layer<Dtype>*> firstLayers;
			vector<Layer<Dtype>*> lastLayers;
			vector<Layer<Dtype>*> layers;
			vector<LearnableLayer*> learnableLayers;
			map<uint32_t, Layer<Dtype>*> idLayerMap;

			uint32_t layerSize = _layerWise.size();

			//for(uint32_t i = 0; i < layerSize; i++) {
			for(typename map<uint32_t, typename Layer<Dtype>::Builder*>::iterator it = _layerWise.begin(); it != _layerWise.end(); it++) {
				//map<uint32_t, Layer<Dtype>::Builder*>::iterator it = _layerWise.find(i);
				//if(it == _layerWise.end()) {
				//	cout << "no layer found for layer index " << i << endl;
				//	exit(1);
				//} else {
					Layer<Dtype>* currentLayer = it->second->build();
					const int numNextLayers = currentLayer->getNextLayerSize();
					const int numPrevLayers = currentLayer->getPrevLayerSize();
					if(numNextLayers < 1 && numPrevLayers < 1) {
						cout << "layer " << currentLayer->getName() << " has no layer relations ... " << endl;
						exit(1);
					}

					if(numPrevLayers < 1) {
						//cout << "firstLayer: " << currentLayer->getName() << endl;
						firstLayers.push_back(currentLayer);
					} else if(numNextLayers < 1) {
						//cout << "lastLayer: " << currentLayer->getName() << endl;
						lastLayers.push_back(currentLayer);
					}

					// 학습 레이어 추가
					LearnableLayer* learnableLayer = dynamic_cast<LearnableLayer*>(currentLayer);
					if(learnableLayer) {
						learnableLayers.push_back(learnableLayer);
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
				Layer<Dtype>* currentLayer = layers[i];
				for(uint32_t j = 0; j < currentLayer->getNextLayers().size(); j++) {
					//currentLayer->getNextLayers()[j] = layers[(size_t)currentLayer->getNextLayers()[j]];
					typename map<uint32_t, Layer<Dtype>*>::iterator it = idLayerMap.find((size_t)currentLayer->getNextLayers()[j]);
					if(it != idLayerMap.end()) {
						currentLayer->getNextLayers()[j] = it->second;
					}
				}
				for(uint32_t j = 0; j < currentLayer->getPrevLayers().size(); j++) {
					//currentLayer->getPrevLayers()[j] = layers[(size_t)currentLayer->getPrevLayers()[j]];
					typename map<uint32_t, Layer<Dtype>*>::iterator it = idLayerMap.find((size_t)currentLayer->getPrevLayers()[j]);
					if(it != idLayerMap.end()) {
						currentLayer->getPrevLayers()[j] = it->second;
					}
				}
			}

			return (new LayersConfig(this))
				->firstLayers(firstLayers)
				->lastLayers(lastLayers)
				->layers(layers)
				->learnableLayers(learnableLayers);
		}

	};




	vector<Layer<Dtype>*> _firstLayers;
	vector<Layer<Dtype>*> _lastLayers;
	vector<Layer<Dtype>*> _layers;
	vector<LearnableLayer*> _learnableLayers;

	LayersConfig(Builder* builder) {}
	LayersConfig<Dtype>* firstLayers(vector<Layer<Dtype>*> firstLayers) {
		this->_firstLayers = firstLayers;
		return this;
	}
	LayersConfig<Dtype>* lastLayers(vector<Layer<Dtype>*> lastLayers) {
		this->_lastLayers = lastLayers;
		return this;
	}
	LayersConfig<Dtype>* layers(vector<Layer<Dtype>*> layers) {
		this->_layers = layers;
		return this;
	}
	LayersConfig<Dtype>* learnableLayers(vector<LearnableLayer*> learnableLayers) {
		this->_learnableLayers = learnableLayers;
		return this;
	}
};


template class LayersConfig<float>;










template <typename Dtype>
class NetworkConfig {
public:
	class Builder {
	public:
		DataSet* _dataSet;
		vector<Evaluation*> _evaluations;
		vector<NetworkListener*> _networkListeners;
		LayersConfig<Dtype>* _layersConfig;

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
		Builder* layersConfig(LayersConfig<Dtype>* layersConfig) {
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
			map<string, Layer<Dtype>*> nameLayerMap;
			for(uint32_t i = 0; i < _layersConfig->_layers.size(); i++) {
				const string& layerName = _layersConfig->_layers[i]->getName();
				typename map<string, Layer<Dtype>*>::iterator it = nameLayerMap.find(layerName);
				if(it != nameLayerMap.end()) {
					cout << "layer name used more than once ... : " << layerName << endl;
					exit(1);
				}
				nameLayerMap[layerName] = _layersConfig->_layers[i];
			}

			vector<Layer<Dtype>*>& firstLayers = _layersConfig->_firstLayers;
			if(firstLayers.size() != 1) {
				cout << "too many first layers ... " << endl;
				exit(1);
			}

			InputLayer<Dtype>* inputLayer = dynamic_cast<InputLayer<Dtype>*>(firstLayers[0]);
			if(!inputLayer) {
				cout << "no input layer ... " << endl;
				exit(1);
			}

			vector<Layer<Dtype>*>& lastLayers = _layersConfig->_lastLayers;
			if(lastLayers.size() < 1) {
				cout << "no output layer ... " << endl;
			}
			vector<OutputLayer<Dtype>*> outputLayers;
			for(uint32_t i = 0; i < lastLayers.size(); i++) {
				OutputLayer<Dtype>* outputLayer = dynamic_cast<OutputLayer<Dtype>*>(lastLayers[i]);
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
					->learnableLayers(_layersConfig->_learnableLayers)
					->nameLayerMap(nameLayerMap);

			for(uint32_t i = 0; i < _layersConfig->_layers.size(); i++) {
				_layersConfig->_layers[i]->setNetworkConfig(networkConfig);
			}

			return networkConfig;
		}
	};


	InputLayer<Dtype>* _inputLayer;
	vector<OutputLayer<Dtype>*> _outputLayers;
	vector<Layer<Dtype>*> _layers;
	vector<LearnableLayer*> _learnableLayers;
	map<string, Layer<Dtype>*> _nameLayerMap;

	DataSet* _dataSet;
	vector<Evaluation*> _evaluations;
	vector<NetworkListener*> _networkListeners;
	LayersConfig<Dtype>* _layersConfig;

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
	NetworkConfig* inputLayer(InputLayer<Dtype>* inputLayer) {
		this->_inputLayer = inputLayer;
		return this;
	}
	NetworkConfig* outputLayers(vector<OutputLayer<Dtype>*> outputLayers) {
		this->_outputLayers = outputLayers;
		return this;
	}
	NetworkConfig* layers(vector<Layer<Dtype>*> layers) {
		this->_layers = layers;
		return this;
	}
	NetworkConfig* learnableLayers(vector<LearnableLayer*> learnableLayers) {
		this->_learnableLayers = learnableLayers;
		return this;
	}
	NetworkConfig* nameLayerMap(map<string, Layer<Dtype>*> nameLayerMap) {
		this->_nameLayerMap = nameLayerMap;
		return this;
	}

};


template class NetworkConfig<float>;



#endif /* NETWORKPARAM_H_ */
