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
#include "../layer/LayerFactory.h"
#include "../Worker.h"

template <typename Dtype> class DataSet;
template <typename Dtype> class Worker;

using namespace std;



template <typename Dtype>
class LayersConfig {
public:
	class Builder {
	public:
		map<uint32_t, typename Layer<Dtype>::Builder*> _layerWise;

		Builder* layer(typename Layer<Dtype>::Builder* layerBuilder) {
			uint32_t layerIndex = layerBuilder->_id;
			typename map<uint32_t, typename Layer<Dtype>::Builder*>::iterator it;
            it = _layerWise.find(layerIndex);
			if (it != _layerWise.end()) {
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
			vector<LearnableLayer<Dtype>*> learnableLayers;
			map<uint32_t, Layer<Dtype>*> idLayerMap;

			uint32_t layerSize = _layerWise.size();
            typename map<uint32_t, typename Layer<Dtype>::Builder*>::iterator it;

			for (it = _layerWise.begin(); it != _layerWise.end(); it++) {
                Layer<Dtype>* currentLayer = it->second->build();
                const int numNextLayers = currentLayer->getNextLayerSize();
                const int numPrevLayers = currentLayer->getPrevLayerSize();
                if (numNextLayers < 1 && numPrevLayers < 1) {
                    cout << "layer " << currentLayer->getName() << " has no layer relations ... "
                        << endl;
                    exit(1);
                }

                if (numPrevLayers < 1) {
                    //cout << "firstLayer: " << currentLayer->getName() << endl;
                    firstLayers.push_back(currentLayer);
                } else if (numNextLayers < 1) {
                    //cout << "lastLayer: " << currentLayer->getName() << endl;
                    lastLayers.push_back(currentLayer);
                }

                // 학습 레이어 추가
                LearnableLayer<Dtype>* learnableLayer =
                    dynamic_cast<LearnableLayer<Dtype>*>(currentLayer);
                if (learnableLayer) {
                    learnableLayers.push_back(learnableLayer);
                }


                layers.push_back(currentLayer);
                idLayerMap[it->first] = currentLayer;
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
					typename map<uint32_t, Layer<Dtype>*>::iterator it =
                        idLayerMap.find((size_t)currentLayer->getNextLayers()[j]);
					if(it != idLayerMap.end()) {
						currentLayer->getNextLayers()[j] = it->second;
					}
				}
				for(uint32_t j = 0; j < currentLayer->getPrevLayers().size(); j++) {
					typename map<uint32_t, Layer<Dtype>*>::iterator it =
                        idLayerMap.find((size_t)currentLayer->getPrevLayers()[j]);
					if(it != idLayerMap.end()) {
						currentLayer->getPrevLayers()[j] = it->second;
					}
				}
			}

			map<string, Layer<Dtype>*> nameLayerMap;
			for(uint32_t i = 0; i < layers.size(); i++) {
				const string& layerName = layers[i]->getName();
				typename map<string, Layer<Dtype>*>::iterator it = nameLayerMap.find(layerName);
				if(it != nameLayerMap.end()) {
					cout << "layer name used more than once ... : " << layerName << endl;
					exit(1);
				}
				nameLayerMap[layerName] = layers[i];
			}

			return (new LayersConfig(this))
				->firstLayers(firstLayers)
				->lastLayers(lastLayers)
				->layers(layers)
				->learnableLayers(learnableLayers)
                ->nameLayerMap(nameLayerMap);

		}
		void save(ofstream& ofs) {
			uint32_t numLayers = _layerWise.size();
			ofs.write((char*)&numLayers, sizeof(uint32_t));
            typename map<uint32_t, typename Layer<Dtype>::Builder*>::iterator it;
			for (it = _layerWise.begin(); it != _layerWise.end(); it++) {
				it->second->save(ofs);
			}
		}
		void load(ifstream& ifs) {
			uint32_t numLayers;
			ifs.read((char*)&numLayers, sizeof(uint32_t));

			for(uint32_t i = 0; i < numLayers; i++) {
				// create layer builder objects from stream
				typename Layer<Dtype>::Type layerType;
				ifs.read((char*)&layerType, sizeof(uint32_t));

				typename Layer<Dtype>::Builder* layerBuilder =
                    LayerBuilderFactory<Dtype>::create(layerType);
				layerBuilder->load(ifs);

				// add to layerWise
				layer(layerBuilder);
			}
		}

	};

	vector<Layer<Dtype>*> _firstLayers;
	vector<Layer<Dtype>*> _lastLayers;
	vector<Layer<Dtype>*> _layers;
	vector<LearnableLayer<Dtype>*> _learnableLayers;
    InputLayer<Dtype>* _inputLayer;
    vector<OutputLayer<Dtype>*> _outputLayers;
	map<string, Layer<Dtype>*> _nameLayerMap;
	Builder* _builder;

    LayersConfig() {}

	LayersConfig(Builder* builder) {
		this->_builder = builder;
	}
	LayersConfig<Dtype>* firstLayers(vector<Layer<Dtype>*> firstLayers) {
		this->_firstLayers = firstLayers;
        this->_inputLayer = dynamic_cast<InputLayer<Dtype>*>(firstLayers[0]);
		return this;
	}
	LayersConfig<Dtype>* lastLayers(vector<Layer<Dtype>*> lastLayers) {
		this->_lastLayers = lastLayers;
        typename vector<Layer<Dtype>*>::iterator iter;
        for (iter = lastLayers.begin(); iter != lastLayers.end(); iter++) {
            OutputLayer<Dtype>* outputLayer = dynamic_cast<OutputLayer<Dtype>*>(*iter);
            if(!outputLayer) {
                cout << "invalid output layer ... " << endl;
                exit(1);
            }
            _outputLayers.push_back(outputLayer);
        }
		return this;
	}
	LayersConfig<Dtype>* layers(vector<Layer<Dtype>*> layers) {
		this->_layers = layers;
		return this;
	}
	LayersConfig<Dtype>* learnableLayers(vector<LearnableLayer<Dtype>*> learnableLayers) {
		this->_learnableLayers = learnableLayers;
		return this;
	}
	LayersConfig* nameLayerMap(map<string, Layer<Dtype>*> nameLayerMap) {
		this->_nameLayerMap = nameLayerMap;
		return this;
	}
	void save(ofstream& ofs) {
		this->_builder->save(ofs);
	}
};


template class LayersConfig<float>;







enum NetworkStatus {
	Train = 0,
	Test = 1
};


template <typename Dtype>
class NetworkConfig {
public:
	//static const string savePrefix = "network";
	//const string configPostfix;// = ".config";
	//const string paramPostfix;// = ".param";

	class Builder {
	public:
		DataSet<Dtype>* _dataSet;
		vector<Evaluation<Dtype>*> _evaluations;
		vector<NetworkListener*> _networkListeners;
        vector<LayersConfig<Dtype>*> layersConfigs;

		uint32_t _batchSize;
		uint32_t _dop;	/* degree of parallel */
		uint32_t _epochs;
		uint32_t _testInterval;
		uint32_t _saveInterval;
		float _baseLearningRate;
		float _momentum;
		float _weightDecay;
		float _clipGradientsLevel;

		string _savePathPrefix;

		Builder() {
			this->_dataSet = NULL;
			this->_batchSize = 1;
			this->_dop = 1;
			this->_epochs = 1;
			this->_clipGradientsLevel = 35.0f;
		}
		Builder* evaluations(const vector<Evaluation<Dtype>*> evaluations) {
			this->_evaluations = evaluations;
			return this;
		}
		Builder* networkListeners(const vector<NetworkListener*> networkListeners) {
			this->_networkListeners = networkListeners;
			return this;
		}
		Builder* batchSize(uint32_t batchSize) {
			this->_batchSize = batchSize;
			return this;
		}
		Builder* dop(uint32_t dop) {
			this->_dop = dop;
			return this;
		}
		Builder* epochs(uint32_t epochs) {
			this->_epochs = epochs;
			return this;
		}
		Builder* testInterval(uint32_t testInterval) {
			this->_testInterval = testInterval;
			return this;
		}
		Builder* saveInterval(uint32_t saveInterval) {
			this->_saveInterval = saveInterval;
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
		Builder* dataSet(DataSet<Dtype>* dataSet) {
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

			//load()를 학습단계에서도 사용할 경우 ...
			//테스트단계에서만 사용할 경우 dataSet 필요없음 ...
			//if(_dataSet == NULL) {
			//	cout << "dataSet should be set ... " << endl;
			//	exit(1);
			//}

			NetworkConfig* networkConfig = (new NetworkConfig(this))
					->evaluations(_evaluations)
					->networkListeners(_networkListeners)
					->batchSize(_batchSize)
					->dop(_dop)
					->epochs(_epochs)
					->testInterval(_testInterval)
					->saveInterval(_saveInterval)
					->savePathPrefix(_savePathPrefix)
					->clipGradientsLevel(_clipGradientsLevel)
					->dataSet(_dataSet)
					->baseLearningRate(_baseLearningRate)
					->momentum(_momentum)
					->weightDecay(_weightDecay);

            networkConfig->layersConfigs.assign(Worker<Dtype>::consumerCount, NULL);

			return networkConfig;
		}
		void save(ofstream& ofs) {
			if(_savePathPrefix == "") {
				cout << "save path not specified ... " << endl;
				// TODO 죽이지 말고 사용자에게 save path를 입력받도록 하자 ...
				exit(1);
			}

			// save primitives
			ofs.write((char*)&_batchSize, sizeof(uint32_t));					//_batchSize
			ofs.write((char*)&_epochs, sizeof(uint32_t));						//_epochs
			ofs.write((char*)&_baseLearningRate, sizeof(float));				//_baseLearningRate
			ofs.write((char*)&_momentum, sizeof(float));						//_momentum
			ofs.write((char*)&_weightDecay, sizeof(float));						//_weightDecay
			ofs.write((char*)&_clipGradientsLevel, sizeof(float));				//_clipGradientsLevel;

			size_t savePathPrefixLength = _savePathPrefix.size();
			ofs.write((char*)&savePathPrefixLength, sizeof(size_t));
			ofs.write((char*)_savePathPrefix.c_str(), savePathPrefixLength);

            //layersConfigs[0]->save(ofs);
#if 0
			_layersConfig->save(ofs);
#endif
		}
		void load(const string& path) {
			ifstream ifs((path+".config").c_str(), ios::in | ios::binary);

			ifs.read((char*)&_batchSize, sizeof(uint32_t));
			ifs.read((char*)&_epochs, sizeof(uint32_t));
			ifs.read((char*)&_baseLearningRate, sizeof(float));
			ifs.read((char*)&_momentum, sizeof(float));
			ifs.read((char*)&_weightDecay, sizeof(float));
			ifs.read((char*)&_clipGradientsLevel, sizeof(float));

			size_t savePathPrefixLength;
			ifs.read((char*)&savePathPrefixLength, sizeof(size_t));

			char* savePathPrefix_c = new char[savePathPrefixLength+1];
			ifs.read(savePathPrefix_c, savePathPrefixLength);
			savePathPrefix_c[savePathPrefixLength] = '\0';
			_savePathPrefix = savePathPrefix_c;
			delete [] savePathPrefix_c;

			typename LayersConfig<Dtype>::Builder* layersBuilder = new typename LayersConfig<Dtype>::Builder();
			layersBuilder->load(ifs);

#if 0
			_layersConfig = layersBuilder->build();
#endif
            // XXX: 여러대의 GPU를 고려해야 한다..
            LayersConfig<Dtype>* layersConfig = layersBuilder->build();
            layersConfigs.push_back(layersConfig);

			ifs.close();
		}

	};






	NetworkStatus _status;

	DataSet<Dtype>* _dataSet;
	vector<Evaluation<Dtype>*> _evaluations;
	vector<NetworkListener*> _networkListeners;
    vector<LayersConfig<Dtype>*> layersConfigs;

	uint32_t _batchSize;
	uint32_t _dop;
	uint32_t _epochs;
	uint32_t _testInterval;
	uint32_t _saveInterval;
	uint32_t _iterations;
	float _baseLearningRate;
	float _momentum;
	float _weightDecay;

	string _savePathPrefix;
	float _clipGradientsLevel;

	// save & load를 위해서 builder도 일단 저장해 두자.
	Builder* _builder;




	NetworkConfig(Builder* builder) {
		this->_builder = builder;
		this->_iterations = 0;
	}

	NetworkConfig* evaluations(const vector<Evaluation<Dtype>*> evaluations) {
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
	NetworkConfig* dop(uint32_t dop) {
		this->_dop = dop;
		return this;
	}
	NetworkConfig* epochs(uint32_t epochs) {
		this->_epochs = epochs;
		return this;
	}
	NetworkConfig* testInterval(uint32_t testInterval) {
		this->_testInterval = testInterval;
		return this;
	}
	NetworkConfig* saveInterval(uint32_t saveInterval) {
		this->_saveInterval = saveInterval;
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
	NetworkConfig* dataSet(DataSet<Dtype>* dataSet) {
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
#if 0
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
	NetworkConfig* learnableLayers(vector<LearnableLayer<Dtype>*> learnableLayers) {
		this->_learnableLayers = learnableLayers;
		return this;
	}
	NetworkConfig* nameLayerMap(map<string, Layer<Dtype>*> nameLayerMap) {
		this->_nameLayerMap = nameLayerMap;
		return this;
	}
#endif

	void save() {
		// save config
		ofstream configOfs((_savePathPrefix+".config").c_str(), ios::out | ios::binary);
		_builder->save(configOfs);
		configOfs.close();

		// save learned params
        LayersConfig<Dtype>* firstLayersConfig = this->layersConfigs[0];
		ofstream paramOfs((_savePathPrefix+".param").c_str(), ios::out | ios::binary);
		uint32_t numLearnableLayers = firstLayersConfig->_learnableLayers.size();
		for(uint32_t i = 0; i < numLearnableLayers; i++) {
			firstLayersConfig->_learnableLayers[i]->saveParams(paramOfs);
		}
		paramOfs.close();
	}
	void load() {
		cout << _savePathPrefix+".param" << endl;

		ifstream ifs((_savePathPrefix+".param").c_str(), ios::in | ios::binary);
        LayersConfig<Dtype>* firstLayersConfig = this->layersConfigs[0];
		uint32_t numLearnableLayers = firstLayersConfig->_learnableLayers.size();
		for(uint32_t i = 0; i < numLearnableLayers; i++) {
			firstLayersConfig->_learnableLayers[i]->loadParams(ifs);
		}
		ifs.close();
	}
	bool doTest() {
		if(_iterations % _testInterval == 0) return true;
		else return false;
	}
	bool doSave() {
		if(_iterations % _saveInterval == 0) return true;
		else return false;
	}
};


template class NetworkConfig<float>;



#endif /* NETWORKPARAM_H_ */
