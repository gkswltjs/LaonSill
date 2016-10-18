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

template <typename Dtype> class DataSet;

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
			vector<LearnableLayer<Dtype>*> learnableLayers;
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
					LearnableLayer<Dtype>* learnableLayer = dynamic_cast<LearnableLayer<Dtype>*>(currentLayer);
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
		void save(ofstream& ofs) {
			uint32_t numLayers = _layerWise.size();
			ofs.write((char*)&numLayers, sizeof(uint32_t));
			for(typename map<uint32_t, typename Layer<Dtype>::Builder*>::iterator it = _layerWise.begin(); it != _layerWise.end(); it++) {
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

				typename Layer<Dtype>::Builder* layerBuilder = LayerBuilderFactory<Dtype>::create(layerType);
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
	Builder* _builder;

	LayersConfig(Builder* builder) {
		this->_builder = builder;
	}
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
	LayersConfig<Dtype>* learnableLayers(vector<LearnableLayer<Dtype>*> learnableLayers) {
		this->_learnableLayers = learnableLayers;
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

enum LRPolicy {
	Fixed = 0,
	Step,
	Exp,
	Inv,
	Multistep,
	Poly
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
		LayersConfig<Dtype>* _layersConfig;

		uint32_t _batchSize;
		uint32_t _epochs;
		uint32_t _testInterval;
		uint32_t _saveInterval;
		uint32_t _stepSize;					// update _baseLearningRate
		float _baseLearningRate;
		float _momentum;
		float _weightDecay;
		float _clipGradientsLevel;
		float _gamma;

		string _savePathPrefix;

		LRPolicy _lrPolicy;

		io_dim _inDim;


		Builder() {
			this->_dataSet = NULL;
			this->_batchSize = 1;
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
		Builder* testInterval(uint32_t testInterval) {
			this->_testInterval = testInterval;
			return this;
		}
		Builder* saveInterval(uint32_t saveInterval) {
			this->_saveInterval = saveInterval;
			return this;
		}
		Builder* stepSize(uint32_t stepSize) {
			this->_stepSize = stepSize;
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
		Builder* gamma(float gamma) {
			this->_gamma = gamma;
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
		Builder* lrPolicy(LRPolicy lrPolicy) {
			this->_lrPolicy = lrPolicy;
			return this;
		}
		Builder* inputShape(const vector<uint32_t>& inputShape) {
			this->_inDim.rows = inputShape[0];
			this->_inDim.cols = inputShape[1];
			this->_inDim.channels = inputShape[2];
			return this;
		}
		NetworkConfig* build() {

			//load()를 학습단계에서도 사용할 경우 ...
			//테스트단계에서만 사용할 경우 dataSet 필요없음 ...
			//if(_dataSet == NULL) {
			//	cout << "dataSet should be set ... " << endl;
			//	exit(1);
			//}

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

			if(_dataSet) {
				_inDim.rows = _dataSet->getRows();
				_inDim.cols = _dataSet->getCols();
				_inDim.channels = _dataSet->getChannels();
			}
			_inDim.batches = _batchSize;


			NetworkConfig* networkConfig = (new NetworkConfig(this))
					->evaluations(_evaluations)
					->networkListeners(_networkListeners)
					->batchSize(_batchSize)
					->epochs(_epochs)
					->testInterval(_testInterval)
					->saveInterval(_saveInterval)
					->stepSize(_stepSize)
					->savePathPrefix(_savePathPrefix)
					->clipGradientsLevel(_clipGradientsLevel)
					->gamma(_gamma)
					->dataSet(_dataSet)
					->baseLearningRate(_baseLearningRate)
					->momentum(_momentum)
					->weightDecay(_weightDecay)
					->inputLayer(inputLayer)
					->outputLayers(outputLayers)
					->layers(_layersConfig->_layers)
					->learnableLayers(_layersConfig->_learnableLayers)
					->nameLayerMap(nameLayerMap)
					->lrPolicy(_lrPolicy)
					->inDim(_inDim);

			for(uint32_t i = 0; i < _layersConfig->_layers.size(); i++) {
				_layersConfig->_layers[i]->setNetworkConfig(networkConfig);
			}
			inputLayer->shape(0, _inDim);

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
			ofs.write((char*)&_testInterval, sizeof(uint32_t));
			ofs.write((char*)&_saveInterval, sizeof(uint32_t));
			ofs.write((char*)&_stepSize, sizeof(uint32_t));

			ofs.write((char*)&_baseLearningRate, sizeof(float));				//_baseLearningRate
			ofs.write((char*)&_momentum, sizeof(float));						//_momentum
			ofs.write((char*)&_weightDecay, sizeof(float));						//_weightDecay
			ofs.write((char*)&_clipGradientsLevel, sizeof(float));				//_clipGradientsLevel;
			ofs.write((char*)&_gamma, sizeof(float));

			ofs.write((char*)&_lrPolicy, sizeof(uint32_t));

			size_t savePathPrefixLength = _savePathPrefix.size();
			ofs.write((char*)&savePathPrefixLength, sizeof(size_t));
			ofs.write((char*)_savePathPrefix.c_str(), savePathPrefixLength);

			ofs.write((char*)&_inDim, sizeof(io_dim));

			_layersConfig->save(ofs);
		}
		void load(const string& path) {
			ifstream ifs((path+".config").c_str(), ios::in | ios::binary);

			ifs.read((char*)&_batchSize, sizeof(uint32_t));
			ifs.read((char*)&_epochs, sizeof(uint32_t));
			ifs.read((char*)&_testInterval, sizeof(uint32_t));
			ifs.read((char*)&_saveInterval, sizeof(uint32_t));
			ifs.read((char*)&_stepSize, sizeof(uint32_t));

			ifs.read((char*)&_baseLearningRate, sizeof(float));
			ifs.read((char*)&_momentum, sizeof(float));
			ifs.read((char*)&_weightDecay, sizeof(float));
			ifs.read((char*)&_clipGradientsLevel, sizeof(float));
			ifs.read((char*)&_gamma, sizeof(float));

			ifs.read((char*)&_lrPolicy, sizeof(uint32_t));

			size_t savePathPrefixLength;
			ifs.read((char*)&savePathPrefixLength, sizeof(size_t));

			char* savePathPrefix_c = new char[savePathPrefixLength+1];
			ifs.read(savePathPrefix_c, savePathPrefixLength);
			savePathPrefix_c[savePathPrefixLength] = '\0';
			_savePathPrefix = savePathPrefix_c;
			delete [] savePathPrefix_c;


			ifs.read((char*)&_inDim, sizeof(io_dim));

			typename LayersConfig<Dtype>::Builder* layersBuilder = new typename LayersConfig<Dtype>::Builder();
			layersBuilder->load(ifs);

			_layersConfig = layersBuilder->build();

			ifs.close();
		}
		void print() {
			cout << "batchSize: " << _batchSize << endl;
			cout << "epochs: " << _epochs << endl;
			cout << "testInterval: " << _testInterval << endl;
			cout << "saveInterval: " << _saveInterval << endl;
			cout << "stepSize: " << _stepSize << endl;

			cout << "baseLearningRate: " << _baseLearningRate << endl;
			cout << "momentum: " << _momentum << endl;
			cout << "weightDecay: " << _weightDecay << endl;
			cout << "clipGradientsLevel: " << _clipGradientsLevel << endl;
			cout << "gamma: " << _gamma << endl;

			cout << "savePathPrefix: " << _savePathPrefix << endl;
			cout << "lrPolicy: " << _lrPolicy << endl;

			cout << "inDim->channels: " << _inDim.channels << endl;
			cout << "inDim->rows: " << _inDim.rows << endl;
			cout << "inDim->cols: " << _inDim.cols << endl;
		}

	};






	NetworkStatus _status;
	LRPolicy _lrPolicy;

	InputLayer<Dtype>* _inputLayer;
	vector<OutputLayer<Dtype>*> _outputLayers;
	vector<Layer<Dtype>*> _layers;
	vector<LearnableLayer<Dtype>*> _learnableLayers;
	map<string, Layer<Dtype>*> _nameLayerMap;

	DataSet<Dtype>* _dataSet;
	vector<Evaluation<Dtype>*> _evaluations;
	vector<NetworkListener*> _networkListeners;
	LayersConfig<Dtype>* _layersConfig;

	uint32_t _batchSize;
	uint32_t _epochs;
	uint32_t _testInterval;
	uint32_t _saveInterval;
	uint32_t _iterations;
	uint32_t _stepSize;
	float _baseLearningRate;
	float _momentum;
	float _weightDecay;
	float _clipGradientsLevel;
	float _gamma;

	string _savePathPrefix;

	io_dim _inDim;


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
	NetworkConfig* stepSize(uint32_t stepSize) {
		this->_stepSize = stepSize;
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
	NetworkConfig* gamma(float gamma) {
		this->_gamma = gamma;
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
	NetworkConfig* lrPolicy(LRPolicy lrPolicy) {
		this->_lrPolicy = lrPolicy;
		return this;
	}
	NetworkConfig* inDim(io_dim inDim) {
		this->_inDim = inDim;
		return this;
	}

	void save() {
		// save config
		ofstream configOfs((_savePathPrefix+to_string(_iterations)+".config").c_str(), ios::out | ios::binary);
		_builder->save(configOfs);
		configOfs.close();

		// save learned params
		ofstream paramOfs((_savePathPrefix+to_string(_iterations)+".param").c_str(), ios::out | ios::binary);
		uint32_t numLearnableLayers = _learnableLayers.size();
		for(uint32_t i = 0; i < numLearnableLayers; i++) {
			_learnableLayers[i]->saveParams(paramOfs);
		}
		paramOfs.close();
	}
	void load(const string& end) {
		ifstream ifs((_savePathPrefix+".param").c_str(), ios::in | ios::binary);
		uint32_t numLearnableLayers = _learnableLayers.size();
		for(uint32_t i = 0; i < numLearnableLayers; i++) {
			_learnableLayers[i]->loadParams(ifs);
			if(_learnableLayers[i]->getName() == end) break;
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
	float getLearningRate() {
		float rate;
		switch(_lrPolicy) {
		case Fixed: {
			rate = _baseLearningRate;
		}
			break;
		case Step: {
			uint32_t currentStep = this->_iterations / this->_stepSize;
			rate = _baseLearningRate * pow(_gamma, currentStep);
		}
			break;
		default: {
			cout << "not supported lr policy type ... " << endl;
			exit(1);
		}
		}
		return rate;
	}
};


template class NetworkConfig<float>;



#endif /* NETWORKPARAM_H_ */
