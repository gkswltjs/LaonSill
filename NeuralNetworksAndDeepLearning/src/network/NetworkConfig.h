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
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "Evaluation.h"
#include "Layer.h"
#include "InputLayer.h"
#include "LearnableLayer.h"
#include "SplitLayer.h"
#include "LossLayer.h"
#include "OutputLayer.h"
#include "LayerFactory.h"
#include "NetworkListener.h"
#include "Worker.h"
#include "Top1Evaluation.h"
#include "Top5Evaluation.h"


//#define OUTPUTLAYER

//template <typename Dtype> class DataSet;
template <typename Dtype> class Worker;


struct WeightsArg {
	std::string weightsPath;
	//std::vector<std::string> weights;
	std::map<std::string, std::string> weightsMap;
};



template <typename Dtype>
class LayersConfig {
public:
	class Builder {
	public:
        std::map<uint32_t, typename Layer<Dtype>::Builder*> _layerWise;


		Builder* layer(typename Layer<Dtype>::Builder* layerBuilder) {
			// id 중복이 없도록 id-layer의 맵에 레이어를 추가한다.
			uint32_t layerIndex = layerBuilder->_id;
			typename std::map<uint32_t, typename Layer<Dtype>::Builder*>::iterator it;
            it = _layerWise.find(layerIndex);
			if (it != _layerWise.end()) {
                std::cout << "already contained layer index " << layerIndex << std::endl;
				exit(1);
			} else {
				_layerWise[layerIndex] = layerBuilder;
			}
			return this;
		}
		LayersConfig<Dtype>* build() {
            std::cout << "LayersConfig::build() ... " << std::endl;

            std::vector<Layer<Dtype>*> firstLayers;
            std::vector<Layer<Dtype>*> lastLayers;
            std::vector<Layer<Dtype>*> layers;
            std::vector<LearnableLayer<Dtype>*> learnableLayers;
            std::map<uint32_t, Layer<Dtype>*> idLayerMap;

            		// (1) 전체 레이어에 대해 Layer Builder의 설정대로 Layer들을 생성한다.
            typename std::map<uint32_t, typename Layer<Dtype>::Builder*>::iterator it;
			for (it = _layerWise.begin(); it != _layerWise.end(); it++) {
                Layer<Dtype>* currentLayer = it->second->build();

                // 시작 레이어 추가
                InputLayer<Dtype>* inputLayer =
                		dynamic_cast<InputLayer<Dtype>*>(currentLayer);
                if (inputLayer) {
                	firstLayers.push_back(currentLayer);
                }

                		// 끝 레이어 추가
#ifndef OUTPUTLAYER
                LossLayer<Dtype>* lossLayer =
                		dynamic_cast<LossLayer<Dtype>*>(currentLayer);
#else
                OutputLayer<Dtype>* lossLayer =
                        dynamic_cast<OutputLayer<Dtype>*>(currentLayer);
#endif
                if (lossLayer) {
                	lastLayers.push_back(currentLayer);
                }

                		// 학습 레이어 추가
                LearnableLayer<Dtype>* learnableLayer =
                    dynamic_cast<LearnableLayer<Dtype>*>(currentLayer);
                if (learnableLayer) {
                    learnableLayers.push_back(learnableLayer);
                }

                		// 일반 레이어 추가
                layers.push_back(currentLayer);
                idLayerMap[it->first] = currentLayer;
			}

			if(firstLayers.size() < 1) {
                std::cout << "no input layer ... " << std::endl;
				exit(1);
			}

			const uint32_t layerSize = layers.size();

			// (2) 레이어 이름으로 레이어 객체를 찾을 수 있도록 이름-레이어 맵을 생성
            std::map<std::string, Layer<Dtype>*> nameLayerMap;
			for (uint32_t i = 0; i < layerSize; i++) {
				const std::string& layerName = layers[i]->getName();
				typename std::map<std::string, Layer<Dtype>*>::iterator it = nameLayerMap.find(layerName);
				if (it != nameLayerMap.end()) {
					std::cout << "layer name used more than once ... : " << layerName << std::endl;
					exit(1);
				}
				nameLayerMap[layerName] = layers[i];
			}

			// 레이어의 입,출력 데이터 이름으로부터 유일한 데이터를 생성, 이름-데이터 맵을 생성
			std::cout << "update layer data to unique objects ... " << std::endl;
			std::map<std::string, Data<Dtype>*> layerDataMap;
			for (uint32_t i = 0; i < layerSize; i++) {
				std::vector<std::string> dataNameVec;
				Layer<Dtype>* layer = layers[i];
				std::cout << "\tfor layer " << layer->getName() << std::endl;
				std::cout << "\tinput data: " << layer->getInputs().size() << std::endl;
				_updateLayerData(layerDataMap, layer->getInputs(), layer->getInputData());
				std::cout << "\toutput data: " << layer->getOutputs().size() << std::endl;
				_updateLayerData(layerDataMap, layer->getOutputs(), layer->getOutputData());
			}

			///////////////////////////////////////////////////////////////////////////////////
			// 레이어 아이디로 multi branch의 input data, output grad인지 조회할 수 있는 맵 생성
			std::map<std::string, uint32_t> outputDataCountMap;
			std::map<std::string, uint32_t> inputGradCountMap;
			for (uint32_t i = 0; i < layerSize; i++) {
				Layer<Dtype>* layer = layers[i];
				std::vector<std::string>& inputs = layer->_inputs;
				std::vector<std::string>& outputs = layer->_outputs;

				for (uint32_t j = 0; j < outputs.size(); j++) {
					outputDataCountMap[outputs[j]] = outputDataCountMap[outputs[j]]+1;
				}
				for (uint32_t j = 0; j < inputs.size(); j++) {
					inputGradCountMap[inputs[j]] = inputGradCountMap[inputs[j]]+1;
				}
			}

			typename std::map<std::string, uint32_t>::iterator odcmItr;
			for (odcmItr = outputDataCountMap.begin(); odcmItr != outputDataCountMap.end(); odcmItr++) {
				if (odcmItr->second > 1) {
					std::cout << "output data multi branch is not allowed ... " << std::endl;
					exit(1);
				}
			}

			typename std::map<std::string, uint32_t>::iterator igcmItr;
			for (igcmItr = inputGradCountMap.begin(); igcmItr != inputGradCountMap.end(); igcmItr++) {
				if (igcmItr->second > 1) {
					std::cout << "Split Layer for data " << igcmItr->first <<
							" will be added ... " << std::endl;
				}
			}


			std::cout << "Adjusting Split Layers ... " << std::endl;
			std::map<std::string, SplitLayer<Dtype>*> dataSplitLayerMap;
			typename std::map<std::string, SplitLayer<Dtype>*>::iterator dataSplitLayerMapItr;
			for (uint32_t i = 0; i < layerSize; i++) {
				Layer<Dtype>* layer = layers[i];
				std::vector<std::string>& inputs = layer->_inputs;
				for (uint32_t j = 0; j < inputs.size(); j++) {
					if (inputGradCountMap[inputs[j]] > 1) {
						std::cout << "input data " << inputs[j] << " of layer " <<
								layer->getName() << " has multi branch " << std::endl;

						const std::string& dataName = inputs[j];
						const std::string splitLayerName = dataName+"-split";
						Data<Dtype>* inputData = layerDataMap[dataName];

						dataSplitLayerMapItr = dataSplitLayerMap.find(splitLayerName);
						SplitLayer<Dtype>* splitLayer = 0;
						if (dataSplitLayerMapItr == dataSplitLayerMap.end()) {
							std::cout << "SplitLayer for data " << dataName << " has not been created yet ... " << std::endl;
							std::cout << "Creating Split Layer " << splitLayerName << " ... " << std::endl;

							splitLayer = new SplitLayer<Dtype>(splitLayerName);
							dataSplitLayerMap[splitLayerName] = splitLayer;
							splitLayer->_inputs.push_back(dataName);
							splitLayer->_inputData.push_back(inputData);
							layers.push_back(splitLayer);
						} else {
							std::cout << "SplitLayer for data " << dataName << " is already created ... " << std::endl;
							splitLayer = dataSplitLayerMapItr->second;
						}

						// SplitLayer 업데이트
						const uint32_t splitDataIndex = splitLayer->_outputs.size();
						const std::string splitDataName = dataName+"-"+std::to_string(splitDataIndex);
						splitLayer->_outputs.push_back(splitDataName);

						std::cout << splitDataIndex << "th SplitLayer Ouput updated with " << splitDataName << std::endl;

						Data<Dtype>* data = new Data<Dtype>(splitDataName, inputData, 0);
						splitLayer->_outputData.push_back(data);
						layerDataMap[splitDataName] = data;

						// Post SplitLayer Layer 업데이트
						inputs[j] = splitDataName;
						layer->_inputData[j] = data;
					}
				}
			}



			// 레이어 호출 순서에 따라 순서 재조정
			// 의존성이 해결된 레이어를 앞에 배치하여 순차적으로 호출시 적절한 결과를 보장하도록 함
			// XXX: 내부적으로 layers vector를 변경시키므로 call by value하든지 다른 수단이 필요.
			// 이후로 사용하지 않으면 되나 이상해 보임.
			std::cout << "ordering layers ... " << std::endl;
			std::vector<Layer<Dtype>*> olayers;
			_orderLayers(layers, olayers);

			std::cout << "final layer configuration: " << std::endl;
			for (uint32_t i = 0; i < olayers.size(); i++) {
				std::cout << i << ": " << olayers[i]->getName() << std::endl;
				std::cout << "\t-inputData: ";
				for (uint32_t j = 0; j < olayers[i]->_inputData.size(); j++) {
					std::cout << olayers[i]->_inputData[j]->_name << ", ";
				}
				std::cout << std::endl;
				std::cout << "\t-outputData: ";
				for (uint32_t j = 0; j < olayers[i]->_outputData.size(); j++) {
					std::cout << olayers[i]->_outputData[j]->_name << ", ";
				}
				std::cout << std::endl;
			}

			return (new LayersConfig(this))
				->firstLayers(firstLayers)
				->lastLayers(lastLayers)
				->layers(olayers)
				->learnableLayers(learnableLayers)
                ->nameLayerMap(nameLayerMap)
                ->layerDataMap(layerDataMap);
		}

		/*
		void save(std::ofstream& ofs) {
			uint32_t numLayers = _layerWise.size();
			ofs.write((char*)&numLayers, sizeof(uint32_t));
            typename std::map<uint32_t, typename Layer<Dtype>::Builder*>::iterator it;
			for (it = _layerWise.begin(); it != _layerWise.end(); it++) {
				it->second->save(ofs);
			}
		}
		void load(std::ifstream& ifs) {
			uint32_t numLayers;
			ifs.read((char*)&numLayers, sizeof(uint32_t));

			for(uint32_t i = 0; i < numLayers; i++) {
				// create layer builder objects from stream
				typename Layer<Dtype>::LayerType layerType;
				ifs.read((char*)&layerType, sizeof(uint32_t));

				typename Layer<Dtype>::Builder* layerBuilder =
                    LayerBuilderFactory<Dtype>::create(layerType);
				layerBuilder->load(ifs);

				// add to layerWise
				layer(layerBuilder);
			}
		}
		*/


	private:
		void _updateLayerData(std::map<std::string, Data<Dtype>*>& dataMap,
				std::vector<std::string>& dataNameVec, std::vector<Data<Dtype>*>& layerDataVec) {
			for (uint32_t i = 0; i < dataNameVec.size(); i++) {
				typename std::map<std::string, Data<Dtype>*>::iterator it = dataMap.find(dataNameVec[i]);
				if (it == dataMap.end()) {
					Data<Dtype>* data = new Data<Dtype>(dataNameVec[i]);
					dataMap[dataNameVec[i]] = data;
					layerDataVec.push_back(data);
					std::cout << "\t\tfor data " << dataNameVec[i] << ": insert new ... " << std::endl;
				} else {
					layerDataVec.push_back(it->second);
					std::cout << "\t\tfor data " << dataNameVec[i] << ": refer old ... " << std::endl;
				}
			}
		}

		void _orderLayers(std::vector<Layer<Dtype>*>& tempLayers, std::vector<Layer<Dtype>*>& layers) {

			std::set<std::string> dataSet;
			while (tempLayers.size() > 0) {
				bool found = false;
				for (uint32_t i = 0; i < tempLayers.size(); i++) {
					InputLayer<Dtype>* inputLayer = dynamic_cast<InputLayer<Dtype>*>(tempLayers[i]);
					if (inputLayer) {
					// 입력 레이어인 경우,
					//if (tempLayers[i]->getInputsSize() < 1) {
						std::cout << tempLayers[i]->getName() << " is input layer ... insert ... " << std::endl;
						layers.push_back(tempLayers[i]);
						dataSet.insert(tempLayers[i]->getOutputs().begin(), tempLayers[i]->getOutputs().end());
						tempLayers.erase(tempLayers.begin()+i);
						found = true;
						break;
					} else {
						// 앞선 레이어가 출력한 데이터가 현재 레이어의 모든 입력 데이터를 커버해야 처리한다.
						if (_isSetContainsAll(dataSet, tempLayers[i]->getInputs())) {
							std::cout << tempLayers[i]->getName() << "'s all input data have processed ... insert ... " << std::endl;
							layers.push_back(tempLayers[i]);
							dataSet.insert(tempLayers[i]->getOutputs().begin(), tempLayers[i]->getOutputs().end());
							tempLayers.erase(tempLayers.begin()+i);
							found = true;
							break;
						} else {
							std::cout << tempLayers[i]->getName() << "'s not all input data have processed ... skip ... " << std::endl;
						}
					}
				}
				if (!found) {
					std::cout << "no input layer or not all layer can find input data ... " << std::endl;
					exit(1);
				}
			}
		}

		bool _isSetContainsAll(std::set<std::string>& dataSet, std::vector<std::string>& inputs) {
			for (uint32_t i = 0; i < inputs.size(); i++) {
				typename std::set<std::string>::iterator it = dataSet.find(inputs[i]);
				// has input not contained in set
				if(it == dataSet.end()) {
					return false;
				}
			}
			return true;
		}
	};

	std::map<std::string, Data<Dtype>*> _layerDataMap;
    std::vector<Layer<Dtype>*> _firstLayers;
    std::vector<Layer<Dtype>*> _lastLayers;
    std::vector<Layer<Dtype>*> _layers;
    std::vector<LearnableLayer<Dtype>*> _learnableLayers;
#ifndef OUTPUTLAYER
    std::vector<LossLayer<Dtype>*> _lossLayers;
#else
    std::vector<OutputLayer<Dtype>*> _lossLayers;
#endif
    InputLayer<Dtype>* _inputLayer;

    std::map<std::string, Layer<Dtype>*> _nameLayerMap;
	Builder* _builder;

    LayersConfig() {}

	LayersConfig(Builder* builder) {
		this->_builder = builder;
	}
	LayersConfig<Dtype>* firstLayers(std::vector<Layer<Dtype>*>& firstLayers) {
		this->_firstLayers = firstLayers;
        this->_inputLayer = dynamic_cast<InputLayer<Dtype>*>(firstLayers[0]);
		return this;
	}
	LayersConfig<Dtype>* lastLayers(std::vector<Layer<Dtype>*>& lastLayers) {
		this->_lastLayers = lastLayers;
        typename std::vector<Layer<Dtype>*>::iterator iter;
        for (iter = lastLayers.begin(); iter != lastLayers.end(); iter++) {
#ifndef OUTPUTLAYER
            LossLayer<Dtype>* lossLayer = dynamic_cast<LossLayer<Dtype>*>(*iter);
#else
        	OutputLayer<Dtype>* lossLayer = dynamic_cast<OutputLayer<Dtype>*>(*iter);
#endif
            if(!lossLayer) {
                std::cout << "invalid output layer ... " << std::endl;
                exit(1);
            }
            _lossLayers.push_back(lossLayer);
        }
		return this;
	}
	LayersConfig<Dtype>* layers(std::vector<Layer<Dtype>*>& layers) {
		this->_layers = layers;
		return this;
	}
	LayersConfig<Dtype>* learnableLayers(std::vector<LearnableLayer<Dtype>*>& learnableLayers) {
		this->_learnableLayers = learnableLayers;
		return this;
	}
	LayersConfig* nameLayerMap(std::map<std::string, Layer<Dtype>*>& nameLayerMap) {
		this->_nameLayerMap = nameLayerMap;
		return this;
	}
	LayersConfig* layerDataMap(std::map<std::string, Data<Dtype>*>& layerDataMap) {
		this->_layerDataMap = layerDataMap;
		return this;
	}
	/*
	LayersConfig* multiBranchDataVec(std::vector<Data<Dtype>*>& multiBranchDataVec) {
		this->_multiBranchDataVec = multiBranchDataVec;
		return this;
	}
	LayersConfig* multiBranchGradVec(std::vector<Data<Dtype>*>& multiBranchGradVec) {
		this->_multiBranchGradVec = multiBranchGradVec;
		return this;
	}
	*/
	/*
	void save(std::ofstream& ofs) {
		this->_builder->save(ofs);
	}
	*/
};


template class LayersConfig<float>;







enum NetworkStatus {
	Train = 0,
	Test = 1
};

enum NetworkPhase {
	TrainPhase = 0,
	TestPhase = 1
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
	//static const std::string savePrefix = "network";
	//const std::string configPostfix;// = ".config";
	//const std::string paramPostfix;// = ".param";

	class Builder {
	public:
		//DataSet<Dtype>* _dataSet;
        //std::vector<Evaluation<Dtype>*> _evaluations;
        std::vector<NetworkListener*> _networkListeners;
        std::vector<LayersConfig<Dtype>*> layersConfigs;

		uint32_t _batchSize;
		uint32_t _dop;	/* degree of parallel */
		uint32_t _epochs;
		uint32_t _testInterval;
		uint32_t _saveInterval;
		uint32_t _stepSize;					// update _baseLearningRate
		float _baseLearningRate;
		float _momentum;
		float _weightDecay;
		float _clipGradientsLevel;
		float _gamma;

        std::string _savePathPrefix;
        std::string _weightsPath;
        std::vector<WeightsArg> _weightsArgs;

		LRPolicy _lrPolicy;
		NetworkPhase _phase;

		std::vector<std::string> _lossLayers;
		//std::vector<std::vector<std::string>> _evaluations;
		//io_dim _inDim;


		Builder() {
			//this->_dataSet = NULL;
			this->_batchSize = 1;
			this->_dop = 1;
			this->_epochs = 1;
			this->_clipGradientsLevel = 35.0f;
			this->_phase = NetworkPhase::TrainPhase;
		}
		/*
		Builder* evaluations(const std::vector<std::vector<std::string>>& evaluations) {
			this->_evaluations = evaluations;
			return this;
		}
		*/
		Builder* networkListeners(const std::vector<NetworkListener*> networkListeners) {
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
		Builder* stepSize(uint32_t stepSize) {
			this->_stepSize = stepSize;
			return this;
		}
		Builder* savePathPrefix(std::string savePathPrefix) {
			this->_savePathPrefix = savePathPrefix;
			return this;
		}
		Builder* weightsPath(std::string weightsPath) {
			this->_weightsPath = weightsPath;
			return this;
		}
		Builder* weightsArgs(std::vector<WeightsArg> weightsArgs) {
			this->_weightsArgs = weightsArgs;
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
		//Builder* dataSet(DataSet<Dtype>* dataSet) {
		//	this->_dataSet = dataSet;
		//	return this;
		//}
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
		Builder* networkPhase(NetworkPhase phase) {
			this->_phase = phase;
			return this;
		}
		Builder* lossLayers(const std::vector<std::string>& lossLayers) {
			this->_lossLayers = lossLayers;
			return this;
		}
		NetworkConfig* build() {

			//load()를 학습단계에서도 사용할 경우 ...
			//테스트단계에서만 사용할 경우 dataSet 필요없음 ...
			//if(_dataSet == NULL) {
			//	std::cout << "dataSet should be set ... " << std::endl;
			//	exit(1);
			//}
			//if (_dataSet) {
			//	_inDim.rows = _dataSet->getRows();
			//	_inDim.cols = _dataSet->getCols();
			//	_inDim.channels = _dataSet->getChannels();
			//}
			//_inDim.batches = _batchSize;

			/*
			std::vector<std::vector<Evaluation<Dtype>*>> evaluations(this->_evaluations.size());
			for (uint32_t i = 0; i < this->_evaluations.size(); i++) {
				evaluations.resize(this->_evaluations[i].size());
				for (uint32_t j = 0; j < this->_evaluations[i].size(); j++) {
					if (this->_evaluations[i][j] == "top1")
						evaluations[i][j] = new Top1Evaluation<Dtype>();
					else if (this->_evaluations[i][j] == "top5")
						evaluations[i][j] = new Top5Evaluation<Dtype>();
					else {
						std::cout << "invalid evaluation type ... " << std::endl;
						exit(-1);
					}
				}
			}
			*/



			NetworkConfig* networkConfig = (new NetworkConfig(this))
					->networkListeners(_networkListeners)
					->batchSize(_batchSize)
					->dop(_dop)
					->epochs(_epochs)
					->testInterval(_testInterval)
					->saveInterval(_saveInterval)
					->stepSize(_stepSize)
					->savePathPrefix(_savePathPrefix)
					->weightsPath(_weightsPath)
					->weightsArgs(_weightsArgs)
					->clipGradientsLevel(_clipGradientsLevel)
					->lrPolicy(_lrPolicy)
					->gamma(_gamma)
					//->dataSet(_dataSet)
					->baseLearningRate(_baseLearningRate)
					->momentum(_momentum)
					->weightDecay(_weightDecay)
					->networkPhase(_phase)
					//->evaluations(evaluations)
					->lossLayers(_lossLayers);
					//->inDim(_inDim);

            networkConfig->layersConfigs.assign(Worker<Dtype>::consumerCount, NULL);

			return networkConfig;
		}



		/*
		void save(std::ofstream& ofs) {
			if(_savePathPrefix == "") {
                std::cout << "save path not specified ... " << std::endl;
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
		void load(const std::string& path) {
            std::ifstream ifs((path+".config").c_str(), std::ios::in | std::ios::binary);

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
		*/


		void print() {
            std::cout << "batchSize: " << _batchSize << std::endl;
            std::cout << "epochs: " << _epochs << std::endl;
            std::cout << "testInterval: " << _testInterval << std::endl;
            std::cout << "saveInterval: " << _saveInterval << std::endl;
            std::cout << "stepSize: " << _stepSize << std::endl;

            std::cout << "baseLearningRate: " << _baseLearningRate << std::endl;
            std::cout << "momentum: " << _momentum << std::endl;
            std::cout << "weightDecay: " << _weightDecay << std::endl;
            std::cout << "clipGradientsLevel: " << _clipGradientsLevel << std::endl;
            std::cout << "gamma: " << _gamma << std::endl;

            std::cout << "savePathPrefix: " << _savePathPrefix << std::endl;
            std::cout << "weightsPath: " << _weightsPath << std::endl;
            std::cout << "lrPolicy: " << _lrPolicy << std::endl;
		}

	};






	NetworkStatus _status;
	LRPolicy _lrPolicy;
	NetworkPhase _phase;
	
	//DataSet<Dtype>* _dataSet;
    //std::vector<std::vector<Evaluation<Dtype>*>> _evaluations;
    std::vector<NetworkListener*> _networkListeners;
    std::vector<LayersConfig<Dtype>*> layersConfigs;

	uint32_t _batchSize;
	uint32_t _dop;
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

    std::string _savePathPrefix;
    std::string _weightsPath;
    std::vector<WeightsArg> _weightsArgs;

    std::vector<std::string> _lossLayers;

	//io_dim _inDim;


	// save & load를 위해서 builder도 일단 저장해 두자.
	Builder* _builder;




	NetworkConfig(Builder* builder) {
		this->_builder = builder;
		this->_iterations = 0;
		this->_rate = -1.0f;
	}
	NetworkConfig* lossLayers(const std::vector<std::string>& lossLayers) {
		this->_lossLayers = lossLayers;
		return this;
	}
	/*
	NetworkConfig* evaluations(const std::vector<std::vector<Evaluation<Dtype>*>>& evaluations) {
		this->_evaluations = evaluations;
		return this;
	}
	*/
	NetworkConfig* networkListeners(const std::vector<NetworkListener*> networkListeners) {
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
	NetworkConfig* stepSize(uint32_t stepSize) {
		this->_stepSize = stepSize;
		return this;
	}
	NetworkConfig* savePathPrefix(std::string savePathPrefix) {
		this->_savePathPrefix = savePathPrefix;
		return this;
	}
	NetworkConfig* weightsPath(const std::string weightsPath) {
		this->_weightsPath = weightsPath;
		return this;
	}
	NetworkConfig* weightsArgs(const std::vector<WeightsArg>& weightsArgs) {
		this->_weightsArgs = weightsArgs;
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
	//NetworkConfig* dataSet(DataSet<Dtype>* dataSet) {
	//	this->_dataSet = dataSet;
	//	return this;
	//}
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
	NetworkConfig* lrPolicy(LRPolicy lrPolicy) {
		this->_lrPolicy = lrPolicy;
		return this;
	}
	NetworkConfig* networkPhase(NetworkPhase phase) {
		this->_phase = phase;
		return this;
	}
	//NetworkConfig* inDim(io_dim inDim) {
	//	this->_inDim = inDim;
	//	return this;
	//}

	void save() {
		if (_savePathPrefix == "") return;

		// save config
		/*
        std::ofstream configOfs(
        		(_savePathPrefix+std::to_string(_iterations)+".config").c_str(),
                std::ios::out | std::ios::binary);
		_builder->save(configOfs);
		configOfs.close();
		*/
		// save learned params
        LayersConfig<Dtype>* firstLayersConfig = this->layersConfigs[0];
        std::ofstream paramOfs(
        		(_savePathPrefix+"/network"+std::to_string(_iterations)+".param").c_str(),
                std::ios::out | std::ios::binary);

		uint32_t numLearnableLayers = firstLayersConfig->_learnableLayers.size();
		//paramOfs.write((char*)&numLearnableLayers, sizeof(uint32_t));

		uint32_t numParams = 0;
		for (uint32_t i = 0; i < numLearnableLayers; i++)
			numParams += firstLayersConfig->_learnableLayers[i]->numParams();

		paramOfs.write((char*)&numParams, sizeof(uint32_t));
		for (uint32_t i = 0; i < numLearnableLayers; i++)
			firstLayersConfig->_learnableLayers[i]->saveParams(paramOfs);

		paramOfs.close();
	}
	void load() {
        std::cout << _savePathPrefix+".param" << std::endl;

        std::ifstream ifs((_savePathPrefix+".param").c_str(), std::ios::in | std::ios::binary);
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

			if (this->_rate < 0.0f || this->_rate != rate) {
				std::cout << "rate updated: " << rate << std::endl;
				this->_rate = rate;
			}
		}
			break;
		default: {
                     std::cout << "not supported lr policy type ... " << std::endl;
			exit(1);
		}
		}
		return rate;
	}

private:
	float _rate;
};


template class NetworkConfig<float>;



#endif /* NETWORKPARAM_H_ */
