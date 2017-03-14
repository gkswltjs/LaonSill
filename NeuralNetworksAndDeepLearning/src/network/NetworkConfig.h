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
#include "NetworkListener.h"
#include "Worker.h"
#include "Top1Evaluation.h"
#include "Top5Evaluation.h"


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
		// layer의 등록 순서를 보장하기 위해 vector를 사용해야 함.
		std::vector<typename Layer<Dtype>::Builder*> _layerWise;
		std::set<uint32_t> _layerIdSet;

		Builder* layer(typename Layer<Dtype>::Builder* layerBuilder) {
			// id 중복이 없도록 id-layer의 맵에 레이어를 추가한다.
			uint32_t layerIndex = layerBuilder->_id;
			if (this->_layerIdSet.find(layerIndex) != this->_layerIdSet.end()) {
				std::cout << "already contained layer index " << layerIndex << std::endl;
				exit(1);
			} else {
				_layerIdSet.insert(layerBuilder->_id);
				this->_layerWise.push_back(layerBuilder);
			}
			return this;
		}

		LayersConfig<Dtype>* build() {
            std::vector<Layer<Dtype>*> firstLayers;
            std::vector<Layer<Dtype>*> lastLayers;
            std::vector<Layer<Dtype>*> layers;
            std::vector<LearnableLayer<Dtype>*> learnableLayers;
            std::map<uint32_t, Layer<Dtype>*> idLayerMap;

            // (1) 전체 레이어에 대해 Layer Builder의 설정대로 Layer들을 생성한다.
            initializeLayers(firstLayers, lastLayers, layers, learnableLayers, idLayerMap);

			// (2) 레이어 이름으로 레이어 객체를 찾을 수 있도록 이름-레이어 맵을 생성
            std::map<std::string, Layer<Dtype>*> nameLayerMap;
            buildNameLayerMap(layers, nameLayerMap);

            // (3) SplitLayer 추가, SplitData로 업데이트
            insertSplitLayers(layers);

			// (4) 레이어의 입출력 데이터 이름으로부터 유일한 데이터를 생성,
            //    이름-데이터 맵을 생성
			std::map<std::string, Data<Dtype>*> layerDataMap;
			buildLayerData(layers, layerDataMap);



			// 레이어 호출 순서에 따라 순서 재조정
			// 의존성이 해결된 레이어를 앞에 배치하여 순차적으로 호출시 적절한 결과를 보장 함
			// XXX: 내부적으로 layers vector를 변경시켜서 call by value하든지 다른 수단 필요.
			// 이후로 사용하지 않으면 되나 이상해 보임.
			std::vector<Layer<Dtype>*> olayers;
			_orderLayers(layers, olayers);

			printFinalLayerConfiguration(olayers);

			return (new LayersConfig(this))
				->firstLayers(firstLayers)
				->lastLayers(lastLayers)
				->layers(olayers)
				->learnableLayers(learnableLayers)
                ->nameLayerMap(nameLayerMap)
                ->layerDataMap(layerDataMap);
		}

	private:
		void initializeLayers(std::vector<Layer<Dtype>*>& firstLayers,
				std::vector<Layer<Dtype>*>& lastLayers,
        		std::vector<Layer<Dtype>*>& layers,
        		std::vector<LearnableLayer<Dtype>*>& learnableLayers,
        		std::map<uint32_t, Layer<Dtype>*>& idLayerMap) {
			for (int i = 0; i < this->_layerWise.size(); i++) {
				Layer<Dtype>* currentLayer = this->_layerWise[i]->build();

				// 시작 레이어 추가
				InputLayer<Dtype>* inputLayer =
						dynamic_cast<InputLayer<Dtype>*>(currentLayer);
				if (inputLayer) {
					firstLayers.push_back(currentLayer);
				}

				// 끝 레이어 추가
				LossLayer<Dtype>* lossLayer =
						dynamic_cast<LossLayer<Dtype>*>(currentLayer);
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
				idLayerMap[currentLayer->id] = currentLayer;
			}
			assert(firstLayers.size() >= 1 && "no input layer ...");
		}

		void buildNameLayerMap(std::vector<Layer<Dtype>*>& layers,
				std::map<std::string, Layer<Dtype>*>& nameLayerMap) {
			const uint32_t layerSize = layers.size();
			for (uint32_t i = 0; i < layerSize; i++) {
				const std::string& layerName = layers[i]->name;
				if (nameLayerMap.find(layerName) != nameLayerMap.end()) {
					std::cout << "layer name used more than once ... : " <<
						layerName << std::endl;
					exit(1);
				}
				nameLayerMap[layerName] = layers[i];
			}
		}

		void buildLayerData(std::vector<Layer<Dtype>*>& layers,
				std::map<std::string, Data<Dtype>*>& layerDataMap) {
			std::cout << "update layer data to unique objects ... " << std::endl;

			const uint32_t layerSize = layers.size();
			for (uint32_t i = 0; i < layerSize; i++) {
				std::vector<std::string> dataNameVec;
				Layer<Dtype>* layer = layers[i];
				std::cout << "\tfor layer " << layer->getName() << std::endl;
				std::cout << "\tinput data: " << layer->getInputs().size() << std::endl;
				_updateLayerData(layerDataMap, layer->getInputs(), layer->getInputData());
				std::cout << "\toutput data: " << layer->getOutputs().size() << std::endl;
				_updateLayerData(layerDataMap, layer->getOutputs(), layer->getOutputData());
			}
		}

		void printFinalLayerConfiguration(std::vector<Layer<Dtype>*>& olayers) {
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
		}

		void insertSplitLayers(std::vector<Layer<Dtype>*>& layers) {
			std::map<std::string, std::pair<int, int>> outputDataLastOneMap;
			std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> outputDataRefMap;

			int layerSize = layers.size();
			for (int i = 0; i < layerSize; i++) {
				Layer<Dtype>* layer = layers[i];
				for (int j = 0; j < layer->_inputs.size(); j++) {
					if (outputDataLastOneMap.find(layer->_inputs[j]) ==
							outputDataLastOneMap.end()) {
						std::cout << "Undefined input data " << layer->_inputs[j] <<
								" of " << layer->name << std::endl;
						exit(1);
					}

					// input data는 가장 최근에 등장한 동일 이름의 output data를 ref.
					const std::pair<int, int>& lastOne =
							outputDataLastOneMap[layer->_inputs[j]];

					if (outputDataRefMap.find(lastOne) == outputDataRefMap.end()) {
						std::vector<std::pair<int, int>> refList;
						outputDataRefMap[lastOne] = refList;
					}
					outputDataRefMap[lastOne].push_back(std::make_pair(i, j));
				}

				for (int j = 0; j < layer->_outputs.size(); j++) {
					outputDataLastOneMap[layer->_outputs[j]] = std::make_pair(i, j);
				}
			}

			// LOGGING FOR DEBUG ///
			std::cout << "for outputDataLastOneMap: " << std::endl;
			typename std::map<std::string, std::pair<int, int>>::iterator itr1;
			for (itr1 = outputDataLastOneMap.begin(); itr1 != outputDataLastOneMap.end();
					itr1++) {
				std::cout << itr1->first << ": (" << layers[itr1->second.first]->name <<
						", " << layers[itr1->second.first]->_outputs[itr1->second.second] <<
						")"  << std::endl;
			}

			std::cout << "for outputDataRefMap: " << std::endl;
			typename std::map<std::pair<int, int>, std::vector<std::pair<int, int>>>::iterator itr2;
			for (itr2 = outputDataRefMap.begin(); itr2 != outputDataRefMap.end(); itr2++) {
				const std::vector<std::pair<int, int>>& refList = itr2->second;

				std::cout << "(" << layers[itr2->first.first]->name << ", " <<
						layers[itr2->first.first]->_outputs[itr2->first.second] << "): " <<
						refList.size() << std::endl;

				for (int j = 0; j < refList.size(); j++) {
					std::cout << "\t(" << layers[refList[j].first]->name << ", " <<
							layers[refList[j].first]->_inputs[refList[j].second] << ")" <<
							std::endl;
				}
			}
			//////////////////////////////////////////////////

			for (itr2 = outputDataRefMap.begin(); itr2 != outputDataRefMap.end(); itr2++) {
				const std::pair<int, int> key = itr2->first;
				const std::vector<std::pair<int, int>>& value = itr2->second;
				if (value.size() <= 1)
					continue;

				// split layer를 추가할 대상 data
				const std::string& layerName = layers[key.first]->name;
				const std::string& dataName = layers[key.first]->_outputs[key.second];
				const std::string splitLayerName = getSplitLayerName(layerName, dataName,
						key.second);

				std::cout << "splitLayerName: " << splitLayerName << std::endl;

				SplitLayer<Dtype>* splitLayer = new SplitLayer<Dtype>(splitLayerName);
				splitLayer->_inputs.push_back(dataName);

				for (int j = 0; j < value.size(); j++) {
					std::string splitDataName = getSplitDataName(layerName, dataName,
							key.second, j);
					splitLayer->_outputs.push_back(splitDataName);
					std::cout << j << "th SplitLayer Ouput updated with " <<
						splitDataName << std::endl;

					layers[value[j].first]->_inputs[value[j].second] = splitDataName;
				}

				// insert를 위 코드의 중간에서 실행할 경우 index 밀려서 헷갈림
				layers.insert(layers.begin() + key.first + 1, splitLayer);
			}


			//exit(1);


			/*
			//////////////////////////////////////////////////////////////////////////////////
			// 레이어 아이디로 multi branch의 input data, output grad인지 조회하는 맵 생성
			// input으로 사용된 key와 횟수, output으로 사용된 key와 횟수를 맵으로 구성
			std::map<std::string, uint32_t> outputDataCountMap;
			std::map<std::string, uint32_t> inputGradCountMap;
			for (uint32_t i = 0; i < layerSize; i++) {
				Layer<Dtype>* layer = layers[i];
				std::vector<std::string>& inputs = layer->_inputs;
				std::vector<std::string>& outputs = layer->_outputs;

				for (uint32_t j = 0; j < outputs.size(); j++) {
					outputDataCountMap[outputs[j]]++;
				}
				for (uint32_t j = 0; j < inputs.size(); j++) {
					inputGradCountMap[inputs[j]]++;
				}
			}

			// input data와 다르게 output data의 경우 동일한 이름이 일반적으로는 있을 수
			// 없다. input data의 경우 하나의 output을 share하는 input이 있는 경우
			// 해당한다. 예외로 relu와 같이 전달된 input을 그대로 output으로 사용하는
			// 경우이다.
			// 예외가 있어 일단 error에 대해서는 고려하지 않는다.
			typename std::map<std::string, uint32_t>::iterator odcmItr;
			for (odcmItr = outputDataCountMap.begin(); odcmItr != outputDataCountMap.end();
				odcmItr++) {
				if (odcmItr->second > 1) {
					std::cout << "output data multi branch is not allowed ... " << std::endl;
					exit(1);
				}
			}

			typename std::map<std::string, uint32_t>::iterator igcmItr;
			for (igcmItr = inputGradCountMap.begin(); igcmItr != inputGradCountMap.end();
				igcmItr++) {
				// 하나의 output을 여러 input에서 공유하는 경우
				// split layer를 추가하여 하나의 output을 여러 input으로 나눠주고
				// backward때 합쳐주는 역할을 하게 해야 한다.
				if (igcmItr->second > 1) {
					std::cout << "Split Layer for data " << igcmItr->first <<
							" will be added ... " << std::endl;
				}
			}
			*/

			/*
			// 전체 레이어들을 순회, 해당 레이어의 전체 입력 데이터들에 대해
			// 동일 입력 데이터를 복수 위치에서 참조할 경우 해당 입력 데이터에 대해
			// SplitLayer를 추가한다.
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
							std::cout << "SplitLayer for data " << dataName <<
								" has not been created yet ... " << std::endl;
							std::cout << "Creating Split Layer " << splitLayerName <<
								" ... " << std::endl;

							splitLayer = new SplitLayer<Dtype>(splitLayerName);
							dataSplitLayerMap[splitLayerName] = splitLayer;
							splitLayer->_inputs.push_back(dataName);
							splitLayer->_inputData.push_back(inputData);
							layers.push_back(splitLayer);
						} else {
							std::cout << "SplitLayer for data " << dataName <<
								" is already created ... " << std::endl;
							splitLayer = dataSplitLayerMapItr->second;
						}

						// SplitLayer 업데이트
						const uint32_t splitDataIndex = splitLayer->_outputs.size();
						const std::string splitDataName =
							dataName+"-"+std::to_string(splitDataIndex);
						splitLayer->_outputs.push_back(splitDataName);

						std::cout << splitDataIndex << "th SplitLayer Ouput updated with " <<
							splitDataName << std::endl;

						Data<Dtype>* data = new Data<Dtype>(splitDataName, inputData, 0);
						splitLayer->_outputData.push_back(data);
						layerDataMap[splitDataName] = data;

						// Post SplitLayer Layer 업데이트
						inputs[j] = splitDataName;
						layer->_inputData[j] = data;
					}
				}
			}
			*/

		}


		const std::string getSplitLayerName(const std::string& layerName,
				const std::string& dataName, const int dataIdx) {
			std::ostringstream splitLayerName;
			splitLayerName << dataName << "_" << layerName << "_" << dataIdx << "_split";
			return splitLayerName.str();
		}
		const std::string getSplitDataName(const std::string& layerName,
				const std::string& dataName, const int dataIdx, const int splitIdx) {
			std::ostringstream splitBlobName;
			splitBlobName << dataName << "_" << layerName << "_" << dataIdx << "_split_" <<
					splitIdx;
			return splitBlobName.str();
		}



		void _updateLayerData(std::map<std::string, Data<Dtype>*>& dataMap,
				std::vector<std::string>& dataNameVec,
                std::vector<Data<Dtype>*>& layerDataVec) {
			for (uint32_t i = 0; i < dataNameVec.size(); i++) {
				typename std::map<std::string, Data<Dtype>*>::iterator it =
                    dataMap.find(dataNameVec[i]);
				if (it == dataMap.end()) {
					Data<Dtype>* data = new Data<Dtype>(dataNameVec[i]);
					dataMap[dataNameVec[i]] = data;
					layerDataVec.push_back(data);
					std::cout << "\t\tfor data " << dataNameVec[i] <<
                        ": insert new ... " << std::endl;
				} else {
					layerDataVec.push_back(it->second);
					std::cout << "\t\tfor data " << dataNameVec[i] <<
                        ": refer old ... " << std::endl;
				}
			}
		}

		void _orderLayers(std::vector<Layer<Dtype>*>& tempLayers,
            std::vector<Layer<Dtype>*>& layers) {
			std::cout << "ordering layers ... " << std::endl;

			std::set<std::string> dataSet;
			while (tempLayers.size() > 0) {
				bool found = false;
				for (uint32_t i = 0; i < tempLayers.size(); i++) {
					InputLayer<Dtype>* inputLayer =
                        dynamic_cast<InputLayer<Dtype>*>(tempLayers[i]);
					if (inputLayer) {
					// 입력 레이어인 경우,
					//if (tempLayers[i]->getInputsSize() < 1) {
						std::cout << tempLayers[i]->getName() <<
                            " is input layer ... insert ... " << std::endl;
						layers.push_back(tempLayers[i]);
						dataSet.insert(tempLayers[i]->getOutputs().begin(),
                            tempLayers[i]->getOutputs().end());
						tempLayers.erase(tempLayers.begin()+i);
						found = true;
						break;
					} else {
						// 앞선 레이어가 출력한 데이터가 현재 레이어의 모든 입력 데이터를
                        // 커버해야 처리한다.
						if (_isSetContainsAll(dataSet, tempLayers[i]->getInputs())) {
							std::cout << tempLayers[i]->getName() <<
                                "'s all input data have processed ... insert ... " <<
                                std::endl;
							layers.push_back(tempLayers[i]);
							dataSet.insert(tempLayers[i]->getOutputs().begin(),
                                           tempLayers[i]->getOutputs().end());
							tempLayers.erase(tempLayers.begin()+i);
							found = true;
							break;
						} else {
							std::cout << tempLayers[i]->getName() <<
                                "'s not all input data have processed ... skip ... " <<
                                std::endl;
						}
					}
				}
				assert(found && "no input layer or not all layer can find input data ... ");
			}
		}

		bool _isSetContainsAll(std::set<std::string>& dataSet,
            std::vector<std::string>& inputs) {
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
    std::vector<LossLayer<Dtype>*> _lossLayers;
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
            LossLayer<Dtype>* lossLayer = dynamic_cast<LossLayer<Dtype>*>(*iter);
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
	LayersConfig<Dtype>* learnableLayers(
        std::vector<LearnableLayer<Dtype>*>& learnableLayers) {
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

	void save() {
		if (_savePathPrefix == "") return;

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
