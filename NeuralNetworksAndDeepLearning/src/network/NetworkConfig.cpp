/**
 * @file NetworkConfig.cpp
 * @date 2017-03-14
 * @author jongheon kim
 * @brief
 * @details
 */


#include "common.h"
#include "NetworkConfig.h"
#include "RoIInputLayer.h"

using namespace std;

template <typename Dtype>
typename LayersConfig<Dtype>::Builder* LayersConfig<Dtype>::Builder::layer(
		typename Layer<Dtype>::Builder* layerBuilder) {
	// id 중복이 없도록 id-layer의 맵에 레이어를 추가한다.
	uint32_t layerIndex = layerBuilder->_id;
	if (this->_layerIdSet.find(layerIndex) != this->_layerIdSet.end()) {
		cout << "already contained layer index " << layerIndex << endl;
		exit(1);
	} else {
		this->_layerIdSet.insert(layerBuilder->_id);
		this->_layerWise.push_back(layerBuilder);
	}
	return this;
}

template <typename Dtype>
LayersConfig<Dtype>* LayersConfig<Dtype>::Builder::build() {
	vector<Layer<Dtype>*> firstLayers;
	vector<Layer<Dtype>*> lastLayers;
	vector<AccuracyLayer<Dtype>*> accuracyLayers;
	vector<Layer<Dtype>*> layers;
	vector<LearnableLayer<Dtype>*> learnableLayers;
	map<uint32_t, Layer<Dtype>*> idLayerMap;
	map<string, uint32_t> nameLayerIdxMap;

	// (1) 전체 레이어에 대해 Layer Builder의 설정대로 Layer들을 생성한다.
	initializeLayers(firstLayers, lastLayers, accuracyLayers, layers, learnableLayers,
			idLayerMap);

	// (2) 레이어 이름으로 레이어 객체를 찾을 수 있도록 이름-레이어 맵을 생성
	map<string, Layer<Dtype>*> nameLayerMap;
	buildNameLayerMap(layers, nameLayerMap);

	// (3) SplitLayer 추가, SplitData로 업데이트
	insertSplitLayers(layers);

	// (4) 레이어의 입출력 데이터 이름으로부터 유일한 데이터를 생성,
	//    이름-데이터 맵을 생성
	map<string, Data<Dtype>*> layerDataMap;
	buildLayerData(layers, layerDataMap);


	// 레이어 호출 순서에 따라 순서 재조정
	// 의존성이 해결된 레이어를 앞에 배치하여 순차적으로 호출시 적절한 결과를 보장 함
	// XXX: 내부적으로 layers vector를 변경시켜서 call by value하든지 다른 수단 필요.
	// 이후로 사용하지 않으면 되나 이상해 보임.
	vector<Layer<Dtype>*> olayers;
	_orderLayers(layers, olayers);


	for (int i = 0; i < olayers.size(); i++) {
		nameLayerIdxMap[olayers[i]->name] = i;
	}


	printFinalLayerConfiguration(olayers);

	return (new LayersConfig(this))
		->firstLayers(firstLayers)
		->lastLayers(lastLayers)
		->accuracyLayers(accuracyLayers)
		->layers(olayers)
		->learnableLayers(learnableLayers)
		->nameLayerMap(nameLayerMap)
		->layerDataMap(layerDataMap)
		->nameLayerIdxMap(nameLayerIdxMap);
}

template <typename Dtype>
void LayersConfig<Dtype>::Builder::initializeLayers(vector<Layer<Dtype>*>& firstLayers,
		vector<Layer<Dtype>*>& lastLayers,
		vector<AccuracyLayer<Dtype>*>& accuracyLayers,
		vector<Layer<Dtype>*>& layers,
		vector<LearnableLayer<Dtype>*>& learnableLayers,
		map<uint32_t, Layer<Dtype>*>& idLayerMap) {

	for (int i = 0; i < this->_layerWise.size(); i++) {
		typename Layer<Dtype>::Builder* builder = this->_layerWise[i];
		Layer<Dtype>* currentLayer = builder->build();

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

		// 정확도 레이어 추가
		AccuracyLayer<Dtype>* accuracyLayer =
				dynamic_cast<AccuracyLayer<Dtype>*>(currentLayer);
		if (accuracyLayer) {
			accuracyLayers.push_back(accuracyLayer);
		}


		// 학습 레이어 추가
		LearnableLayer<Dtype>* learnableLayer =
			dynamic_cast<LearnableLayer<Dtype>*>(currentLayer);
		if (learnableLayer) {
			learnableLayers.push_back(learnableLayer);

			uint32_t donatorID;
			if (builder->_isReceiver) {
				donatorID = builder->_donatorID;
			} else {
				donatorID = currentLayer->id;
			}

			learnableLayer->fillDonatorInfo(builder->_isDonator,
				builder->_isReceiver, donatorID);

			if (learnableLayer->isReceiver) {
				SASSERT(!learnableLayer->isDonator,
					"layer can not be donator and receiver at the same time. "
					"donator ID : %d", learnableLayer->donatorID);
				// FIXME: dangerous casting..
				Donator<Dtype>::receive(learnableLayer->donatorID,
										(void*)currentLayer);
			}

			if (learnableLayer->isDonator) {
				Donator<Dtype>::donate(learnableLayer->donatorID,
									   (void*)currentLayer);
			}
		}

		// 일반 레이어 추가
		layers.push_back(currentLayer);
		idLayerMap[currentLayer->id] = currentLayer;
	}
	assert(firstLayers.size() >= 1 && "no input layer ...");
}

template <typename Dtype>
void LayersConfig<Dtype>::Builder::buildNameLayerMap(vector<Layer<Dtype>*>& layers,
		map<string, Layer<Dtype>*>& nameLayerMap) {
	const uint32_t layerSize = layers.size();
	for (uint32_t i = 0; i < layerSize; i++) {
		const string& layerName = layers[i]->name;
		if (nameLayerMap.find(layerName) != nameLayerMap.end()) {
			cout << "layer name used more than once ... : " <<
				layerName << endl;
			exit(1);
		}
		nameLayerMap[layerName] = layers[i];
	}
}

template <typename Dtype>
void LayersConfig<Dtype>::Builder::buildLayerData(vector<Layer<Dtype>*>& layers,
		map<string, Data<Dtype>*>& layerDataMap) {
	cout << "update layer data to unique objects ... " << endl;

	const uint32_t layerSize = layers.size();
	for (uint32_t i = 0; i < layerSize; i++) {
		Layer<Dtype>* layer = layers[i];
		cout << "for layer: " << layer->name << endl;

		// in-place check
		bool inPlaceCheck = true;
		int lastInPlaceIdx = 0;
		int numInPlaceCheck = min(layer->_inputs.size(), layer->_outputs.size());
		for (; lastInPlaceIdx < numInPlaceCheck; lastInPlaceIdx++) {
			cout << "\tinput: " << layer->_inputs[lastInPlaceIdx] << ", output: " << layer->_outputs[lastInPlaceIdx] << endl;
			if (layer->_inputs[lastInPlaceIdx] != layer->_outputs[lastInPlaceIdx]) {
				inPlaceCheck = false;
				break;
			}
		}
		cout << "\tin-place data is till " << lastInPlaceIdx-1 << endl;


		// 현재 레이어의 input data는 무조건 map에 이미 포함되어 있어야 함.
		for (int j = 0; j < layer->_inputs.size(); j++) {
			assert(layerDataMap.find(layer->_inputs[j]) != layerDataMap.end()
					&& "input data should be in layerDataMap ... ");

			layer->_inputData.push_back(layerDataMap[layer->_inputs[j]]);
			cout << "\tfor input data " << layer->_inputs[j] << ": ref old ... " << endl;
		}

		// 현재 레이어의 output data에 대해 in-place인 경우 이미 map에 포함되어 있어야 함
		// in-place가 아닌 경우 이미 map에 포함되어 있을 경우 error
		SplitLayer<Dtype>* splitLayer = dynamic_cast<SplitLayer<Dtype>*>(layer);
		if (splitLayer) {
			// SplitLayer인 경우 inputData와 outputData가 Data를 share하여 memory를 절약
			for (int j = 0; j < layer->_outputs.size(); j++) {
				const string& dataName = layer->_outputs[j];
				assert(layerDataMap.find(dataName) == layerDataMap.end()
						&& "non in-place output data should not be in layerDataMap ... ");

				Data<Dtype>* data = new Data<Dtype>(dataName, layer->_inputData[0], 0);
				layerDataMap[dataName] = data;
				splitLayer->_outputData.push_back(data);
			}
		} else {
			for (int j = 0; j < layer->_outputs.size(); j++) {
				const string& dataName = layer->_outputs[j];

				// in-place case
				if (j < lastInPlaceIdx) {
					assert(layerDataMap.find(dataName) != layerDataMap.end()
							&& "in-place output data should be in layerDataMap ... ");
					layer->_outputData.push_back(layerDataMap[dataName]);
					cout << "\tfor output data " << dataName << ": ref old ... " << endl;
				} else {
					assert(layerDataMap.find(dataName) == layerDataMap.end()
							&& "non in-place output data should not be in layerDataMap ... ");

					Data<Dtype>* data = new Data<Dtype>(dataName);
					layerDataMap[dataName] = data;
					layer->_outputData.push_back(data);
					cout << "\tfor output data " << dataName << ": insert new ... " << endl;
				}
			}
		}
	}
}

template <typename Dtype>
void LayersConfig<Dtype>::Builder::printFinalLayerConfiguration(
		vector<Layer<Dtype>*>& olayers) {
	cout << "final layer configuration: " << endl;

	for (uint32_t i = 0; i < olayers.size(); i++) {
		cout << i << ": " << olayers[i]->getName() << endl;
		cout << "\t-inputData: ";
		for (uint32_t j = 0; j < olayers[i]->_inputData.size(); j++) {
			cout << olayers[i]->_inputData[j]->_name << ", ";
		}
		cout << endl;
		cout << "\t-outputData: ";
		for (uint32_t j = 0; j < olayers[i]->_outputData.size(); j++) {
			cout << olayers[i]->_outputData[j]->_name << ", ";
		}
		cout << endl;
	}
}

template <typename Dtype>
void LayersConfig<Dtype>::Builder::insertSplitLayers(vector<Layer<Dtype>*>& layers) {
	map<string, pair<int, int>> outputDataLastOneMap;
	map<pair<int, int>, vector<pair<int, int>>> outputDataRefMap;

	int layerSize = layers.size();
	for (int i = 0; i < layerSize; i++) {
		Layer<Dtype>* layer = layers[i];
		for (int j = 0; j < layer->_inputs.size(); j++) {
			if (outputDataLastOneMap.find(layer->_inputs[j]) ==
					outputDataLastOneMap.end()) {
				cout << "Undefined input data " << layer->_inputs[j] <<
						" of " << layer->name << endl;
				exit(1);
			}

			// input data는 가장 최근에 등장한 동일 이름의 output data를 ref.
			const pair<int, int>& lastOne =
					outputDataLastOneMap[layer->_inputs[j]];

			if (outputDataRefMap.find(lastOne) == outputDataRefMap.end()) {
				vector<pair<int, int>> refList;
				outputDataRefMap[lastOne] = refList;
			}
			outputDataRefMap[lastOne].push_back(make_pair(i, j));
		}

		for (int j = 0; j < layer->_outputs.size(); j++) {
			outputDataLastOneMap[layer->_outputs[j]] = make_pair(i, j);
		}
	}

	// LOGGING FOR DEBUG ///
	cout << "for outputDataLastOneMap: " << endl;
	typename map<string, pair<int, int>>::iterator itr1;
	for (itr1 = outputDataLastOneMap.begin(); itr1 != outputDataLastOneMap.end();
			itr1++) {
		cout << itr1->first << ": (" << layers[itr1->second.first]->name <<
				", " << layers[itr1->second.first]->_outputs[itr1->second.second] <<
				")"  << endl;
	}

	cout << "for outputDataRefMap: " << endl;
	typename map<pair<int, int>, vector<pair<int, int>>>::iterator itr2;
	for (itr2 = outputDataRefMap.begin(); itr2 != outputDataRefMap.end(); itr2++) {
		const vector<pair<int, int>>& refList = itr2->second;

		cout << "(" << layers[itr2->first.first]->name << ", " <<
				layers[itr2->first.first]->_outputs[itr2->first.second] << "): " <<
				refList.size() << endl;

		for (int j = 0; j < refList.size(); j++) {
			cout << "\t(" << layers[refList[j].first]->name << ", " <<
					layers[refList[j].first]->_inputs[refList[j].second] << ")" <<
					endl;
		}
	}
	//////////////////////////////////////////////////

	vector<pair<int, Layer<Dtype>*>> locationLayerPairList;
	for (itr2 = outputDataRefMap.begin(); itr2 != outputDataRefMap.end(); itr2++) {
		const pair<int, int> key = itr2->first;
		const vector<pair<int, int>>& value = itr2->second;
		if (value.size() <= 1)
			continue;

		// split layer를 추가할 대상 data
		const string& layerName = layers[key.first]->name;
		const string& dataName = layers[key.first]->_outputs[key.second];
		const string splitLayerName = getSplitLayerName(layerName, dataName,
				key.second);

		cout << "splitLayerName: " << splitLayerName << endl;

		SplitLayer<Dtype>* splitLayer = new SplitLayer<Dtype>(splitLayerName);
		splitLayer->_inputs.push_back(dataName);

		for (int j = 0; j < value.size(); j++) {
			string splitDataName = getSplitDataName(layerName, dataName,
					key.second, j);
			splitLayer->_outputs.push_back(splitDataName);
			cout << j << "th SplitLayer Ouput updated with " <<
				splitDataName << endl;

			layers[value[j].first]->_inputs[value[j].second] = splitDataName;
		}

		// (수정 2) insert를 여기서 할 경우 참조하고 있는 레이어상 index값이 모두 밀림
		//			일단 save만 해두고 나중에 일괄 insert하는 방식으로 수정
		// (수정 1) insert를 위 코드의 중간에서 실행할 경우 index 밀려서 헷갈림
		//layers.insert(layers.begin() + key.first + 1, splitLayer);
		locationLayerPairList.push_back(
				make_pair(key.first + 1 + locationLayerPairList.size(), splitLayer));
	}

	for (int i = 0; i < locationLayerPairList.size(); i++) {
		const pair<int, Layer<Dtype>*>& locationLayerPair = locationLayerPairList[i];
		layers.insert(layers.begin() + locationLayerPair.first, locationLayerPair.second);
	}

	//exit(1);


	/*
	//////////////////////////////////////////////////////////////////////////////////
	// 레이어 아이디로 multi branch의 input data, output grad인지 조회하는 맵 생성
	// input으로 사용된 key와 횟수, output으로 사용된 key와 횟수를 맵으로 구성
	map<string, uint32_t> outputDataCountMap;
	map<string, uint32_t> inputGradCountMap;
	for (uint32_t i = 0; i < layerSize; i++) {
		Layer<Dtype>* layer = layers[i];
		vector<string>& inputs = layer->_inputs;
		vector<string>& outputs = layer->_outputs;

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
	typename map<string, uint32_t>::iterator odcmItr;
	for (odcmItr = outputDataCountMap.begin(); odcmItr != outputDataCountMap.end();
		odcmItr++) {
		if (odcmItr->second > 1) {
			cout << "output data multi branch is not allowed ... " << endl;
			exit(1);
		}
	}

	typename map<string, uint32_t>::iterator igcmItr;
	for (igcmItr = inputGradCountMap.begin(); igcmItr != inputGradCountMap.end();
		igcmItr++) {
		// 하나의 output을 여러 input에서 공유하는 경우
		// split layer를 추가하여 하나의 output을 여러 input으로 나눠주고
		// backward때 합쳐주는 역할을 하게 해야 한다.
		if (igcmItr->second > 1) {
			cout << "Split Layer for data " << igcmItr->first <<
					" will be added ... " << endl;
		}
	}
	*/

	/*
	// 전체 레이어들을 순회, 해당 레이어의 전체 입력 데이터들에 대해
	// 동일 입력 데이터를 복수 위치에서 참조할 경우 해당 입력 데이터에 대해
	// SplitLayer를 추가한다.
	cout << "Adjusting Split Layers ... " << endl;
	map<string, SplitLayer<Dtype>*> dataSplitLayerMap;
	typename map<string, SplitLayer<Dtype>*>::iterator dataSplitLayerMapItr;
	for (uint32_t i = 0; i < layerSize; i++) {
		Layer<Dtype>* layer = layers[i];
		vector<string>& inputs = layer->_inputs;
		for (uint32_t j = 0; j < inputs.size(); j++) {
			if (inputGradCountMap[inputs[j]] > 1) {
				cout << "input data " << inputs[j] << " of layer " <<
						layer->getName() << " has multi branch " << endl;

				const string& dataName = inputs[j];
				const string splitLayerName = dataName+"-split";
				Data<Dtype>* inputData = layerDataMap[dataName];

				dataSplitLayerMapItr = dataSplitLayerMap.find(splitLayerName);
				SplitLayer<Dtype>* splitLayer = 0;
				if (dataSplitLayerMapItr == dataSplitLayerMap.end()) {
					cout << "SplitLayer for data " << dataName <<
						" has not been created yet ... " << endl;
					cout << "Creating Split Layer " << splitLayerName <<
						" ... " << endl;

					splitLayer = new SplitLayer<Dtype>(splitLayerName);
					dataSplitLayerMap[splitLayerName] = splitLayer;
					splitLayer->_inputs.push_back(dataName);
					splitLayer->_inputData.push_back(inputData);
					layers.push_back(splitLayer);
				} else {
					cout << "SplitLayer for data " << dataName <<
						" is already created ... " << endl;
					splitLayer = dataSplitLayerMapItr->second;
				}

				// SplitLayer 업데이트
				const uint32_t splitDataIndex = splitLayer->_outputs.size();
				const string splitDataName =
					dataName+"-"+to_string(splitDataIndex);
				splitLayer->_outputs.push_back(splitDataName);

				cout << splitDataIndex << "th SplitLayer Ouput updated with " <<
					splitDataName << endl;

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

template <typename Dtype>
const string LayersConfig<Dtype>::Builder::getSplitLayerName(const string& layerName,
		const string& dataName, const int dataIdx) {
	ostringstream splitLayerName;
	splitLayerName << dataName << "_" << layerName << "_" << dataIdx << "_split";
	return splitLayerName.str();
}

template <typename Dtype>
const string LayersConfig<Dtype>::Builder::getSplitDataName(const string& layerName,
		const string& dataName, const int dataIdx, const int splitIdx) {
	ostringstream splitBlobName;
	splitBlobName << dataName << "_" << layerName << "_" << dataIdx << "_split_" << splitIdx;
	return splitBlobName.str();
}


template <typename Dtype>
void LayersConfig<Dtype>::Builder::_updateLayerData(map<string, Data<Dtype>*>& dataMap,
		vector<string>& dataNameVec, vector<Data<Dtype>*>& layerDataVec) {
	for (uint32_t i = 0; i < dataNameVec.size(); i++) {
		typename map<string, Data<Dtype>*>::iterator it = dataMap.find(dataNameVec[i]);
		if (it == dataMap.end()) {
			Data<Dtype>* data = new Data<Dtype>(dataNameVec[i]);
			dataMap[dataNameVec[i]] = data;
			layerDataVec.push_back(data);
			cout << "\tfor data " << dataNameVec[i] << ": insert new ... " << endl;
		} else {
			layerDataVec.push_back(it->second);
			cout << "\tfor data " << dataNameVec[i] << ": refer old ... " << endl;
		}
	}
}

template <typename Dtype>
void LayersConfig<Dtype>::Builder::_orderLayers(vector<Layer<Dtype>*>& tempLayers,
	vector<Layer<Dtype>*>& layers) {
	cout << "ordering layers ... " << endl;

	set<string> dataSet;
	while (tempLayers.size() > 0) {
		bool found = false;
		for (uint32_t i = 0; i < tempLayers.size(); i++) {
			InputLayer<Dtype>* inputLayer =
				dynamic_cast<InputLayer<Dtype>*>(tempLayers[i]);
			if (inputLayer) {
			// 입력 레이어인 경우,
			//if (tempLayers[i]->getInputsSize() < 1) {
				cout << tempLayers[i]->getName() << " is input layer ... insert ... " << endl;
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
					cout << tempLayers[i]->getName() <<
							"'s all input data have processed ... insert ... " << endl;
					layers.push_back(tempLayers[i]);
					dataSet.insert(tempLayers[i]->getOutputs().begin(),
								   tempLayers[i]->getOutputs().end());
					tempLayers.erase(tempLayers.begin()+i);
					found = true;
					break;
				} else {
					cout << tempLayers[i]->getName() <<
						"'s not all input data have processed ... skip ... " << endl;
				}
			}
		}
		assert(found && "no input layer or not all layer can find input data ... ");
	}
}

template <typename Dtype>
bool LayersConfig<Dtype>::Builder::_isSetContainsAll(set<string>& dataSet,
	vector<string>& inputs) {
	for (uint32_t i = 0; i < inputs.size(); i++) {
		typename set<string>::iterator it = dataSet.find(inputs[i]);
		// has input not contained in set
		if(it == dataSet.end()) {
			return false;
		}
	}
	return true;
}



template <typename Dtype>
LayersConfig<Dtype>::LayersConfig(Builder* builder) {
	this->_builder = builder;
}

template <typename Dtype>
LayersConfig<Dtype>* LayersConfig<Dtype>::firstLayers(vector<Layer<Dtype>*>& firstLayers) {
	this->_firstLayers = firstLayers;
	this->_inputLayer = dynamic_cast<InputLayer<Dtype>*>(firstLayers[0]);
	return this;
}

template <typename Dtype>
LayersConfig<Dtype>* LayersConfig<Dtype>::lastLayers(vector<Layer<Dtype>*>& lastLayers) {
	this->_lastLayers = lastLayers;
	typename vector<Layer<Dtype>*>::iterator iter;
	for (iter = lastLayers.begin(); iter != lastLayers.end(); iter++) {
		LossLayer<Dtype>* lossLayer = dynamic_cast<LossLayer<Dtype>*>(*iter);
		if (!lossLayer) {
			cout << "invalid output layer ... " << endl;
			exit(1);
		}
		cout << lossLayer->name << " is LossLayer ... " << endl;

		this->_lossLayers.push_back(lossLayer);

		//AccuracyLayer<Dtype>* accuracyLayer = dynamic_cast<AccuracyLayer<Dtype>*>(*iter);
		//if (!accuracy)
	}
	return this;
}

template <typename Dtype>
LayersConfig<Dtype>* LayersConfig<Dtype>::accuracyLayers(vector<AccuracyLayer<Dtype>*>& accuracyLayers) {
	this->_accuracyLayers = accuracyLayers;
	return this;
}

template <typename Dtype>
LayersConfig<Dtype>* LayersConfig<Dtype>::layers(vector<Layer<Dtype>*>& layers) {
	this->_layers = layers;
	return this;
}

template <typename Dtype>
LayersConfig<Dtype>* LayersConfig<Dtype>::learnableLayers(
	vector<LearnableLayer<Dtype>*>& learnableLayers) {
	this->_learnableLayers = learnableLayers;
	return this;
}

template <typename Dtype>
LayersConfig<Dtype>* LayersConfig<Dtype>::nameLayerMap(
		map<string, Layer<Dtype>*>& nameLayerMap) {
	this->_nameLayerMap = nameLayerMap;
	return this;
}

template <typename Dtype>
LayersConfig<Dtype>* LayersConfig<Dtype>::layerDataMap(
		map<string, Data<Dtype>*>& layerDataMap) {
	this->_layerDataMap = layerDataMap;
	return this;
}

template <typename Dtype>
LayersConfig<Dtype>* LayersConfig<Dtype>::nameLayerIdxMap(
		map<string, uint32_t>& nameLayerIdxMap) {
	this->_nameLayerIdxMap = nameLayerIdxMap;
	return this;
}


template class LayersConfig<float>;





template <typename Dtype>
NetworkConfig<Dtype>* NetworkConfig<Dtype>::Builder::build() {
	NetworkConfig<Dtype>* networkConfig = (new NetworkConfig<Dtype>(this))
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
			->optimizer(_optimizer)
			->gamma(_gamma)
			->beta(_beta1, _beta2)
			->epsilon(_epsilon)
			->decayRate(_decayRate)
			->baseLearningRate(_baseLearningRate)
			->momentum(_momentum)
			->weightDecay(_weightDecay)
			->networkPhase(_phase)
			->lossLayers(_lossLayers)
			->accuracyLayers(_accuracyLayers);

	networkConfig->layersConfigs.assign(Worker<Dtype>::consumerCount, NULL);
	return networkConfig;
}

template <typename Dtype>
void NetworkConfig<Dtype>::Builder::print() {
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
	cout << "weightsPath: " << _weightsPath << endl;
	cout << "Optimizer: " << _optimizer << endl;
	cout << "learningRatePolicy: " << _lrPolicy << endl;
}


template <typename Dtype>
void NetworkConfig<Dtype>::save() {
	if (_savePathPrefix == "") return;

	// save learned params
	LayersConfig<Dtype>* firstLayersConfig = this->layersConfigs[0];
	ofstream paramOfs(
			(_savePathPrefix+"/network"+to_string(_iterations)+".param").c_str(),
			ios::out | ios::binary);

	uint32_t numLearnableLayers = firstLayersConfig->_learnableLayers.size();
	//paramOfs.write((char*)&numLearnableLayers, sizeof(uint32_t));



	RoIInputLayer<Dtype>* roiInputLayer = dynamic_cast<RoIInputLayer<Dtype>*>(firstLayersConfig->_inputLayer);

	uint32_t numParams = 0;
	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		numParams += firstLayersConfig->_learnableLayers[i]->numParams();
	}

	paramOfs.write((char*)&numParams, sizeof(uint32_t));
	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		LearnableLayer<Dtype>* learnableLayer = firstLayersConfig->_learnableLayers[i];

		// ONLY FOR FRCNN & BBOX_PRED LAYER!!!
		// frcnn의 경우 bbox_pred 레이어의 params에 대해서만
		// mean, std 적용하여 저장!!!
		if (roiInputLayer && learnableLayer->name == "bbox_pred") {
			//Data<Dtype>::printConfig = 1;
			//SyncMem<Dtype>::printConfig = 1;


			uint32_t numParams = learnableLayer->_params.size();

			vector<vector<float>>& bboxMeans = roiInputLayer->bboxMeans;
			vector<vector<float>>& bboxStds = roiInputLayer->bboxStds;

			//print2dArray("bboxMeans", bboxMeans);
			//print2dArray("bboxStds", bboxStds);

			//cout << "means: " << bboxMeans.size() << ", " << bboxMeans[0].size() << endl;
			//cout << "stds: " << bboxStds.size() << ", " << bboxStds[0].size() << endl;

			Data<Dtype>* param0 = learnableLayer->_params[0];
			Data<Dtype>* orig0 = new Data<Dtype>(param0->_name, true);
			orig0->reshapeLike(param0);

			//param0->print_shape();

			const Dtype* srcPtr0 = param0->host_data();
			Dtype* dstPtr0 = orig0->mutable_host_data();

			const int numRows0 = param0->getShape(2);
			const int numCols0 = param0->getShape(3);
			int index;
			int id1, id2;
			for (int row = 0; row < numRows0; row++) {
				id2 = row / 4;
				id1 = row % 4;
				for (int col = 0; col < numCols0; col++) {
					index = row * numCols0 + col;
					dstPtr0[index] = srcPtr0[index] * bboxStds[id2][id1];
				}
			}

			//param0->print_data({}, false);
			//orig0->print_data({}, false);

			Data<Dtype>* param1 = learnableLayer->_params[1];
			Data<Dtype>* orig1 = new Data<Dtype>(param1->_name, true);
			orig1->reshapeLike(param1);

			//param1->print_shape();

			const Dtype* srcPtr1 = param1->host_data();
			Dtype* dstPtr1 = orig1->mutable_host_data();

			const int numRows1 = param1->getShape(1);
			for (int row = 0; row < numRows1; row++) {
				id2 = row / 4;
				id1 = row % 4;
				dstPtr1[row] = srcPtr1[row] * bboxStds[id2][id1] + bboxMeans[id2][id1];
			}

			//param1->print_data({}, false);
			//orig1->print_data({}, false);

			orig0->save(paramOfs);
			orig1->save(paramOfs);

			delete orig0;
			delete orig1;
			//for(uint32_t i = 0; i < numParams; i++) {
				//_params[i]->save(ofs);
			//}

			//Data<Dtype>::printConfig = 0;
			//SyncMem<Dtype>::printConfig = 0;
		} else
			learnableLayer->saveParams(paramOfs);
	}

	paramOfs.close();
}

template <typename Dtype>
void NetworkConfig<Dtype>::load() {
	cout << _savePathPrefix+".param" << endl;

	ifstream ifs((_savePathPrefix+".param").c_str(), ios::in | ios::binary);
	LayersConfig<Dtype>* firstLayersConfig = this->layersConfigs[0];
	uint32_t numLearnableLayers = firstLayersConfig->_learnableLayers.size();
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		firstLayersConfig->_learnableLayers[i]->loadParams(ifs);
	}
	ifs.close();
}

template <typename Dtype>
bool NetworkConfig<Dtype>::doTest() {
	if(this->_iterations % this->_testInterval == 0) return true;
	else return false;
}

template <typename Dtype>
bool NetworkConfig<Dtype>::doSave() {
	if(this->_iterations % this->_saveInterval == 0) return true;
	else return false;
}

template <typename Dtype>
float NetworkConfig<Dtype>::getLearningRate() {
	float rate;
	switch(this->_lrPolicy) {
	case Fixed: {
		rate = this->_baseLearningRate;
	}
		break;
	case Step: {
		uint32_t currentStep = this->_iterations / this->_stepSize;
		rate = this->_baseLearningRate * pow(this->_gamma, currentStep);

		if (this->_rate < 0.0f || this->_rate != rate) {
			cout << "rate updated: " << rate << endl;
			this->_rate = rate;
		}
	}
		break;
	default: {
		cout << "not supported lr policy type ... " << endl;
		exit(1);
	}
	}
	return rate;
}




template class NetworkConfig<float>;


