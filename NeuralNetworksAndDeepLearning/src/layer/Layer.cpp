/*
 * Layer.cpp
 *
 *  Created on: 2016. 6. 10.
 *      Author: jhkim
 */



/*
#include "Layer.h"
#include "LayerFactory.h"
#include "../exception/Exception.h"
#include "../network/NetworkConfig.h"
*/

#include "Layer.h"

#include <stddef.h>
#include <utility>

#include "../exception/Exception.h"
#include "Layer_device.cu"
#include "LayerFactory.h"
#include "../network/NetworkConfig.h"





template <typename Dtype>
Layer<Dtype>::Layer(const string name) {
	initialize(0, name);
}

template <typename Dtype>
Layer<Dtype>::Layer(Builder* builder) {
	for(uint32_t i = 0; i < builder->_nextLayerIndices.size(); i++) {
		this->nextLayers.push_back((Layer<Dtype>*)((size_t)builder->_nextLayerIndices[i]));
	}
	initialize(builder->_id, builder->_name);
}

template <typename Dtype>
Layer<Dtype>::~Layer() {
	// 다음 레이어들에 대해 소멸을 요청
	// 현재의 레이어가 요청하는 다음 레이어에 대해 마지막 이전 레이어인 경우,
	// 다음 레이어에 대해 소멸을 요청하게 된다. (multi-branch인 경우 복수의 소멸 요청을 피하기 위해)
	for(uint32_t i = 0; i < nextLayers.size(); i++) {
		if(nextLayers[i] && nextLayers[i]->isLastPrevLayerRequest(id)) {
			delete nextLayers[i];
		}
	}
	nextLayers.clear();

	_clearShape();
}

template <typename Dtype>
void Layer<Dtype>::addPrevLayer(Layer<Dtype>* prevLayer) {
	prevLayers.push_back(prevLayer);
}

template <typename Dtype>
void Layer<Dtype>::addNextLayer(Layer<Dtype>* nextLayer) {
	nextLayers.push_back(nextLayer);
}

template <typename Dtype>
bool Layer<Dtype>::isFirstPrevLayerRequest(uint32_t idx) {
	if(prevLayers.size() < 1) return true;
	if(prevLayers[0]->getId() != idx) return false;
	else return true;
}

template <typename Dtype>
bool Layer<Dtype>::isLastPrevLayerRequest(uint32_t idx) {
	uint32_t numPrevLayers = prevLayers.size();
	if(numPrevLayers < 1) return true;
	if(prevLayers[numPrevLayers-1]->getId() != idx) return false;
	else return true;
}

template <typename Dtype>
bool Layer<Dtype>::isFirstNextLayerRequest(uint32_t idx) {
	if(nextLayers.size() < 1) return true;
	if(nextLayers[0]->getId() != idx) return false;
	else return true;
}

template <typename Dtype>
bool Layer<Dtype>::isLastNextLayerRequest(uint32_t idx) {
	uint32_t numNextLayers = nextLayers.size();
	if(numNextLayers < 1) return true;
	if(nextLayers[numNextLayers-1]->getId() != idx) return false;
	else return true;
}


template <typename Dtype>
bool Layer<Dtype>::w_isLastPrevLayerRequest(uint32_t idx, const string method) {
#ifdef PRINT_CALLSTACK
	cout << method << this->name << "-";
#endif
	bool result = isLastPrevLayerRequest(idx);
#ifdef PRINT_CALLSTACK
	if (!result) cout << "skipped ... " << endl;
	else cout << "entered ... " << endl;
#endif
	return result;
}

template <typename Dtype>
bool Layer<Dtype>::w_isLastNextLayerRequest(uint32_t idx, const string method) {
#ifdef PRINT_CALLSTACK
	cout << method << this->name << "-";
#endif
	bool result = isLastNextLayerRequest(idx);
#ifdef PRINT_CALLSTACK
	if (!result) cout << "skipped ... " << endl;
	else cout << "entered ... " << endl;
#endif
	return result;
}

template <typename Dtype>
void Layer<Dtype>::shape(uint32_t idx, io_dim in_dim) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::shape()")) return;

	this->in_dim = in_dim;
	_shape();
	propShape();
}

template <typename Dtype>
void Layer<Dtype>::reshape(uint32_t idx, io_dim in_dim) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::reshape()")) return;

	this->in_dim = in_dim;
	_reshape();
	propReshape();
}

template <typename Dtype>
void Layer<Dtype>::clearShape(uint32_t idx) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::clearShape()")) return;

	_clearShape();
	propClearShape();
}


/*
template <typename Dtype>
void Layer<Dtype>::save(uint32_t idx, ofstream &ofs) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::save()")) return;

	_save(ofs);
	propSave(ofs);
}

template <typename Dtype>
void Layer<Dtype>::saveHeader(uint32_t idx, ofstream &ofs) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::saveHeader()")) return;

	Layer<Dtype>* p = this;
	ofs.write((char *)&type, sizeof(int));
	ofs.write((char *)&p, sizeof(Layer<Dtype>*));

	//cout << "save header for " << name << ", type: " << (int)type << ", address: " << p << endl;
	for(uint32_t i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->saveHeader(i, ofs);
	}
}

template <typename Dtype>
void Layer<Dtype>::load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*> &layerMap) {
	_load(ifs, layerMap);
}
*/

template <typename Dtype>
void Layer<Dtype>::feedforward(uint32_t idx, Data<Dtype>* input, const char *end) {
	_concat(idx, input);
	if (!w_isLastPrevLayerRequest(idx, "Layer::feedforward()")) return;

	//_scaleInput();
	_feedforward();
	propFeedforward(end);
}

template <typename Dtype>
void Layer<Dtype>::initialize(uint32_t id, const string name) {
	this->id = id;
	this->name = name;
	this->_input = new Data<Dtype>();
	this->_output = new Data<Dtype>();
}

/*
template <typename Dtype>
void Layer<Dtype>::loadNetwork(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*> &layerMap) {
	// fill layer map
	while(true) {
		typename Layer<Dtype>::Type layerType;
		Layer<Dtype>* address;

		ifs.read((char *)&layerType, sizeof(int));
		ifs.read((char *)&address, sizeof(Layer<Dtype>*));

		//int loc = ifs.tellg();
		//cout << loc << ", " << (int)layerType << endl;

		if(address == 0) break;
		if(layerType == Layer<Dtype>::Input) {
			layerMap.insert(pair<Layer<Dtype>*, Layer<Dtype>*>(address, this));
		}
		else {
			Layer<Dtype>* layer = LayerFactory<Dtype>::create(layerType);
			layerMap.insert(pair<Layer<Dtype>*, Layer<Dtype>*>(address, layer));
			//cout << "created layer type: " << (int)layerType << ", address: " << layer << endl;
		}
	}
	//cout << "map size: " << layerMap.size() << endl;

	Layer<Dtype>* layerKey;
	//ifs.read((char *)&layerKey, sizeof(Layer<Dtype>*));
	//initialize();

	ifs.read((char *)&layerKey, sizeof(Layer<Dtype>*));
	while(ifs && layerKey) {
		Layer<Dtype>* layer = layerMap.find(layerKey)->second;
		if(!layer) throw Exception();

		if(layer->getType() == Layer<Dtype>::Input) {
			Layer<Dtype>::load(ifs, layerMap);
		} else {
			layer->load(ifs, layerMap);
		}
		ifs.read((char *)&layerKey, sizeof(Layer<Dtype>*));
	}
}
*/

/*
template <typename Dtype>
void Layer<Dtype>::updateLayerRelation(map<Layer<Dtype>*, Layer<Dtype>*> &layerMap) {
	for(uint32_t i = 0; i < nextLayers.size(); i++) {
		nextLayers[i] = layerMap.find((Layer<Dtype>*)nextLayers[i])->second;
	}

	// 학습된 네트워크를 load하는 경우 backward pass가 없으므로 불필요
	//HiddenLayer<Dtype>* hiddenLayer = dynamic_cast<HiddenLayer<Dtype>*>(this);
	//if(hiddenLayer) {
		for(uint32_t i = 0; i < prevLayers.size(); i++) {
			prevLayers[i] = layerMap.find(prevLayers[i])->second;
		}
	//}
}
*/

template <typename Dtype>
void Layer<Dtype>::_reshape() {
	// 이전의 input, output 설정과 관련된 memory 정리
	_clearShape();
	_shape();
}

template <typename Dtype>
void Layer<Dtype>::_feedforward() {
	//_input->print_data("input:");
	_output->set_device_data(_input);
	//_output->print_data("output:");
}

template <typename Dtype>
void Layer<Dtype>::_concat(uint32_t idx, Data<Dtype>* input) {
	input->print_data("param input:");
	//_input->print_data("input:");

	// 첫번째 branch로부터의 input, 그대로 copy
	if(isFirstPrevLayerRequest(idx)) {
		_input->set_device_data(input);
	}
	// 첫번째 이후의 branch로부터의 input, accumulate input
	else {
		_input->add_device_data(input);
	}
	//_input->print_data("input:");
}

template <typename Dtype>
void Layer<Dtype>::_scaleInput() {
	if(prevLayers.size() > 1) {
		float branchFactor = 1.0f / prevLayers.size();
		_input->scale_device_data(branchFactor);
	}
}

template <typename Dtype>
void Layer<Dtype>::propShape() {
	for(uint32_t i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->shape(id, out_dim);
	}
}

template <typename Dtype>
void Layer<Dtype>::propReshape() {
	for(uint32_t i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->reshape(id, out_dim);
	}
}

template <typename Dtype>
void Layer<Dtype>::propClearShape() {
	for(uint32_t i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->clearShape(id);
	}
}

/*
template <typename Dtype>
void Layer<Dtype>::propSave(ofstream &ofs) {
	for(uint32_t i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->save(id, ofs);
	}
}
*/

template <typename Dtype>
void Layer<Dtype>::propFeedforward(const char *end) {
	if(end != 0 && name == end) return;

	for(uint32_t i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->feedforward(id, this->getOutput(), end);
	}
}


















#ifndef GPU_MODE
void Layer<Dtype>::_shape(bool recursive) {
	cout << "Layer::_shape() is not implemented in CPU_MODE" << endl;
	exit(1);
}

void Layer<Dtype>::_clearShape() {
	cout << "Layer::_clearShape() is not implemented in CPU_MODE" << endl;
	exit(1);
}

/*
void Layer<Dtype>::_save(ofstream &ofs) {
	cout << "Layer::_save() is not implemented in CPU_MODE" << endl;
	exit(1);
}

void Layer<Dtype>::_load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*> &layerMap) {
	cout << "Layer::_load() is not implemented in CPU_MODE" << endl;
	exit(1);
}
*/

#endif






template class Layer<float>;







