/*
 * Layer.cpp
 *
 *  Created on: 2016. 6. 10.
 *      Author: jhkim
 */



#include "Layer.h"
#include "LayerFactory.h"
#include "../exception/Exception.h"
#include "../network/NetworkConfig.h"


Layer::Layer(const string name) {
	initialize(0, name);
}

Layer::Layer(Builder* builder) {
	for(uint32_t i = 0; i < builder->_nextLayerIndices.size(); i++) {
		this->nextLayers.push_back((Layer*)((size_t)builder->_nextLayerIndices[i]));
	}
	initialize(builder->_id, builder->_name);
}

Layer::~Layer() {
	// 다음 레이어들에 대해 소멸을 요청
	// 현재의 레이어가 요청하는 다음 레이어에 대해 마지막 이전 레이어인 경우,
	// 다음 레이어에 대해 소멸을 요청하게 된다. (multi-branch인 경우 복수의 소멸 요청을 피하기 위해)
	for(UINT i = 0; i < nextLayers.size(); i++) {
		if(nextLayers[i] && nextLayers[i]->isLastPrevLayerRequest(id)) {
			delete nextLayers[i];
		}
	}
	nextLayers.clear();

	_clearShape();
}


void Layer::addPrevLayer(Layer* prevLayer) {
	prevLayers.push_back(prevLayer);
}

void Layer::addNextLayer(Layer* nextLayer) {
	nextLayers.push_back(nextLayer);
}

bool Layer::isFirstPrevLayerRequest(UINT idx) {
	if(prevLayers.size() < 1) return true;
	if(prevLayers[0]->getId() != idx) return false;
	else return true;
}

bool Layer::isLastPrevLayerRequest(UINT idx) {
	uint32_t numPrevLayers = prevLayers.size();
	if(numPrevLayers < 1) return true;
	if(prevLayers[numPrevLayers-1]->getId() != idx) return false;
	else return true;
}

bool Layer::isFirstNextLayerRequest(UINT idx) {
	if(nextLayers.size() < 1) return true;
	if(nextLayers[0]->getId() != idx) return false;
	else return true;
}

bool Layer::isLastNextLayerRequest(UINT idx) {
	uint32_t numNextLayers = nextLayers.size();
	if(numNextLayers < 1) return true;
	if(nextLayers[numNextLayers-1]->getId() != idx) return false;
	else return true;
}



bool Layer::w_isLastPrevLayerRequest(UINT idx, const string method) {
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

bool Layer::w_isLastNextLayerRequest(UINT idx, const string method) {
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

void Layer::shape(UINT idx, io_dim in_dim) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::shape()")) return;

	this->in_dim = in_dim;
	_shape();
	propShape();
}

void Layer::reshape(UINT idx, io_dim in_dim) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::reshape()")) return;

	this->in_dim = in_dim;
	_reshape();
	propReshape();
}

void Layer::clearShape(UINT idx) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::clearShape()")) return;

	_clearShape();
	propClearShape();
}

void Layer::save(UINT idx, ofstream &ofs) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::save()")) return;

	_save(ofs);
	propSave(ofs);
}

void Layer::saveHeader(UINT idx, ofstream &ofs) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::saveHeader()")) return;

	Layer *p = this;
	ofs.write((char *)&type, sizeof(int));
	ofs.write((char *)&p, sizeof(Layer *));

	//cout << "save header for " << name << ", type: " << (int)type << ", address: " << p << endl;
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->saveHeader(i, ofs);
	}
}

void Layer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	_load(ifs, layerMap);
}

void Layer::feedforward(UINT idx, Data* input, const char *end) {
	_concat(idx, input);
	if (!w_isLastPrevLayerRequest(idx, "Layer::feedforward()")) return;

	//_scaleInput();
	_feedforward();
	propFeedforward(end);
}

void Layer::initialize(uint32_t id, const string name) {
	this->id = id;
	this->name = name;
	this->_input = new Data();
	this->_output = new Data();
}

void Layer::loadNetwork(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	// fill layer map
	while(true) {
		Layer::Type layerType;
		Layer *address;

		ifs.read((char *)&layerType, sizeof(int));
		ifs.read((char *)&address, sizeof(Layer *));

		//int loc = ifs.tellg();
		//cout << loc << ", " << (int)layerType << endl;

		if(address == 0) break;
		if(layerType == Layer::Input) {
			layerMap.insert(pair<Layer *, Layer *>(address, this));
		}
		else {
			Layer *layer = LayerFactory::create(layerType);
			layerMap.insert(pair<Layer *, Layer *>(address, layer));
			//cout << "created layer type: " << (int)layerType << ", address: " << layer << endl;
		}
	}
	//cout << "map size: " << layerMap.size() << endl;

	Layer *layerKey;
	//ifs.read((char *)&layerKey, sizeof(Layer *));
	//initialize();

	ifs.read((char *)&layerKey, sizeof(Layer *));
	while(ifs && layerKey) {
		Layer *layer = layerMap.find(layerKey)->second;
		if(!layer) throw Exception();

		if(layer->getType() == Layer::Input) {
			Layer::load(ifs, layerMap);
		} else {
			layer->load(ifs, layerMap);
		}
		ifs.read((char *)&layerKey, sizeof(Layer *));
	}
}

void Layer::updateLayerRelation(map<Layer*, Layer*> &layerMap) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i] = layerMap.find((Layer*)nextLayers[i])->second;
	}

	// 학습된 네트워크를 load하는 경우 backward pass가 없으므로 불필요
	//HiddenLayer *hiddenLayer = dynamic_cast<HiddenLayer *>(this);
	//if(hiddenLayer) {
		for(uint32_t i = 0; i < prevLayers.size(); i++) {
			prevLayers[i] = layerMap.find(prevLayers[i])->second;
		}
	//}
}

void Layer::_reshape() {
	// 이전의 input, output 설정과 관련된 memory 정리
	_clearShape();
	_shape();
}

void Layer::_feedforward() {
	//checkCudaErrors(cudaMemcpyAsync(this->d_output, this->d_input, sizeof(DATATYPE)*in_dim.batchsize(), cudaMemcpyDeviceToDevice));
	_output->set_device_data(_input);
}


void Layer::_concat(UINT idx, Data* input) {
	//Util::printDeviceData(input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "input:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	input->print_data("input:");
	_input->print_data("d_input:");

	// 첫번째 branch로부터의 input, 그대로 copy
	if(isFirstPrevLayerRequest(idx)) {
		//checkCudaErrors(cudaMemcpyAsync(d_input, input, sizeof(DATATYPE)*in_dim.batchsize(), cudaMemcpyDeviceToDevice));
		_input->set_device_data(input);
	}
	// 첫번째 이후의 branch로부터의 input, accumulate input
	else {
		//checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(in_dim.batchsize()),
		//		&Cuda::alpha, input, 1, d_input, 1));
		_input->add_device_data(input);
	}
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	_input->print_data("d_input:");
}


void Layer::_scaleInput() {
	if(prevLayers.size() > 1) {
		float branchFactor = 1.0f / prevLayers.size();
		//cout << this->name << "'s feedforward branch factor is " << branchFactor << endl;
		//checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(in_dim.batchsize()), &branchFactor, d_input, 1));
		_input->scale_device_data(branchFactor);
	}
}

void Layer::propShape() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->shape(id, out_dim);
	}
}

void Layer::propReshape() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->reshape(id, out_dim);
	}
}

void Layer::propClearShape() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->clearShape(id);
	}
}

void Layer::propSave(ofstream &ofs) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->save(id, ofs);
	}
}

void Layer::propFeedforward(const char *end) {
	if(end != 0 && name == end) return;

	for(UINT i = 0; i < nextLayers.size(); i++) {
		//_distDataToNext(i, nextLayers[i]);
		nextLayers[i]->feedforward(id, this->getOutput(), end);
	}
}


















#ifndef GPU_MODE
void Layer::_shape(bool recursive) {
	cout << "Layer::_shape() is not implemented in CPU_MODE" << endl;
	exit(1);
}

void Layer::_clearShape() {
	cout << "Layer::_clearShape() is not implemented in CPU_MODE" << endl;
	exit(1);
}

void Layer::_save(ofstream &ofs) {
	cout << "Layer::_save() is not implemented in CPU_MODE" << endl;
	exit(1);
}

void Layer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	cout << "Layer::_load() is not implemented in CPU_MODE" << endl;
	exit(1);
}

#endif



































