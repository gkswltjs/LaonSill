/*
 * Layer.cpp
 *
 *  Created on: 2016. 6. 10.
 *      Author: jhkim
 */

#include "Layer.h"
#include "LayerFactory.h"
#include "../exception/Exception.h"


int Layer::layerCount = 0;

int Layer::generateLayerId() {
	return layerCount++;
}





Layer::Layer(const string name) {
	initialize(name);
}

#ifndef GPU_MODE
Layer::Layer(const string name, int n_in, int n_out) {
	initialize(name, io_dim(n_in, 1, 1), io_dim(n_out, 1, 1));
}

Layer::~Layer() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		if(nextLayers[i].next_layer && nextLayers[i].next_layer->isLastPrevLayerRequest(nextLayers[i].idx)) {
			delete nextLayers[i].next_layer;
			nextLayers[i].next_layer = NULL;
		}
	}
	//cout << "destroying " << name << " layer ... " << endl;
}
#else
Layer::~Layer() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		if(nextLayers[i].next_layer && nextLayers[i].next_layer->isLastPrevLayerRequest(nextLayers[i].idx)) {
			delete nextLayers[i].next_layer;
			nextLayers[i].next_layer = NULL;
		}
	}
	_clearShape();
	//checkCudaErrors(cudaFree(d_output));
	//checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	//checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
	//cout << "destroying " << name << " layer ... " << endl;
}
#endif



void Layer::addPrevLayer(prev_layer_relation prevLayer) {
	prevLayers.push_back(prevLayer);
}

void Layer::addNextLayer(next_layer_relation nextLayer) {
	nextLayers.push_back(nextLayer);
}

bool Layer::isLastPrevLayerRequest(UINT idx) {
	if(prevLayers.size() > idx+1) return false;
	else return true;
}

bool Layer::isLastNextLayerRequest(UINT idx) {
	if(nextLayers.size() > idx+1) return false;
	else return true;
}







Layer* Layer::find(UINT idx, const string name) {
	// 레이어 찾기의 경우 마지막 branch의 요청에 대해서만 처리하면 된다.
	if (!isLastPrevLayerRequest(idx)) return 0;

	// 현재 레이어가 찾는 레이어인 경우
	if (this->name == name) {
		return this;
	}
	// 현재 레이어가 찾는 레이어가 아닌 경우 다음 레이어로 호출 전달
	else {
		for (uint32_t i = 0; i < nextLayers.size(); i++) {
			Layer* result = nextLayers[i].next_layer->find(nextLayers[i].idx, name);
			if(result != 0) return result;
		}
		return 0;
	}
}

void Layer::shape(UINT idx, io_dim in_dim) {
	if (!isLastPrevLayerRequest(idx)) return;

	Util::printMessage(string(name)+"---shape()");
	this->in_dim = in_dim;
	_shape();
	propShape();
}

void Layer::reshape(UINT idx, io_dim in_dim) {
	if (!isLastPrevLayerRequest(idx)) return;

	Util::printMessage(string(name)+"---reshape()");
	this->in_dim = in_dim;
	_reshape();
	propReshape();
}

void Layer::clearShape(UINT idx) {
	if (!isLastPrevLayerRequest(idx)) return;

	Util::printMessage(string(name)+"---clearShape()");
	_clearShape();
	propClearShape();
}

DATATYPE Layer::sumSquareGrad(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) return 0.0;

	DATATYPE result =_sumSquareGrad();
	result += propSumSquareGrad();
	return result;
}

DATATYPE Layer::sumSquareParam(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) return 0.0;

	DATATYPE result =_sumSquareParam();
	result += propSumSquareParam();
	return result;
}

void Layer::scaleParam(UINT idx, DATATYPE scale_factor) {
	if(!isLastPrevLayerRequest(idx)) return;

	_scaleParam(scale_factor);
	propScaleParam(scale_factor);
}

void Layer::save(UINT idx, ofstream &ofs) {
	if(!isLastPrevLayerRequest(idx)) return;

	_save(ofs);
	propSave(ofs);
}

void Layer::saveHeader(UINT idx, ofstream &ofs) {
	if(!isLastPrevLayerRequest(idx)) return;

	Layer *p = this;
	ofs.write((char *)&type, sizeof(int));
	ofs.write((char *)&p, sizeof(Layer *));

	//cout << "save header for " << name << ", type: " << (int)type << ", address: " << p << endl;
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->saveHeader(nextLayers[i].idx, ofs);
	}
}

void Layer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	int layerId;
	char name[LAYER_NAME_LENGTH];
	UINT nextLayerSize, prevLayerSize;

	ifs.read((char *)&layerId, sizeof(int));
	ifs.read(name, LAYER_NAME_LENGTH);
	ifs.read((char *)&in_dim, sizeof(io_dim));
	ifs.read((char *)&out_dim, sizeof(io_dim));
	ifs.read((char *)&nextLayerSize, sizeof(UINT));
	for(UINT i = 0; i < nextLayerSize; i++) {
		next_layer_relation nextLayer;
		ifs.read((char *)&nextLayer, sizeof(next_layer_relation));
		nextLayers.push_back(nextLayer);
	}
	ifs.read((char *)&prevLayerSize, sizeof(UINT));
	for(UINT i = 0; i < prevLayerSize; i++) {
		prev_layer_relation prevLayer;
		ifs.read((char *)&prevLayer, sizeof(prev_layer_relation));
		prevLayers.push_back(prevLayer);
	}
	initialize(name);

	Util::printMessage(string(name)+"---load()");
	Layer::_shape(false);
	updateLayerRelation(layerMap);
}

void Layer::update(UINT idx, UINT n, UINT miniBatchSize) {
	if (!isLastPrevLayerRequest(idx)) return;

	_update(n, miniBatchSize);
	propUpdate(n, miniBatchSize);
}

#ifndef GPU_MODE
void Layer::feedforward(UINT idx, const rcube &input, const char *end=0) {
	propFeedforward(input, end);
}
#else
void Layer::feedforward(UINT idx, const DATATYPE *input, const char *end) {
	if (!isLastPrevLayerRequest(idx)) return;

	_feedforward(input, end);
	propFeedforward(input, end);
}
#endif









#ifndef GPU_MODE
void Layer::initialize(const string name, io_dim in_dim, io_dim out_dim) {
	strcpy(this->name, name);
	//this->name = name;
	this->in_dim = in_dim;
	this->out_dim = out_dim;
	this->input.set_size(in_dim.rows, in_dim.cols, in_dim.channels);
	this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);
}
#else
void Layer::initialize(const string name) {
	this->name = name;
	this->id = Layer::generateLayerId();
}
#endif

void Layer::loadNetwork(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	// fill layer map
	while(true) {
		LayerType layerType;
		Layer *address;

		ifs.read((char *)&layerType, sizeof(int));
		ifs.read((char *)&address, sizeof(Layer *));

		//int loc = ifs.tellg();
		//cout << loc << ", " << (int)layerType << endl;

		if(address == 0) break;
		if(layerType == LayerType::Input) {
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

		if(layer->getType() == LayerType::Input) {
			Layer::load(ifs, layerMap);
		} else {
			layer->load(ifs, layerMap);
		}
		ifs.read((char *)&layerKey, sizeof(Layer *));
	}
}

void Layer::updateLayerRelation(map<Layer *, Layer *> &layerMap) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		Layer *nextLayer = nextLayers[i].next_layer;
		nextLayers[i].next_layer = layerMap.find(nextLayer)->second;
	}

	// 학습된 네트워크를 load하는 경우 backward pass가 없으므로 불필요
	//HiddenLayer *hiddenLayer = dynamic_cast<HiddenLayer *>(this);
	//if(hiddenLayer) {
		for(UINT i = 0; i < prevLayers.size(); i++) {
			Layer *prevLayer = prevLayers[i].prev_layer;
			prevLayers[i].prev_layer = layerMap.find(prevLayer)->second;
		}
	//}
}










void Layer::_shape(bool recursive) {
	char message[256];
	sprintf(message, "%s---_shape():in-%dx%dx%dx%d, out-%dx%dx%dx%d", name.c_str(), in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches,
			out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches);
	Util::printMessage(string(message));

	checkCudaErrors(Util::ucudaMalloc(&this->d_output, sizeof(DATATYPE)*out_dim.batchsize()));		//batch size 고려

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));

	checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols));
	checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			out_dim.batches, out_dim.channels, out_dim.rows, out_dim.cols));
}

void Layer::_reshape() {
	Util::printMessage(string(name)+"---_reshape()");
	// 이전의 input, output 설정과 관련된 memory 정리
	_clearShape();
	_shape();
}

void Layer::_clearShape() {
	Util::printMessage(string(name)+"---_clearShape()");
	checkCudaErrors(cudaFree(d_output));
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));

	d_output = 0;
	inputTensorDesc = 0;
	outputTensorDesc = 0;
}

DATATYPE Layer::_sumSquareGrad() {
	return 0.0;
}

DATATYPE Layer::_sumSquareParam() {
	return 0.0;
}

void Layer::_scaleParam(DATATYPE scale_factor) {
	return;
}

void Layer::_save(ofstream &ofs) {
	Layer *address = this;
	UINT nextLayerSize = nextLayers.size();
	UINT prevLayerSize = prevLayers.size();

	ofs.write((char *)&address, sizeof(Layer *));							// layer address
	ofs.write((char *)&id, sizeof(int));									// layer id
	ofs.write(name.c_str(), LAYER_NAME_LENGTH);								// layer name
	ofs.write((char *)&in_dim, sizeof(io_dim));								// layer in_dim
	ofs.write((char *)&out_dim, sizeof(io_dim));							// layer out_dim
	ofs.write((char *)&nextLayerSize, sizeof(UINT));						// layer next layer size
	for(UINT i = 0; i < nextLayerSize; i++) {								// layer next layers
		ofs.write((char *)&nextLayers[i], sizeof(next_layer_relation));
	}
	ofs.write((char *)&prevLayerSize, sizeof(UINT));						// layer prev layer size
	for(UINT i = 0; i < prevLayers.size(); i++) {							// layer prev layers
		ofs.write((char *)&prevLayers[i], sizeof(prev_layer_relation));
	}
}

void Layer::_update(UINT n, UINT miniBatchSize) {
	return;
}

void Layer::_feedforward(const DATATYPE *input, const char *end) {
	return;
}














void Layer::propShape() {
	Util::printMessage(string(name)+"---propShape()");
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->shape(nextLayers[i].idx, out_dim);
	}
}

void Layer::propReshape() {
	Util::printMessage(string(name)+"---propReshape()");
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->reshape(nextLayers[i].idx, out_dim);
	}
}

void Layer::propClearShape() {
	Util::printMessage(string(name)+"---propClearShape()");
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->clearShape(nextLayers[i].idx);
	}
}

DATATYPE Layer::propSumSquareGrad() {
	Util::printMessage(string(name)+"---propSumSquareGrad()");
	DATATYPE result = 0.0;
	for(UINT i = 0; i < nextLayers.size(); i++) {
		result += nextLayers[i].next_layer->sumSquareGrad(nextLayers[i].idx);
	}
	return result;
}

DATATYPE Layer::propSumSquareParam() {
	Util::printMessage(string(name)+"---propSumSquareParam()");
	DATATYPE result = 0.0;
	for(UINT i = 0; i < nextLayers.size(); i++) {
		result += nextLayers[i].next_layer->sumSquareParam(nextLayers[i].idx);
	}
	return result;
}

void Layer::propScaleParam(DATATYPE scale_factor) {
	Util::printMessage(string(name)+"---propScaleParam()");
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->scaleParam(nextLayers[i].idx, scale_factor);
	}
}

void Layer::propSave(ofstream &ofs) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->save(nextLayers[i].idx, ofs);
	}
}

void Layer::propUpdate(UINT n, UINT miniBatchSize) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->update(nextLayers[i].idx, n, miniBatchSize);
	}
}

#ifndef GPU_MODE
void Layer::propFeedforward(const rcube output, const char *end=0) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->feedforward(nextLayers[i].idx, output, end);
	}
}

void Layer::propResetNParam() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->reset_nabla(nextLayers[i].idx);
	}
}
#else
void Layer::propFeedforward(const DATATYPE *output, const char *end) {
	if(end != 0 && name == end) return;

	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->feedforward(nextLayers[i].idx, output, end);
	}
}
#endif



