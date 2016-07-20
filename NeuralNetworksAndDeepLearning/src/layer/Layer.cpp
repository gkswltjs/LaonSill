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

Layer::Layer(const char *name) {
	initialize(name);
}

void Layer::addPrevLayer(prev_layer_relation prevLayer) { prevLayers.push_back(prevLayer); }
void Layer::addNextLayer(next_layer_relation nextLayer) { nextLayers.push_back(nextLayer); }

void Layer::reset_nabla(UINT idx) { propResetNParam(); }
void Layer::update(UINT idx, UINT n, UINT miniBatchSize) { propUpdate(n, miniBatchSize); }

void Layer::save(UINT idx, ofstream &ofs) {
	if(!isLastPrevLayerRequest(idx)) return;
	_save(ofs);
	propSave(ofs);
}

void Layer::_save(ofstream &ofs) {
	Layer *address = this;
	UINT nextLayerSize = nextLayers.size();
	UINT prevLayerSize = prevLayers.size();

	ofs.write((char *)&address, sizeof(Layer *));					// layer address
	ofs.write((char *)&id, sizeof(int));									// layer id
	ofs.write(name, LAYER_NAME_LENGTH);										// layer name
	ofs.write((char *)&in_dim, sizeof(io_dim));						// layer in_dim
	ofs.write((char *)&out_dim, sizeof(io_dim));					// layer out_dim
	ofs.write((char *)&nextLayerSize, sizeof(UINT));			// layer next layer size
	for(UINT i = 0; i < nextLayerSize; i++) {							// layer next layers
		ofs.write((char *)&nextLayers[i], sizeof(next_layer_relation));
	}
	ofs.write((char *)&prevLayerSize, sizeof(UINT));			// layer prev layer size
	for(UINT i = 0; i < prevLayers.size(); i++) {					// layer prev layers
		ofs.write((char *)&prevLayers[i], sizeof(prev_layer_relation));
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



bool Layer::isLastPrevLayerRequest(UINT idx) {
	if(prevLayers.size() > idx+1) return false;
	else return true;
}

bool Layer::isLastNextLayerRequest(UINT idx) {
	if(nextLayers.size() > idx+1) return false;
	else return true;
}


void Layer::propResetNParam() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->reset_nabla(nextLayers[i].idx);
	}
}

void Layer::propUpdate(UINT n, UINT miniBatchSize) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->update(nextLayers[i].idx, n, miniBatchSize);
	}
}

void Layer::propSave(ofstream &ofs) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->save(nextLayers[i].idx, ofs);
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


int Layer::generateLayerId() {
	return layerCount++;
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



#if CPU_MODE

Layer::Layer(const char *name, int n_in, int n_out) {
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

/**
 * 주어진 입력 input에 대해 출력 activation을 계산
 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
 */
// sub class에서 구현이 없을 때에만 참조, 구현이 있을 경우 prop*() 함수를 참조
void Layer::feedforward(UINT idx, const rcube &input, const char *end=0) { propFeedforward(input, end); }






void Layer::initialize(const char *name, io_dim in_dim, io_dim out_dim) {
	strcpy(this->name, name);
	//this->name = name;
	this->in_dim = in_dim;
	this->out_dim = out_dim;
	this->input.set_size(in_dim.rows, in_dim.cols, in_dim.channels);
	this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);
}

void Layer::propFeedforward(const rcube output, const char *end=0) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->feedforward(nextLayers[i].idx, output, end);
	}
}










#else


void Layer::shape(UINT idx, io_dim in_dim) {
	Util::printMessage(string(name)+"---shape()");
	this->in_dim = in_dim;
	_shape();
	propShape();
}

void Layer::_shape(bool recursive) {
	Util::setPrint(true);
	Util::printMessage(string(name)+"---_shape()");
	Util::setPrint(false);
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

void Layer::propShape() {
	Util::printMessage(string(name)+"---propShape()");
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->shape(nextLayers[i].idx, out_dim);
	}
}

void Layer::reshape(UINT idx, io_dim in_dim) {
	Util::printMessage(string(name)+"---reshape()");
	this->in_dim = in_dim;
	_reshape();
	propReshape();
}

void Layer::_reshape() {
	Util::printMessage(string(name)+"---_reshape()");
	// 이전의 input, output 설정과 관련된 memory 정리
	_clearShape();
	_shape();
}

void Layer::clearShape(UINT idx) {
	Util::printMessage(string(name)+"---clearShape()");
	_clearShape();
	propClearShape();
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



void Layer::initialize(const char *name) {
	Cuda::refresh();
	strcpy(this->name, name);
	this->id = Layer::generateLayerId();
}


Layer::~Layer() {
	Cuda::refresh();

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



/**
 * 주어진 입력 input에 대해 출력 activation을 계산
 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
 */
// sub class에서 구현이 없을 때에만 참조, 구현이 있을 경우 prop*() 함수를 참조
void Layer::feedforward(UINT idx, const DATATYPE *input, const char *end) { propFeedforward(input, end); }

void Layer::propFeedforward(const DATATYPE *output, const char *end) {
	if(end != 0 && strcmp(name, end) == 0) return;

	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->feedforward(nextLayers[i].idx, output, end);
	}
}


#endif


Layer *Layer::find(UINT idx, const char *name) {
	if(!isLastPrevLayerRequest(idx)) return 0;

	if(strcmp(name, this->name) == 0) {
		return this;
	} else {
		for(UINT i = 0; i < nextLayers.size(); i++) {
			Layer* result = nextLayers[i].next_layer->find(nextLayers[i].idx, name);
			if(result != 0) return result;
		}
	}
	return 0;
}








