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



Layer::Layer(const char *name, int n_in, int n_out) {
	initialize(name, io_dim(n_in, 1, 1), io_dim(n_out, 1, 1));
}

Layer::Layer(const char *name, io_dim in_dim, io_dim out_dim) {
	initialize(name, in_dim, out_dim);
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


void Layer::addPrevLayer(prev_layer_relation prevLayer) { prevLayers.push_back(prevLayer); }
void Layer::addNextLayer(next_layer_relation nextLayer) { nextLayers.push_back(nextLayer); }

/**
 * 주어진 입력 input에 대해 출력 activation을 계산
 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
 */
// sub class에서 구현이 없을 때에만 참조, 구현이 있을 경우 prop*() 함수를 참조
void Layer::feedforward(UINT idx, const rcube &input) { propFeedforward(input); }
void Layer::reset_nabla(UINT idx) { propResetNParam(); }
void Layer::update(UINT idx, UINT n, UINT miniBatchSize) { propUpdate(n, miniBatchSize); }

void Layer::save(UINT idx, ofstream &ofs) {
	if(!isLastPrevLayerRequest(idx)) return;
	save(ofs);
	propSave(ofs);
}


void Layer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	int layerId;
	ifs.read((char *)&layerId, sizeof(int));

	char name[LAYER_NAME_LENGTH];
	ifs.read(name, LAYER_NAME_LENGTH);

	io_dim in_dim;
	ifs.read((char *)&in_dim, sizeof(io_dim));

	io_dim out_dim;
	ifs.read((char *)&out_dim, sizeof(io_dim));

	UINT nextLayerSize;
	ifs.read((char *)&nextLayerSize, sizeof(UINT));

	for(UINT i = 0; i < nextLayerSize; i++) {
		next_layer_relation nextLayer;
		ifs.read((char *)&nextLayer, sizeof(next_layer_relation));
		nextLayers.push_back(nextLayer);
	}

	UINT prevLayerSize;
	ifs.read((char *)&prevLayerSize, sizeof(UINT));

	for(UINT i = 0; i < prevLayerSize; i++) {
		prev_layer_relation prevLayer;
		ifs.read((char *)&prevLayer, sizeof(prev_layer_relation));
		prevLayers.push_back(prevLayer);
	}

	initialize(name, in_dim, out_dim);

	updateLayerRelation(layerMap);
}


void Layer::loadHeader(ofstream &ofs) {

}



void Layer::initialize(const char *name, io_dim in_dim, io_dim out_dim) {
	strcpy(this->name, name);
	//this->name = name;
	this->in_dim = in_dim;
	this->out_dim = out_dim;
	this->input.set_size(in_dim.rows, in_dim.cols, in_dim.channels);
	this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);
}

bool Layer::isLastPrevLayerRequest(UINT idx) {
	//cout << name << " received request from " << idx << "th prev layer ... " << endl;
	if(prevLayers.size() > idx+1) {
		//cout << name << " is not from last prev layer... " << endl;
		return false;
	} else {
		return true;
	}
}

bool Layer::isLastNextLayerRequest(UINT idx) {
	//cout << name << " received request from " << idx << "th next layer ... " << endl;
	if(nextLayers.size() > idx+1) {
		//cout << name << " is not from last next layer... " << endl;
		return false;
	} else {
		return true;
	}
}

void Layer::propFeedforward(const rcube output) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->feedforward(nextLayers[i].idx, output);
	}
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

void Layer::saveHeader(UINT idx, ofstream &ofs) {
	if(!isLastPrevLayerRequest(idx)) return;

	Layer *p = this;
	ofs.write((char *)&type, sizeof(int));
	ofs.write((char *)&p, sizeof(Layer *));

	cout << "save header for " << name << ", type: " << (int)type << ", address: " << p << endl;
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i].next_layer->saveHeader(nextLayers[i].idx, ofs);
	}
}

void Layer::save(ofstream &ofs) {
	/*
	int bodyOffset = ofs.tellp();
	Layer *p = this;
	ofs.seekp(headerOffset);
	ofs.write((char *)&type, sizeof(int));
	headerOffset += sizeof(int);
	ofs.write((char *)&p, sizeof(Layer *));
	headerOffset += sizeof(Layer *);
	ofs.seekp(bodyOffset);
	*/
	Layer *p = this;
	UINT nextLayerSize = nextLayers.size();
	UINT prevLayerSize = prevLayers.size();

	ofs.write((char *)&p, sizeof(Layer *));				// layer address
	ofs.write((char *)&id, sizeof(int));				// layer id
	ofs.write(name, LAYER_NAME_LENGTH);					// layer name
	ofs.write((char *)&in_dim, sizeof(io_dim));			// layer in_dim
	ofs.write((char *)&out_dim, sizeof(io_dim));		// layer out_dim
	ofs.write((char *)&nextLayerSize, sizeof(UINT));	// layer next layer size
	// layer next layers
	for(UINT i = 0; i < nextLayerSize; i++) {
		ofs.write((char *)&nextLayers[i], sizeof(next_layer_relation));
	}
	ofs.write((char *)&prevLayerSize, sizeof(UINT));	// layer prev layer size
	// layer prev layers
	for(UINT i = 0; i < prevLayers.size(); i++) {
		ofs.write((char *)&prevLayers[i], sizeof(prev_layer_relation));
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

void Layer::loadNetwork(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	// fill layer map
	while(true) {
		LayerType layerType;
		ifs.read((char *)&layerType, sizeof(int));
		Layer *address;
		ifs.read((char *)&address, sizeof(Layer *));

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






int Layer::generateLayerId() {
	return layerCount++;
}















