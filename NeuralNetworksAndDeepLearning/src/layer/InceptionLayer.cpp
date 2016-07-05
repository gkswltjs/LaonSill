/*
 * InceptionLayer.cpp
 *
 *  Created on: 2016. 5. 27.
 *      Author: jhkim
 */

#include "InceptionLayer.h"
#include "../exception/Exception.h"
#include "../layer/ConvLayer.h"
#include "../layer/PoolingLayer.h"
#include "../layer/DepthConcatLayer.h"
#include "../activation/ReLU.h"
#include "../activation/Sigmoid.h"
#include "../pooling/MaxPooling.h"
#include "../network/Network.h"


InceptionLayer::InceptionLayer(const char *name, io_dim in_dim, io_dim out_dim,
		int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp)
	: HiddenLayer(name, in_dim, out_dim) {
	initialize(cv1x1, cv3x3reduce, cv3x3, cv5x5reduce, cv5x5, cp);
}





void InceptionLayer::save(UINT idx, ofstream &ofs) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	save(ofs);
	propSave(ofs);
}

void InceptionLayer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::load(ifs, layerMap);

	UINT firstLayerSize;
	ifs.read((char *)&firstLayerSize, sizeof(UINT));
	for(UINT i = 0; i < firstLayerSize; i++) {
		HiddenLayer *firstLayer;
		ifs.read((char *)&firstLayer, sizeof(HiddenLayer *));
		firstLayers.push_back(firstLayer);
	}

	UINT lastLayerSize;
	ifs.read((char *)&lastLayerSize, sizeof(UINT));
	ifs.read((char *)&lastLayer, sizeof(HiddenLayer *));

	initialize();

	map<Layer *, Layer *> ninLayerMap;
	loadNetwork(ifs, ninLayerMap);

	for(UINT i = 0; i < firstLayerSize; i++) {
		firstLayers[i] = (HiddenLayer *)ninLayerMap.find(firstLayers[i])->second;
	}
	lastLayer = (HiddenLayer *)ninLayerMap.find(lastLayer)->second;

}

void InceptionLayer::save(ofstream &ofs) {
	HiddenLayer::save(ofs);

	UINT firstLayerSize = firstLayers.size();
	ofs.write((char *)&firstLayerSize, sizeof(UINT));
	// layer next layers
	for(UINT i = 0; i < firstLayerSize; i++) {
		ofs.write((char *)&firstLayers[i], sizeof(Layer *));
	}

	UINT lastLayerSize = 1;
	ofs.write((char *)&lastLayerSize, sizeof(UINT));
	ofs.write((char *)&lastLayer, sizeof(Layer *));


	// InceptionLayer의 NIN (내부 네트워크)를 save ////////
	saveNinHeader(0, ofs);
	// header boundary
	int type = 0;
	Layer *layer = 0;
	ofs.write((char *)&type, sizeof(int));
	ofs.write((char *)&layer, sizeof(Layer *));

	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->save(0, ofs);
	}
	//////////////////////////////////////////////////

	Layer *boundary = 0;
	ofs.write((char *)&boundary, sizeof(Layer *));
}





void InceptionLayer::saveNinHeader(UINT idx, ofstream &ofs) {
	if(!isLastPrevLayerRequest(idx)) return;

	//Layer *p = this;
	//ofs.write((char *)&type, sizeof(int));
	//ofs.write((char *)&p, sizeof(Layer *));

	//cout << "save header for " << name << ", type: " << (int)type << ", address: " << p << endl;
	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->saveHeader(0, ofs);
	}
}

void InceptionLayer::reset_nabla(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->reset_nabla(0);
	}
	propResetNParam();
}

void InceptionLayer::update(UINT idx, UINT n, UINT miniBatchSize) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->update(0, n, miniBatchSize);
	}
	propUpdate(n, miniBatchSize);
}



#if CPU_MODE

InceptionLayer::InceptionLayer(const char *name, int n_in, int n_out,
		int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp)
	: HiddenLayer(name, n_in, n_out) {
	initialize(cv1x1, cv3x3reduce, cv3x3, cv5x5reduce, cv5x5, cp);
}

InceptionLayer::~InceptionLayer() {
	for(UINT i = 0; i < firstLayers.size(); i++) {
		if(firstLayers[i]) {
			delete firstLayers[i];
			firstLayers[i] = NULL;
		}
	}
}

void InceptionLayer::initialize() {
	this->type = LayerType::Inception;
	this->id = Layer::generateLayerId();

	delta_input.set_size(size(output));
	delta_input.zeros();
}

void InceptionLayer::initialize(int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp) {
	initialize();

	double weight_lr_mult = 1.0;
	double weight_decay_mult = 1.0;
	double bias_lr_mult = 2.0;
	double bias_decay_mult = 0.0;

	ConvLayer *conv1x1Layer = new ConvLayer(
			(char *)"conv1x1",
			in_dim,
			filter_dim(1, 1, in_dim.channels, cv1x1, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.03),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU
			);

	ConvLayer *conv3x3reduceLayer = new ConvLayer(
			"conv3x3reduce",
			in_dim,
			filter_dim(1, 1, in_dim.channels, cv3x3reduce, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.09),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU);

	ConvLayer *conv3x3Layer = new ConvLayer(
			"conv3x3",
			io_dim(in_dim.rows, in_dim.cols, cv3x3reduce),
			filter_dim(3, 3, cv3x3reduce, cv3x3, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.03),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU
			);

	ConvLayer *conv5x5recudeLayer = new ConvLayer(
			"conv5x5reduce",
			in_dim,
			filter_dim(1, 1, in_dim.channels, cv5x5reduce, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.2),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU
			);

	ConvLayer *conv5x5Layer = new ConvLayer(
			"conv5x5",
			io_dim(in_dim.rows, in_dim.cols, cv5x5reduce),
			filter_dim(5, 5, cv5x5reduce, cv5x5, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.03),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU
			);

	PoolingLayer *pool3x3Layer = new PoolingLayer(
			"pool3x3",
			in_dim,
			pool_dim(3, 3, 1),
			PoolingType::Max
			);

	ConvLayer *convProjectionLayer = new ConvLayer(
			"convProjection",
			in_dim,
			filter_dim(1, 1, in_dim.channels, cp, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.1),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU);

	DepthConcatLayer *depthConcatLayer = new DepthConcatLayer(
			"depthConcat",
			io_dim(in_dim.rows, in_dim.cols, cv1x1+cv3x3+cv5x5+cp)
			);

	firstLayers.push_back(conv1x1Layer);
	firstLayers.push_back(conv3x3reduceLayer);
	firstLayers.push_back(conv5x5recudeLayer);
	firstLayers.push_back(pool3x3Layer);

	lastLayer = depthConcatLayer;

	Network::addLayerRelation(conv3x3reduceLayer, conv3x3Layer);
	Network::addLayerRelation(conv5x5recudeLayer, conv5x5Layer);
	Network::addLayerRelation(pool3x3Layer, convProjectionLayer);
	Network::addLayerRelation(conv1x1Layer, depthConcatLayer);
	Network::addLayerRelation(conv3x3Layer, depthConcatLayer);
	Network::addLayerRelation(conv5x5Layer, depthConcatLayer);
	Network::addLayerRelation(convProjectionLayer, depthConcatLayer);
}



void InceptionLayer::feedforward(UINT idx, const rcube &input) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->feedforward(0, input);
	}
	propFeedforward(lastLayer->getOutput());
}

void InceptionLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	rcube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);
	Util::printCube(w_next_delta, "w_next_delta:");
	Util::printCube(delta_input, "delta_input:");
	delta_input += w_next_delta;

	if(!isLastNextLayerRequest(idx)) return;
	lastLayer->backpropagation(0, this);

	delta_input.set_size(size(input));
	delta_input.zeros();
	for(UINT i = 0; i < firstLayers.size(); i++) {
		delta_input += firstLayers[i]->getDeltaInput();
	}

	propBackpropagation();

	delta_input.set_size(size(output));
	delta_input.zeros();
}

#else

InceptionLayer::~InceptionLayer() {
	for(UINT i = 0; i < firstLayers.size(); i++) {
		if(firstLayers[i]) {
			delete firstLayers[i];
			firstLayers[i] = NULL;
		}
	}
	checkCudaErrors(cudaFree(d_delta_input));
}

void InceptionLayer::initialize() {
	this->type = LayerType::Inception;
	this->id = Layer::generateLayerId();

	int workspaceSize = std::max(in_dim.batchsize(), out_dim.batchsize());
	checkCudaErrors(cudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*workspaceSize));
}

void InceptionLayer::initialize(int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp) {
	initialize();

	double weight_lr_mult = 1.0;
	double weight_decay_mult = 1.0;
	double bias_lr_mult = 2.0;
	double bias_decay_mult = 0.0;

	ConvLayer *conv1x1Layer = new ConvLayer(
			(char *)"conv1x1",
			in_dim,
			io_dim(in_dim.rows, in_dim.cols, cv1x1, in_dim.batches),
			filter_dim(1, 1, in_dim.channels, cv1x1, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.03),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU
			);

	ConvLayer *conv3x3reduceLayer = new ConvLayer(
			"conv3x3reduce",
			in_dim,
			io_dim(in_dim.rows, in_dim.cols, cv3x3reduce, in_dim.batches),
			filter_dim(1, 1, in_dim.channels, cv3x3reduce, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.09),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU);

	ConvLayer *conv3x3Layer = new ConvLayer(
			"conv3x3",
			io_dim(in_dim.rows, in_dim.cols, cv3x3reduce, in_dim.batches),
			io_dim(in_dim.rows, in_dim.cols, cv3x3, in_dim.batches),
			filter_dim(3, 3, cv3x3reduce, cv3x3, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.03),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU
			);

	ConvLayer *conv5x5recudeLayer = new ConvLayer(
			"conv5x5reduce",
			in_dim,
			io_dim(in_dim.rows, in_dim.cols, cv5x5reduce, in_dim.batches),
			filter_dim(1, 1, in_dim.channels, cv5x5reduce, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.2),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU
			);

	ConvLayer *conv5x5Layer = new ConvLayer(
			"conv5x5",
			io_dim(in_dim.rows, in_dim.cols, cv5x5reduce, in_dim.batches),
			io_dim(in_dim.rows, in_dim.cols, cv5x5, in_dim.batches),
			filter_dim(5, 5, cv5x5reduce, cv5x5, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.03),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU
			);

	PoolingLayer *pool3x3Layer = new PoolingLayer(
			"pool3x3",
			in_dim,
			in_dim,
			pool_dim(3, 3, 1),
			PoolingType::Max
			);

	ConvLayer *convProjectionLayer = new ConvLayer(
			"convProjection",
			in_dim,
			io_dim(in_dim.rows, in_dim.cols, cp, in_dim.batches),
			filter_dim(1, 1, in_dim.channels, cp, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.1),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU);

	DepthConcatLayer *depthConcatLayer = new DepthConcatLayer(
			"depthConcat",
			io_dim(in_dim.rows, in_dim.cols, cv1x1+cv3x3+cv5x5+cp, in_dim.batches)
			);

	firstLayers.push_back(conv1x1Layer);
	firstLayers.push_back(conv3x3reduceLayer);
	firstLayers.push_back(conv5x5recudeLayer);
	firstLayers.push_back(pool3x3Layer);

	lastLayer = depthConcatLayer;

	Network::addLayerRelation(conv3x3reduceLayer, conv3x3Layer);
	Network::addLayerRelation(conv5x5recudeLayer, conv5x5Layer);
	Network::addLayerRelation(pool3x3Layer, convProjectionLayer);
	Network::addLayerRelation(conv1x1Layer, depthConcatLayer);
	Network::addLayerRelation(conv3x3Layer, depthConcatLayer);
	Network::addLayerRelation(conv5x5Layer, depthConcatLayer);
	Network::addLayerRelation(convProjectionLayer, depthConcatLayer);
}



void InceptionLayer::feedforward(UINT idx, const DATATYPE *input) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();
	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->feedforward(0, input);
	}
	propFeedforward(lastLayer->getOutput());
}

void InceptionLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	Util::printMessage("InceptionLayer::backpropagation()---");
	if(idx == 0) {
		checkCudaErrors(cudaMemset(d_delta_input, 0, sizeof(DATATYPE)*out_dim.batchsize()));
	}
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(out_dim.batchsize()),
			&alpha, next_layer->getDeltaInput(), 1, d_delta_input, 1));

	if(!isLastNextLayerRequest(idx)) return;
	lastLayer->backpropagation(0, this);

	checkCudaErrors(cudaMemset(d_delta_input, 0, sizeof(DATATYPE)*in_dim.batchsize()));
	for(UINT i = 0; i < firstLayers.size(); i++) {
		checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(in_dim.batchsize()),
					&alpha, firstLayers[i]->getDeltaInput(), 1, d_delta_input, 1));
	}

	propBackpropagation();

	/*
	rcube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);
	Util::printCube(w_next_delta, "w_next_delta:");
	Util::printCube(delta_input, "delta_input:");
	delta_input += w_next_delta;

	if(!isLastNextLayerRequest(idx)) return;
	lastLayer->backpropagation(0, this);

	delta_input.set_size(size(input));
	delta_input.zeros();
	for(UINT i = 0; i < firstLayers.size(); i++) {
		delta_input += firstLayers[i]->getDeltaInput();
	}

	propBackpropagation();

	delta_input.set_size(size(output));
	delta_input.zeros();
	*/
}





#endif





























