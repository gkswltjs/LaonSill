/*
 * InceptionLayer.cpp
 *
 *  Created on: 2016. 5. 27.
 *      Author: jhkim
 */


#ifndef GPU_MODE
#include "InceptionLayer.h"
#include "../exception/Exception.h"
#include "../layer/ConvLayer.h"
#include "../layer/PoolingLayer.h"
#include "../layer/DepthConcatLayer.h"
#include "../activation/ReLU.h"
#include "../activation/Sigmoid.h"
#include "../pooling/MaxPooling.h"
#include "../network/Network.h"


InceptionLayer::InceptionLayer() {
	this->type = Layer::Inception;
}

InceptionLayer::InceptionLayer(Builder* builder)
	: HiddenLayer(builder) {
	initialize(builder->_ic, builder->_oc_cv1x1, builder->_oc_cv3x3reduce, builder->_oc_cv3x3, builder->_oc_cv5x5reduce, builder->_oc_cv5x5,
			builder->_oc_cp, builder->_weightUpdateParam, builder->_biasUpdateParam);
}

InceptionLayer::InceptionLayer(const string name, int ic, int oc_cv1x1, int oc_cv3x3reduce, int oc_cv3x3, int oc_cv5x5reduce, int oc_cv5x5, int oc_cp,
		update_param weight_update_param, update_param bias_update_param)
	: HiddenLayer(name) {
	initialize(ic, oc_cv1x1, oc_cv3x3reduce, oc_cv3x3, oc_cv5x5reduce, oc_cv5x5, oc_cp, weight_update_param, bias_update_param);
}

#ifndef GPU_MODE
InceptionLayer::InceptionLayer(const string name, int n_in, int n_out,
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
#else
InceptionLayer::~InceptionLayer() {
	for(UINT i = 0; i < firstLayers.size(); i++) {
		if(firstLayers[i]) {
			delete firstLayers[i];
			firstLayers[i] = NULL;
		}
	}
	//checkCudaErrors(cudaFree(d_delta_input));
}
#endif


#ifndef GPU_MODE
void InceptionLayer::initialize() {
	this->type = Layer::Inception;

	delta_input.set_size(size(output));
	delta_input.zeros();
}

#else
void InceptionLayer::initialize() {
	this->type = Layer::Inception;
}
#endif

void InceptionLayer::initialize(int ic, int oc_cv1x1, int oc_cv3x3reduce, int oc_cv3x3, int oc_cv5x5reduce, int oc_cv5x5, int oc_cp,
		update_param weight_update_param, update_param bias_update_param) {
	initialize();

	/*
	LayersConfig* layersConfig =
			(new LayersConfig::Builder())
			->layer(0, (new ConvLayer::Builder())
					->name(this->name+"/conv1x1")
					->filterDim(1, 1, ic, oc_cv1x1, 1)
					->weightUpdateParam(1, 1)
					->biasUpdateParam(2, 0)
					->weightFiller(ParamFillerType::Xavier, 0.1)
					->biasFiller(ParamFillerType::Constant, 0.2)
					->activationType(Activation::ReLU)
					->nextLayerIndices({7}))
			->layer(1, (new ConvLayer::Builder())
					->name(this->name+"/conv3x3reduce")
					->filterDim(1, 1, ic, oc_cv3x3reduce, 1)
					->weightUpdateParam(1, 1)
					->biasUpdateParam(2, 0)
					->weightFiller(ParamFillerType::Xavier, 0.1)
					->biasFiller(ParamFillerType::Constant, 0.2)
					->activationType(Activation::ReLU)
					->nextLayerIndices({2}))
			->layer(2, (new ConvLayer::Builder())
					->name(this->name+"/conv3x3")
					->filterDim(3, 3, oc_cv3x3reduce, oc_cv3x3, 1)
					->weightUpdateParam(1, 1)
					->biasUpdateParam(2, 0)
					->weightFiller(ParamFillerType::Xavier, 0.1)
					->biasFiller(ParamFillerType::Constant, 0.2)
					->activationType(Activation::ReLU)
					->prevLayerIndices({1})
					->nextLayerIndices({7}))
			->layer(3, (new ConvLayer::Builder())
					->name(this->name+"/conv5x5reduce")
					->filterDim(3, 3, ic, oc_cv5x5reduce, 1)
					->weightUpdateParam(1, 1)
					->biasUpdateParam(2, 0)
					->weightFiller(ParamFillerType::Xavier, 0.1)
					->biasFiller(ParamFillerType::Constant, 0.2)
					->activationType(Activation::ReLU)
					->nextLayerIndices({4}))
			->layer(4, (new ConvLayer::Builder())
					->name(this->name+"/conv5x5")
					->filterDim(3, 3, oc_cv5x5reduce, oc_cv5x5, 1)
					->weightUpdateParam(1, 1)
					->biasUpdateParam(2, 0)
					->weightFiller(ParamFillerType::Xavier, 0.1)
					->biasFiller(ParamFillerType::Constant, 0.2)
					->activationType(Activation::ReLU)
					->prevLayerIndices({3})
					->nextLayerIndices({7}))
			->layer(5, (new PoolingLayer::Builder())
					->name(this->name+"/pool3x3")
					->poolDim(3, 3, 1)
					->poolingType(Pooling::Max)
					->nextLayerIndices({6}))
			->layer(6, (new ConvLayer::Builder())
					->name(this->name+"/convProjection")
					->filterDim(3, 3, ic, oc_cp, 1)
					->weightUpdateParam(1, 1)
					->biasUpdateParam(2, 0)
					->weightFiller(ParamFillerType::Xavier, 0.1)
					->biasFiller(ParamFillerType::Constant, 0.2)
					->activationType(Activation::ReLU)
					->prevLayerIndices({5})
					->nextLayerIndices({7}))
			->layer(7, (new DepthConcatLayer::Builder())
					->name(this->name+"/dpethConcat")
					->prevLayerIndices({0, 2, 4, 6}))
			->build();
			*/
	LayersConfig *layersConfig = NULL;


	vector<Layer*>& _firstLayers = layersConfig->_firstLayers;
	for(uint32_t i = 0; i < _firstLayers.size(); i++) {
		HiddenLayer* firstLayer = dynamic_cast<HiddenLayer*>(_firstLayers[i]);
		if(!firstLayer) {
			cout << "invalid first layer ... " << endl;
			exit(1);
		}
		this->firstLayers.push_back(firstLayer);
	}
	vector<Layer*>& _lastLayers = layersConfig->_lastLayers;
	if(_lastLayers.size() != 1) {
		cout << "last layer of inception layer should be 1 ... " << endl;
		exit(1);
	}
	HiddenLayer* lastLayer = dynamic_cast<HiddenLayer*>(_lastLayers[0]);
	if(!lastLayer) {
		cout << "invalid last layer ... " << endl;
		exit(1);
	}
	this->lastLayer = lastLayer;
	this->layers = layersConfig->_layers;

	/*
	//double weight_lr_mult = 1.0;
	//double weight_decay_mult = 1.0;
	//double bias_lr_mult = 2.0;
	//double bias_decay_mult = 0.0;

	double bias_const = 0.1;

	ConvLayer *conv1x1Layer = new ConvLayer(
			this->name+"/conv1x1",
			filter_dim(1, 1, ic, oc_cv1x1, 1),
			//update_param(weight_lr_mult, weight_decay_mult),
			//update_param(bias_lr_mult, bias_decay_mult),
			weight_update_param,
			bias_update_param,
			param_filler(ParamFillerType::Xavier, 0.03),
			param_filler(ParamFillerType::Constant, bias_const),
			Activation::ReLU
			);

	ConvLayer *conv3x3reduceLayer = new ConvLayer(
			this->name+"/conv3x3reduce",
			filter_dim(1, 1, ic, oc_cv3x3reduce, 1),
			//update_param(weight_lr_mult, weight_decay_mult),
			//update_param(bias_lr_mult, bias_decay_mult),
			weight_update_param,
			bias_update_param,
			param_filler(ParamFillerType::Xavier, 0.09),
			param_filler(ParamFillerType::Constant, bias_const),
			Activation::ReLU);

	ConvLayer *conv3x3Layer = new ConvLayer(
			this->name+"/conv3x3",
			filter_dim(3, 3, oc_cv3x3reduce, oc_cv3x3, 1),
			//update_param(weight_lr_mult, weight_decay_mult),
			//update_param(bias_lr_mult, bias_decay_mult),
			weight_update_param,
			bias_update_param,
			param_filler(ParamFillerType::Xavier, 0.03),
			param_filler(ParamFillerType::Constant, bias_const),
			Activation::ReLU
			);

	ConvLayer *conv5x5recudeLayer = new ConvLayer(
			this->name+"/conv5x5reduce",
			filter_dim(1, 1, ic, oc_cv5x5reduce, 1),
			//update_param(weight_lr_mult, weight_decay_mult),
			//update_param(bias_lr_mult, bias_decay_mult),
			weight_update_param,
			bias_update_param,
			param_filler(ParamFillerType::Xavier, 0.2),
			param_filler(ParamFillerType::Constant, bias_const),
			Activation::ReLU
			);

	ConvLayer *conv5x5Layer = new ConvLayer(
			this->name+"/conv5x5",
			filter_dim(5, 5, oc_cv5x5reduce, oc_cv5x5, 1),
			//update_param(weight_lr_mult, weight_decay_mult),
			//update_param(bias_lr_mult, bias_decay_mult),
			weight_update_param,
			bias_update_param,
			param_filler(ParamFillerType::Xavier, 0.03),
			param_filler(ParamFillerType::Constant, bias_const),
			Activation::ReLU
			);

	PoolingLayer *pool3x3Layer = new PoolingLayer(
			this->name+"/pool3x3",
			pool_dim(3, 3, 1),
			Pooling::Max
			);

	ConvLayer *convProjectionLayer = new ConvLayer(
			this->name+"/convProjection",
			filter_dim(1, 1, ic, oc_cp, 1),
			//update_param(weight_lr_mult, weight_decay_mult),
			//update_param(bias_lr_mult, bias_decay_mult),
			weight_update_param,
			bias_update_param,
			param_filler(ParamFillerType::Xavier, 0.1),
			param_filler(ParamFillerType::Constant, bias_const),
			Activation::ReLU);

	DepthConcatLayer *depthConcatLayer = new DepthConcatLayer(
			this->name+"/depthConcat"
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
	*/


}


void InceptionLayer::setNetworkConfig(NetworkConfig* networkConfig) {
	HiddenLayer::setNetworkConfig(networkConfig);
	for(uint32_t i = 0; i < layers.size(); i++) {
		layers[i]->setNetworkConfig(networkConfig);
	}
}


Layer* InceptionLayer::_find(const string name) {
	for(UINT i = 0; i < firstLayers.size(); i++) {
		Layer* result = firstLayers[i]->find(0, name);
		if(result) return result;
	}
	return 0;
}

void InceptionLayer::_shape(bool recursive) {
	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->shape(0, in_dim);
	}
	out_dim = lastLayer->getOutDimension();

	if(recursive) {
		HiddenLayer::_shape();
	}

	// TODO HiddenLayer에서 in_dim.batchsize()기준으로 생성...
	// out_dim.batchsize()가 in_dim.batchsize() 대비 더 큰 경우를 따져봐야 한다.
	//int workspaceSize = std::max(in_dim.batchsize(), out_dim.batchsize());
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*workspaceSize));
}

void InceptionLayer::_reshape() {
	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->reshape(0, in_dim);
	}
	HiddenLayer::_reshape();
}

void InceptionLayer::_clearShape() {
	//checkCudaErrors(cudaFree(d_delta_input));

	HiddenLayer::_clearShape();
}

double InceptionLayer::_sumSquareGrad() {
	double result;
	for(UINT i = 0; i < firstLayers.size(); i++) {
		result += firstLayers[i]->sumSquareGrad(0);
	}
	return result;
}

double InceptionLayer::_sumSquareParam() {
	double result;
	for(UINT i = 0; i < firstLayers.size(); i++) {
		result += firstLayers[i]->sumSquareParam(0);
	}
	return result;
}


void InceptionLayer::_scaleParam(DATATYPE scale_factor) {
	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->scaleParam(0, scale_factor);
	}
}

void InceptionLayer::_save(ofstream &ofs) {
	HiddenLayer::_save(ofs);

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
	_saveNinHeader(ofs);
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

void InceptionLayer::_saveNinHeader(ofstream &ofs) {
	//Layer *p = this;
	//ofs.write((char *)&type, sizeof(int));
	//ofs.write((char *)&p, sizeof(Layer *));

	//cout << "save header for " << name << ", type: " << (int)type << ", address: " << p << endl;
	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->saveHeader(0, ofs);
	}
}

void InceptionLayer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::_load(ifs, layerMap);

	UINT firstLayerSize, lastLayerSize;

	ifs.read((char *)&firstLayerSize, sizeof(UINT));
	for(UINT i = 0; i < firstLayerSize; i++) {
		HiddenLayer *firstLayer;
		ifs.read((char *)&firstLayer, sizeof(HiddenLayer *));
		firstLayers.push_back(firstLayer);
	}
	ifs.read((char *)&lastLayerSize, sizeof(UINT));
	ifs.read((char *)&lastLayer, sizeof(HiddenLayer *));

	map<Layer *, Layer *> ninLayerMap;
	loadNetwork(ifs, ninLayerMap);

	for(UINT i = 0; i < firstLayerSize; i++) {
		firstLayers[i] = (HiddenLayer *)ninLayerMap.find(firstLayers[i])->second;
	}
	lastLayer = (HiddenLayer *)ninLayerMap.find(lastLayer)->second;

	initialize();
	InceptionLayer::_shape(false);
}

void InceptionLayer::_update(UINT n, UINT miniBatchSize) {
	Util::printMessage("InceptionLayer::update()---"+string(name));

	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->update(0, n, miniBatchSize);
	}
}

#ifndef GPU_MODE
void InceptionLayer::_feedforward(UINT idx, const rcube &input, const char *end=0) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->feedforward(0, input, end);
	}
	propFeedforward(lastLayer->getOutput(), end);
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

void InceptionLayer::reset_nabla(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->reset_nabla(0);
	}
	propResetNParam();
}
#else
void InceptionLayer::feedforward(UINT idx, const DATATYPE *input, const char *end) {
	//Util::printMessage("InceptionLayer::feedforward()---"+string(name));

	_concat(idx, input);
	if (!isLastPrevLayerRequest(idx)) return;

	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->feedforward(0, this->d_input, end);
	}
	checkCudaErrors(cudaMemcpyAsync(this->d_output, lastLayer->getOutput(), sizeof(DATATYPE)*out_dim.batchsize(), cudaMemcpyDeviceToDevice));

	propFeedforward(end);
}




void InceptionLayer::_backpropagation() {
	lastLayer->backpropagation(0, d_delta_output);

	//checkCudaErrors(cudaMemset(d_delta_input, 0, sizeof(DATATYPE)*in_dim.batchsize()));
	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "delta_input:");

	for(UINT i = 0; i < firstLayers.size(); i++) {
		if (i == 0) {
			checkCudaErrors(cudaMemcpyAsync(d_delta_input, firstLayers[i]->getDeltaInput(), sizeof(DATATYPE)*in_dim.batchsize(), cudaMemcpyDeviceToDevice));
		} else {
			checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(in_dim.batchsize()),
					&Cuda::alpha, firstLayers[i]->getDeltaInput(), 1, d_delta_input, 1));
		}
	}
	Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "delta_input:");
}
#endif


#endif


