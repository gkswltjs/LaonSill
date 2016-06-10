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

InceptionLayer::InceptionLayer(const char *name, int n_in, int n_out,
		int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp)
	: HiddenLayer(name, n_in, n_out) {
	initialize(cv1x1, cv3x3reduce, cv3x3, cv5x5reduce, cv5x5, cp);
}

InceptionLayer::InceptionLayer(const char *name, io_dim in_dim, io_dim out_dim,
		int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp)
	: HiddenLayer(name, in_dim, out_dim) {
	initialize(cv1x1, cv3x3reduce, cv3x3, cv5x5reduce, cv5x5, cp);
}

InceptionLayer::~InceptionLayer() {}



void InceptionLayer::initialize(int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int cp) {
	double weight_lr_mult = 1.0;
	double weight_decay_mult = 1.0;
	double bias_lr_mult = 2.0;
	double bias_decay_mult = 0.0;

	this->type = LayerType::Inception;
	this->id = Layer::generateLayerId();


	//inputLayer = new InputLayer("inputLayer", in_dim);
	//ConvLayer *conv1x1Layer = new ConvLayer("conv1x1", in_dim, filter_dim(1, 1, in_dim.channels, cv1x1, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv1x1)));
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

	//ConvLayer *conv3x3reduceLayer = new ConvLayer("conv3x3reduce", in_dim, filter_dim(1, 1, in_dim.channels, cv3x3reduce, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv3x3reduce)));
	ConvLayer *conv3x3reduceLayer = new ConvLayer(
			"conv3x3reduce",
			in_dim,
			filter_dim(1, 1, in_dim.channels, cv3x3reduce, 1),
			update_param(weight_lr_mult, weight_decay_mult),
			update_param(bias_lr_mult, bias_decay_mult),
			param_filler(ParamFillerType::Xavier, 0.09),
			param_filler(ParamFillerType::Constant, 0.2),
			ActivationType::ReLU);

	//ConvLayer *conv3x3Layer = new ConvLayer("conv3x3", io_dim(in_dim.rows, in_dim.cols, cv3x3reduce), filter_dim(3, 3, cv3x3reduce, cv3x3, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv3x3)));
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

	//ConvLayer *conv5x5recudeLayer = new ConvLayer("conv5x5reduce", in_dim, filter_dim(1, 1, in_dim.channels, cv5x5reduce, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv5x5reduce)));
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

	//ConvLayer *conv5x5Layer = new ConvLayer("conv5x5", io_dim(in_dim.rows, in_dim.cols, cv5x5reduce), filter_dim(5, 5, cv5x5reduce, cv5x5, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv5x5)));
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

	//ConvLayer *convProjectionLayer = new ConvLayer("convProjection", in_dim, filter_dim(1, 1, in_dim.channels, cp, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cp)));
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

	delta_input.set_size(size(output));
	delta_input.zeros();
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








void InceptionLayer::save(UINT idx, ofstream &ofs) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();
	save(ofs);
	propSave(ofs);
}

void InceptionLayer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::load(ifs, layerMap);

	lrn_dim lrn_d;
	ifs.read((char *)&lrn_d, sizeof(lrn_dim));

	//initialize(lrn_d);
}

void InceptionLayer::save(ofstream &ofs) {
	for(UINT i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->save(0, ofs);
	}
	propSave(ofs);
}







































