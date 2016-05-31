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
#include "../Network.h"

InceptionLayer::InceptionLayer(string name, int n_in, int n_out,
		int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int p3x3, int cp)
	: HiddenLayer(name, n_in, n_out) {
	initialize(cv1x1, cv3x3reduce, cv3x3, cv5x5reduce, cv5x5, p3x3, cp);
}

InceptionLayer::InceptionLayer(string name, io_dim in_dim, io_dim out_dim,
		int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int p3x3, int cp)
	: HiddenLayer(name, in_dim, out_dim) {
	initialize(cv1x1, cv3x3reduce, cv3x3, cv5x5reduce, cv5x5, p3x3, cp);
}

InceptionLayer::~InceptionLayer() {}



void InceptionLayer::initialize(int cv1x1, int cv3x3reduce, int cv3x3, int cv5x5reduce, int cv5x5, int p3x3, int cp) {

	//inputLayer = new InputLayer("inputLayer", in_dim);
	ConvLayer *conv1x1Layer = new ConvLayer("conv1x1", in_dim, filter_dim(1, 1, in_dim.channels, cv1x1, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv1x1)));
	ConvLayer *conv3x3reduceLayer = new ConvLayer("conv3x3reduce", in_dim, filter_dim(1, 1, in_dim.channels, cv3x3reduce, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv3x3reduce)));
	ConvLayer *conv3x3Layer = new ConvLayer("conv3x3", io_dim(in_dim.rows, in_dim.cols, cv3x3reduce), filter_dim(3, 3, cv3x3reduce, cv3x3, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv3x3)));
	ConvLayer *conv5x5recudeLayer = new ConvLayer("conv5x5reduce", in_dim, filter_dim(1, 1, in_dim.channels, cv5x5reduce, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv5x5reduce)));
	ConvLayer *conv5x5Layer = new ConvLayer("conv5x5", io_dim(in_dim.rows, in_dim.cols, cv5x5reduce), filter_dim(5, 5, cv5x5reduce, cv5x5, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cv5x5)));
	PoolingLayer *pool3x3Layer = new PoolingLayer("pool3x3", in_dim, pool_dim(3, 3, 1), new MaxPooling());
	ConvLayer *convProjectionLayer = new ConvLayer("convProjection", in_dim, filter_dim(1, 1, in_dim.channels, cp, 1), new ReLU(io_dim(in_dim.rows, in_dim.cols, cp)));
	DepthConcatLayer *depthConcatLayer = new DepthConcatLayer("depthConcat", io_dim(in_dim.rows, in_dim.cols, cv1x1+cv3x3+cv5x5+cp));

	/*
	ConvLayer *conv1x1Layer = new ConvLayer("conv1x1", in_dim, filter_dim(1, 1, in_dim.channels, cv1x1, 1), new Sigmoid());
	ConvLayer *conv3x3reduceLayer = new ConvLayer("conv3x3reduce", in_dim, filter_dim(1, 1, in_dim.channels, cv3x3reduce, 1), new Sigmoid());
	ConvLayer *conv3x3Layer = new ConvLayer("conv3x3", io_dim(in_dim.rows, in_dim.cols, cv3x3reduce), filter_dim(3, 3, cv3x3reduce, cv3x3, 1), new Sigmoid());
	ConvLayer *conv5x5recudeLayer = new ConvLayer("conv5x5reduce", in_dim, filter_dim(1, 1, in_dim.channels, cv5x5reduce, 1), new Sigmoid());
	ConvLayer *conv5x5Layer = new ConvLayer("conv5x5", io_dim(in_dim.rows, in_dim.cols, cv5x5reduce), filter_dim(5, 5, cv5x5reduce, cv5x5, 1), new Sigmoid());
	PoolingLayer *pool3x3Layer = new PoolingLayer("pool3x3", in_dim, pool_dim(3, 3, 1), new MaxPooling());
	ConvLayer *convProjectionLayer = new ConvLayer("convProjection", in_dim, filter_dim(1, 1, in_dim.channels, cp, 1), new Sigmoid());
	DepthConcatLayer *depthConcatLayer = new DepthConcatLayer("depthConcat", io_dim(in_dim.rows, in_dim.cols, cv1x1+cv3x3+cv5x5+cp));
	*/

	//Network::addLayerRelation(this, conv1x1Layer);
	//Network::addLayerRelation(this, conv3x3reduceLayer);
	//Network::addLayerRelation(this, conv5x5recudeLayer);
	//Network::addLayerRelation(this, pool3x3Layer);

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



void InceptionLayer::feedforward(int idx, const cube &input) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	for(int i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->feedforward(0, input);
	}
	Layer::feedforward(idx, lastLayer->getOutput());
}

void InceptionLayer::backpropagation(int idx, HiddenLayer *next_layer) {
	cube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);
	delta_input += w_next_delta;

	if(!isLastNextLayerRequest(idx)) return;
	lastLayer->backpropagation(0, this);

	delta_input.set_size(size(input));
	delta_input.zeros();
	for(int i = 0; i < firstLayers.size(); i++) {
		delta_input += firstLayers[i]->getDeltaInput();
	}

	HiddenLayer::backpropagation(idx, this);

	delta_input.set_size(size(output));
	delta_input.zeros();
}

void InceptionLayer::reset_nabla(int idx) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	for(int i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->reset_nabla(0);
	}

	Layer::reset_nabla(idx);
}

void InceptionLayer::update(int idx, double eta, double lambda, int n, int miniBatchSize) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	for(int i = 0; i < firstLayers.size(); i++) {
		firstLayers[i]->update(0, eta, lambda, n, miniBatchSize);
	}

	Layer::reset_nabla(idx);
}









