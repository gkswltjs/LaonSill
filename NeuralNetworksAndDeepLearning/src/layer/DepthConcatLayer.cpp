/*
 * DepthConcatLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "DepthConcatLayer.h"

//#define DEPTHCONCAT_LOG


DepthConcatLayer::DepthConcatLayer() {
	this->type = Layer::DepthConcat;
}

DepthConcatLayer::DepthConcatLayer(Builder* builder)
	: HiddenLayer(builder) {
	initialize();
}

DepthConcatLayer::DepthConcatLayer(const string name)
	: HiddenLayer(name) {
	initialize();
}


DepthConcatLayer::~DepthConcatLayer() {}


void DepthConcatLayer::shape(UINT idx, io_dim in_dim) {
	// DepthConcatLayer에서 필요로하는 output channel수만 카운트하고
	// 나머지는 모두 상위 레이어의 shape()로 위임한다.
	if (isFirstPrevLayerRequest(idx)) out_dim.channels = 0;
	out_dim.channels += in_dim.channels;

	HiddenLayer::shape(idx, in_dim);

#ifdef DEPTHCONCAT_LOG
	cout << "shape depthConcatLayer in_dim: " << this->in_dim.batches << "x" << this->in_dim.channels << "x" << this->in_dim.rows << "x" << this->in_dim.cols << endl;
	cout << "shape depthConcatLayer out_dim: " << this->out_dim.batches << "x" << this->out_dim.channels << "x" << this->out_dim.rows << "x" << this->out_dim.cols << endl;
#endif
}

void DepthConcatLayer::reshape(UINT idx, io_dim in_dim) {
	if (isFirstPrevLayerRequest(idx)) out_dim.channels = 0;
	out_dim.channels += in_dim.channels;
	HiddenLayer::reshape(idx, in_dim);

#ifdef DEPTHCONCAT_LOG
	cout << "reshape depthConcatLayer in_dim: " << this->in_dim.batches << "x" << this->in_dim.channels << "x" << this->in_dim.rows << "x" << this->in_dim.cols << endl;
	cout << "reshape depthConcatLayer out_dim: " << this->out_dim.batches << "x" << this->out_dim.channels << "x" << this->out_dim.rows << "x" << this->out_dim.cols << endl;
#endif
}


void DepthConcatLayer::_shape(bool recursive) {
	in_dim.channels = out_dim.channels;
	out_dim.rows = in_dim.rows;
	out_dim.cols = in_dim.cols;
	out_dim.batches = in_dim.batches;

	if (recursive) {
		HiddenLayer::_shape();
	}
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*in_dim.batchsize()));
}

void DepthConcatLayer::_clearShape() {
	//checkCudaErrors(cudaFree(d_delta_input));
	//d_delta_input = NULL;
	offsetIndex = 0;
	//out_dim.channels = 0;

	HiddenLayer::_clearShape();
}

void DepthConcatLayer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::_load(ifs, layerMap);
	initialize();
	DepthConcatLayer::_shape(false);
}



void DepthConcatLayer::propBackpropagation() {
	HiddenLayer *hiddenLayer;
	uint32_t offset = 0;
	for(UINT i = 0; i < prevLayers.size(); i++) {
		hiddenLayer = dynamic_cast<HiddenLayer *>(prevLayers[i]);
		if(i > 0) {
			offset += prevLayers[i-1]->getOutDimension().batchsize();
		}

		// !!! 대부분의 경우 _backpropagation에서 사용한 d_delta_input을 그대로 사용하므로 문제가 없지만
		// DepthConcatLayer와 같이 d_delta_input을 분배해야 하는 케이스가 있으므로 d_delta_input을 그대로 사용하지 말고
		// getter를 사용하여 이전 레이어에 d_delta_input을 전달해야 한다.
		if(hiddenLayer) {
			//_distGradToPrev(i, hiddenLayer);
			hiddenLayer->backpropagation(id, getInput(), offset);
		}
	}
}




#ifndef GPU_MODE
void DepthConcatLayer::initialize() {
	this->type = Layer::DepthConcat;

	this->offsetIndex = 0;
	this->input.reset();
	this->delta_input.set_size(size(output));
	this->delta_input.zeros();
}

void DepthConcatLayer::feedforward(UINT idx, const rcube &input, const char *end=0) {
	this->input = join_slices(this->input, input);
	Util::printCube(this->input, "input:");

	this->offsets.push_back(this->input.n_slices);

	if(!isLastPrevLayerRequest(idx)) return;

	this->output = this->input;

	propFeedforward(this->output, end);

	// backward pass에서 input을 사용하지 않으므로 여기서 reset할 수 있음
	this->input.reset();
	this->offsetIndex = 0;
}

void DepthConcatLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	Util::printCube(delta_input, "delta_input:");
	rcube w_next_delta(size(delta_input));
	Util::convertCube(next_layer->getDeltaInput(), delta_input);
	delta_input += w_next_delta;
	// delta_input = join_slices(this->delta_input, next_layer->getDeltaInput());
	if(!isLastNextLayerRequest(idx)) return;

	propBackpropagation();
	this->delta_input.zeros();
}
#endif











