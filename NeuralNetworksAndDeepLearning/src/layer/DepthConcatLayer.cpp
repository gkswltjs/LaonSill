/*
 * DepthConcatLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "DepthConcatLayer.h"

DepthConcatLayer::DepthConcatLayer(string name, int n_in)
	: HiddenLayer(name, n_in, n_in) {
	initialize();
}

DepthConcatLayer::DepthConcatLayer(string name, io_dim in_dim)
	: HiddenLayer(name, in_dim, in_dim) {
	initialize();
}

void DepthConcatLayer::initialize() {
	this->type = LayerType::DepthConcat;
	this->offsetIndex = 0;
	this->input.reset();
	this->delta_input.set_size(size(output));
	this->delta_input.zeros();
}




void DepthConcatLayer::feedforward(UINT idx, const rcube &input) {
	this->input = join_slices(this->input, input);
	Util::printCube(this->input, "input:");

	this->offsets.push_back(this->input.n_slices);

	if(!isLastPrevLayerRequest(idx)) return;

	this->output = this->input;

	Layer::feedforward(idx, this->output);

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

	HiddenLayer::backpropagation(idx, this);
	this->delta_input.zeros();
}


rcube &DepthConcatLayer::getDeltaInput() {
	int startIndex = (offsetIndex>0)?offsets[offsetIndex-1]:0;
	delta_input_sub = delta_input.subcube(0, 0, startIndex, delta_input.n_rows-1, delta_input.n_cols-1, offsets[offsetIndex]-1);
	offsetIndex++;
	//if(offsetIndex > prevLayers.size()) offsetIndex = 0;
	return delta_input_sub;
}
