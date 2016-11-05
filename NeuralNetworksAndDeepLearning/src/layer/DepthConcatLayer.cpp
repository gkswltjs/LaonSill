/*
 * DepthConcatLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "DepthConcatLayer.h"

using namespace std;

//#define DEPTHCONCAT_LOG

template <typename Dtype>
DepthConcatLayer<Dtype>::DepthConcatLayer() {
	this->type = Layer<Dtype>::DepthConcat;
}

template <typename Dtype>
DepthConcatLayer<Dtype>::DepthConcatLayer(Builder* builder)
	: HiddenLayer<Dtype>(builder) {
	initialize();
}

template <typename Dtype>
DepthConcatLayer<Dtype>::DepthConcatLayer(const string name)
	: HiddenLayer<Dtype>(name) {
	initialize();
}

template <typename Dtype>
DepthConcatLayer<Dtype>::~DepthConcatLayer() {}


template <typename Dtype>
//void DepthConcatLayer<Dtype>::shape(uint32_t idx, io_dim in_dim, Data<Dtype>* prevLayerOutput) {
void DepthConcatLayer<Dtype>::shape() {
	// DepthConcatLayer에서 필요로하는 output channel수만 카운트하고
	// 나머지는 모두 상위 레이어의 shape()로 위임한다.
	//if (this->isFirstPrevLayerRequest(idx)) this->out_dim.channels = 0;
	//this->out_dim.channels += in_dim.channels;

	//HiddenLayer<Dtype>::shape(idx, in_dim, prevLayerOutput);

#ifdef DEPTHCONCAT_LOG
	cout << "shape depthConcatLayer in_dim: " << this->in_dim.batches << "x" << this->in_dim.channels << "x" << this->in_dim.rows << "x" << this->in_dim.cols << endl;
	cout << "shape depthConcatLayer out_dim: " << this->out_dim.batches << "x" << this->out_dim.channels << "x" << this->out_dim.rows << "x" << this->out_dim.cols << endl;
#endif
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::reshape(uint32_t idx, io_dim in_dim) {
	//if (this->isFirstPrevLayerRequest(idx)) this->out_dim.channels = 0;
	this->out_dim.channels += in_dim.channels;
	HiddenLayer<Dtype>::reshape(idx, in_dim);

#ifdef DEPTHCONCAT_LOG
	cout << "reshape depthConcatLayer in_dim: " << this->in_dim.batches << "x" << this->in_dim.channels << "x" << this->in_dim.rows << "x" << this->in_dim.cols << endl;
	cout << "reshape depthConcatLayer out_dim: " << this->out_dim.batches << "x" << this->out_dim.channels << "x" << this->out_dim.rows << "x" << this->out_dim.cols << endl;
#endif
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::_shape(bool recursive) {
	this->in_dim.channels = this->out_dim.channels;
	this->out_dim.rows = this->in_dim.rows;
	this->out_dim.cols = this->in_dim.cols;
	this->out_dim.batches = this->in_dim.batches;

	if (recursive) {
		HiddenLayer<Dtype>::_shape();
	}
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::_clearShape() {
	offsetIndex = 0;
	HiddenLayer<Dtype>::_clearShape();
}


/*
template <typename Dtype>
void DepthConcatLayer<Dtype>::_load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*> &layerMap) {
	HiddenLayer<Dtype>::_load(ifs, layerMap);
	initialize();
	DepthConcatLayer<Dtype>::_shape(false);
}
*/


template <typename Dtype>
void DepthConcatLayer<Dtype>::propBackpropagation() {
	HiddenLayer<Dtype>*hiddenLayer;
	uint32_t offset = 0;
	/*
	for(uint32_t i = 0; i < this->prevLayers.size(); i++) {
		hiddenLayer = dynamic_cast<HiddenLayer<Dtype>*>(this->prevLayers[i]);
		if(i > 0) {
			offset += this->prevLayers[i-1]->getOutDimension().batchsize();
		}

		// !!! 대부분의 경우 _backpropagation에서 사용한 d_inputGrad을 그대로 사용하므로 문제가 없지만
		// DepthConcatLayer와 같이 d_inputGrad을 분배해야 하는 케이스가 있으므로 d_inputGrad을 그대로 사용하지 말고
		// getter를 사용하여 이전 레이어에 d_inputGrad을 전달해야 한다.
		if(hiddenLayer) {
			//hiddenLayer->backpropagation(this->id, this->getInput(), offset);
		}
	}
	*/
}




#ifndef GPU_MODE
template <typename Dtype>
void DepthConcatLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::DepthConcat;

	this->offsetIndex = 0;
	this->input.reset();
	this->delta_input.set_size(size(output));
	this->delta_input.zeros();
}

template <typename Dtype>
void DepthConcatLayer<Dtype>::feedforward(uint32_t idx, const rcube &input, const char *end=0) {
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

template <typename Dtype>
void DepthConcatLayer<Dtype>::backpropagation(uint32_t idx, HiddenLayer<Dtype>* next_layer) {
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



template class DepthConcatLayer<float>;







