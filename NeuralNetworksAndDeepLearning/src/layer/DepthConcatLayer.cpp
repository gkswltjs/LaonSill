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

#ifndef GPU_MODE
DepthConcatLayer::DepthConcatLayer(const string name, int n_in)
	: HiddenLayer(name, n_in, n_in) {
	initialize();
}
#endif

DepthConcatLayer::~DepthConcatLayer() {}



/*
#ifndef GPU_MODE
rcube &DepthConcatLayer::getDeltaInput() {
	int startIndex = (offsetIndex>0)?offsets[offsetIndex-1]:0;
	delta_input_sub = delta_input.subcube(0, 0, startIndex, delta_input.n_rows-1, delta_input.n_cols-1, offsets[offsetIndex]-1);
	offsetIndex++;
	return delta_input_sub;
}
#else
DATATYPE* DepthConcatLayer::getDeltaInput() {
	int inBatchOffset = 0;
	for(int i = 0; i < offsetIndex; i++) {
		inBatchOffset += prevLayers[i]->getOutDimension().batchsize();
	}

	io_dim prev_out_dim = prevLayers[offsetIndex]->getOutDimension();
	Util::printDeviceData(d_delta_input+inBatchOffset, prev_out_dim.rows, prev_out_dim.cols, prev_out_dim.channels, prev_out_dim.batches, "d_delta_input:");

#ifdef DEPTHCONCAT_LOG
	cout << "getDeltaInput Request from " << offsetIndex << "th prev layer " << prevLayers[offsetIndex]->getName() << ", inBatchOffset: " << inBatchOffset << endl;
	cout << "whose out_dim is: " << prev_out_dim.batches << "x" << prev_out_dim.channels << "x" << prev_out_dim.rows << "x" << prev_out_dim.cols << endl;
#endif

	offsetIndex++;
	return d_delta_input+inBatchOffset;
}
#endif
*/




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
#else
void DepthConcatLayer::initialize() {
	this->type = Layer::DepthConcat;
	this->offsetIndex = 0;
	this->out_dim.channels = 0;
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

void DepthConcatLayer::_concat(UINT idx, Data* input) {
#ifdef DEPTHCONCAT_LOG
	cout << "depthConcat _concat " << endl;
#endif
	// 이전 레이어들의 전달값을 batch 단위로 합쳐서 다음 레이어로 전달
	// 한 batch내에서의 해당 이전 레이어 전달값의 offset 위치 계산

	int inBatchOffset = 0;
	int i = 0;
	while(i < prevLayers.size() && prevLayers[i]->getId() != idx) {
		inBatchOffset += prevLayers[i]->getOutDimension().unitsize();
		i++;
	}
	io_dim prev_out_dim = prevLayers[i]->getOutDimension();

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	//Util::printDeviceData(input, prev_out_dim.rows, prev_out_dim.cols, prev_out_dim.channels, prev_out_dim.batches, "input:");
	_input->print_data("d_input:");
	input->print_data("input:");

	DATATYPE* d_input = _input->mutable_device_data();
	const DATATYPE* prev_input = input->device_data();
	for(int i = 0; i < prev_out_dim.batches; i++) {
		checkCudaErrors(cudaMemcpyAsync(d_input+in_dim.unitsize()*i+inBatchOffset, prev_input+prev_out_dim.unitsize()*i,
				sizeof(DATATYPE)*prev_out_dim.unitsize(), cudaMemcpyDeviceToDevice));
	}

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, this->name+string("/d_input:"));
	_input->print_data(this->name+string("/d_input:"));
}


void DepthConcatLayer::_deconcat(UINT idx, Data* next_delta_input, uint32_t offset) {
#ifdef DEPTHCONCAT_LOG
	cout << "depthConcat _deconcat " << endl;
#endif
	if(isFirstNextLayerRequest(idx)) {
		//checkCudaErrors(cudaMemset(d_delta_output, 0, sizeof(DATATYPE)*out_dim.batchsize()));
		_output->reset_device_grad();
		offsetIndex = 0;
	}

	//vector<int> offsets(prevLayers.size());
	//offsets[0] = 0;
	//for(int i = 1; i < offsets.size(); i++) {
	//	offsets[i] = offsets[i-1] + prevLayers[i-1]->getOutDimension().unitsize();
	//}

	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	//Util::printDeviceData(next_delta_input, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "next_delta_input:");
	_output->print_grad("d_delta_output:");
	next_delta_input->print_grad("next_delta_input:");


	const DATATYPE* d_next_delta_input = next_delta_input->device_grad();
	DATATYPE* d_delta_output = _output->mutable_device_grad();
	uint32_t layerOffset = 0;
	for(int j = 0; j < prevLayers.size(); j++) {
		if(j > 0) {
			layerOffset += prevLayers[j-1]->getOutDimension().unitsize();
		}
		for(int i = 0; i < out_dim.batches; i++) {
			//cout << "next_delta_input offset: " << out_dim.unitsize()*i+offsets[j] << ", d_delta_input offset: " << offsets[j]*2+prevLayers[j]->getOutDimension().unitsize()*i << endl;
			checkCudaErrors(cublasSaxpy(
					Cuda::cublasHandle,
					static_cast<int>(prevLayers[j]->getOutDimension().unitsize()),
					&Cuda::alpha,
					d_next_delta_input+out_dim.unitsize()*i+layerOffset,
					1,
					d_delta_output+layerOffset*2+prevLayers[j]->getOutDimension().unitsize()*i,
					1));
		}
	}
	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	_output->print_grad("d_delta_output:");
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




void DepthConcatLayer::_scaleInput() {}

void DepthConcatLayer::_scaleGradient() {}

#endif
















