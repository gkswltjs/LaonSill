/*
 * DepthConcatLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "DepthConcatLayer.h"

DepthConcatLayer::DepthConcatLayer(const char *name)
	: HiddenLayer(name) {
	initialize();
}

void DepthConcatLayer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::load(ifs, layerMap);
	initialize();
	DepthConcatLayer::_shape(false);
}

#if CPU_MODE
DepthConcatLayer::DepthConcatLayer(const char *name, int n_in)
	: HiddenLayer(name, n_in, n_in) {
	initialize();
}

DepthConcatLayer::~DepthConcatLayer() {}

void DepthConcatLayer::initialize() {
	this->type = LayerType::DepthConcat;

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


rcube &DepthConcatLayer::getDeltaInput() {
	int startIndex = (offsetIndex>0)?offsets[offsetIndex-1]:0;
	delta_input_sub = delta_input.subcube(0, 0, startIndex, delta_input.n_rows-1, delta_input.n_cols-1, offsets[offsetIndex]-1);
	offsetIndex++;
	//if(offsetIndex > prevLayers.size()) offsetIndex = 0;
	return delta_input_sub;
}

#else


void DepthConcatLayer::initialize() {
	this->type = LayerType::DepthConcat;
	this->offsetIndex = 0;
	this->out_dim.channels = 0;
}

void DepthConcatLayer::shape(UINT idx, io_dim in_dim) {
	out_dim.channels += in_dim.channels;
	if(!isLastPrevLayerRequest(idx)) return;

	HiddenLayer::shape(idx, in_dim);
}

void DepthConcatLayer::_shape(bool recursive) {
	in_dim.channels = out_dim.channels;
	out_dim.rows = in_dim.rows;
	out_dim.cols = in_dim.cols;
	out_dim.batches = in_dim.batches;

	if(recursive) {
		HiddenLayer::_shape();
	}

	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*in_dim.batchsize()));
}

void DepthConcatLayer::reshape(UINT idx, io_dim in_dim) {
	if(!isLastPrevLayerRequest(idx)) return;
	Layer::reshape(idx, in_dim);
}

void DepthConcatLayer::clearShape(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) return;
	Layer::clearShape(idx);
}

void DepthConcatLayer::_clearShape() {
	checkCudaErrors(cudaFree(d_delta_input));
	d_delta_input = 0;
	offsetIndex = 0;
	out_dim.channels = 0;

	HiddenLayer::_clearShape();
}

DepthConcatLayer::~DepthConcatLayer() {

	checkCudaErrors(cudaFree(d_delta_input));
}



void DepthConcatLayer::feedforward(UINT idx, const DATATYPE *input, const char *end) {
	bool print = Util::getPrint();
	//Util::setPrint(true);

	Util::printMessage("DepthConcatLayer::feedforward()---"+string(name));
	//if(idx == 0) {
	//	checkCudaErrors(cudaMemset(d_output, 0, sizeof(DATATYPE)*out_dim.batchsize()));
	//	offsetIndex = 0;
	//}

	// 이전 레이어들의 전달값을 batch 단위로 합쳐서 다음 레이어로 전달
	// 한 batch내에서의 해당 이전 레이어 전달값의 offset 위치 계산
	int inBatchOffset = 0;
	for(int i = 0; i < idx; i++) {
		inBatchOffset += prevLayers[i].prev_layer->getOutDimension().unitsize();
	}
	io_dim prev_out_dim = prevLayers[idx].prev_layer->getOutDimension();

	Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_output:");
	Util::printDeviceData(input, prev_out_dim.rows, prev_out_dim.cols, prev_out_dim.channels, prev_out_dim.batches, "input:");
	for(int i = 0; i < prev_out_dim.batches; i++) {
		checkCudaErrors(cudaMemcpyAsync(this->d_output+out_dim.unitsize()*i+inBatchOffset, input+prev_out_dim.unitsize()*i,
				sizeof(DATATYPE)*prev_out_dim.unitsize(), cudaMemcpyDeviceToDevice));
	}

	//if(Util::temp_flag && strncmp("inception", this->name, 9) == 0) {
	//if(Util::validPage()) {
		//Util::setPrint(true);
		Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, this->name+string("/d_output:"));
		//Util::setPrint(false);
	//}

	if(!isLastPrevLayerRequest(idx)) return;
	propFeedforward(this->d_output, end);
}

void DepthConcatLayer::backpropagation(UINT idx, DATATYPE *next_delta_input) {
	bool print = Util::getPrint();
	//Util::setPrint(true);

	Util::printMessage("DepthConcatLayer::backpropagation()---"+string(name));
	if(idx == 0) {
		checkCudaErrors(cudaMemset(d_delta_input, 0, sizeof(DATATYPE)*out_dim.batchsize()));
		offsetIndex = 0;
	}

	vector<int> offsets(prevLayers.size());
	offsets[0] = 0;
	for(int i = 1; i < offsets.size(); i++) {
		offsets[i] = offsets[i-1] + prevLayers[i-1].prev_layer->getOutDimension().unitsize();
	}

	//DATATYPE *next_delta_input = next_layer->getDeltaInput();
	Util::printDeviceData(d_delta_input, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_input:");
	Util::printDeviceData(next_delta_input, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "next_delta_input:");
	for(int j = 0; j < prevLayers.size(); j++) {
		for(int i = 0; i < out_dim.batches; i++) {
			//cout << "next_delta_input offset: " << out_dim.unitsize()*i+offsets[j] << ", d_delta_input offset: " << offsets[j]*2+prevLayers[j].prev_layer->getOutDimension().unitsize()*i << endl;
			checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(prevLayers[j].prev_layer->getOutDimension().unitsize()),
					&alpha, next_delta_input+out_dim.unitsize()*i+offsets[j], 1, d_delta_input+offsets[j]*2+prevLayers[j].prev_layer->getOutDimension().unitsize()*i, 1));
		}
	}
	Util::printDeviceData(d_delta_input, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_input:");

	if(!isLastNextLayerRequest(idx)) return;

	//Util::setPrint(print);
	propBackpropagation();
}


DATATYPE *DepthConcatLayer::getDeltaInput() {
	bool print = Util::getPrint();
	//Util::setPrint(true);

	int inBatchOffset = 0;
	for(int i = 0; i < offsetIndex; i++) {
		inBatchOffset += prevLayers[i].prev_layer->getOutDimension().batchsize();
	}

	io_dim prev_out_dim = prevLayers[offsetIndex].prev_layer->getOutDimension();
	Util::printDeviceData(d_delta_input+inBatchOffset, prev_out_dim.rows, prev_out_dim.cols, prev_out_dim.channels, prev_out_dim.batches, "d_delta_input:");

	offsetIndex++;

	//Util::setPrint(print);
	return d_delta_input+inBatchOffset;
}


#endif
















