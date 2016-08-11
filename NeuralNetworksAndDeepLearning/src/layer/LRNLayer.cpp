/*
 * LRNLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "LRNLayer.h"
#include "../Util.h"





LRNLayer::LRNLayer(const string name, lrn_dim lrn_d) : HiddenLayer(name) {
	initialize(lrn_d);
}








void LRNLayer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::load(ifs, layerMap);

	lrn_dim lrn_d;
	ifs.read((char *)&lrn_d, sizeof(lrn_dim));

	initialize(lrn_d);
	LRNLayer::_shape(false);
}

void LRNLayer::_save(ofstream &ofs) {
	HiddenLayer::_save(ofs);
	ofs.write((char *)&lrn_d, sizeof(lrn_dim));
}



#ifndef GPU_MODE

LRNLayer::~LRNLayer() {}

void LRNLayer::initialize(lrn_dim lrn_d) {
	this->type = LayerType::LRN;

	this->lrn_d = lrn_d;
	this->z.set_size(size(input));
	this->delta_input.set_size(size(output));
	this->delta_input.zeros();
}


// (1 + alpha/n * sigma(i)(xi^2))^beta
void LRNLayer::feedforward(UINT idx, const rcube &input, const char *end) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	UINT i, j;
	int top_pad = (lrn_d.local_size-1)/2;
	int in_channel_idx;

	Util::convertCube(input, this->input);
	rcube sq = square(this->input);
	rmat temp(this->input.n_rows, this->input.n_cols);

	//Util::printCube(this->input, "input:");
	//Util::printCube(sq, "sq:");

	for(i = 0; i < this->input.n_slices; i++) {
		temp.zeros();
		for(j = 0; j < lrn_d.local_size; j++) {
			in_channel_idx = i - top_pad + j;
			if(in_channel_idx >= 0 && (UINT)in_channel_idx < this->input.n_slices) {
				temp += sq.slice(in_channel_idx);
			}
		}
		//Util::printMat(temp, "temp:");
		z.slice(i) = 1+(lrn_d.alpha/lrn_d.local_size)*temp;
		//Util::printMat(z.slice(i), "z:");
		temp = pow(z.slice(i), -lrn_d.beta);
		//Util::printMat(temp, "pow temp:");

		this->output.slice(i) = this->input.slice(i) % temp;
		//Util::printMat(this->output.slice(i), "output:");
	}

	propFeedforward(this->output, end);

}


void LRNLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	if(!isLastNextLayerRequest(idx)) throw Exception();

	UINT i, j;
	int top_pad = (lrn_d.local_size-1)/2;
	int in_channel_idx;
	double c = -2*lrn_d.alpha*lrn_d.beta/lrn_d.local_size;
	rmat temp(input.n_rows, input.n_cols);

	rcube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);

	//Util::printCube(input, "input:");
	//Util::printCube(w_next_delta, "w_next_delta:");
	//Util::printCube(z, "z:");

	for(i = 0; i < input.n_slices; i++) {
		temp.zeros();
		for(j = 0; j < lrn_d.local_size; j++) {
			in_channel_idx = i - top_pad + j;
			if(in_channel_idx >= 0 && (UINT)in_channel_idx < input.n_slices) {
				//Util::printMat(pow(z.slice(in_channel_idx), -lrn_d.beta-1), "pow");
				//Util::printMat(input.slice(in_channel_idx), "input:");
				//Util::printMat(w_next_delta.slice(in_channel_idx), "w_next_delta:");
				temp += pow(z.slice(in_channel_idx), -lrn_d.beta-1) % input.slice(in_channel_idx) % w_next_delta.slice(in_channel_idx);
			}
		}
		//Util::printMat(temp, "temp:");
		delta_input.slice(i) = c * temp % input.slice(i) + pow(z.slice(i), -lrn_d.beta) % w_next_delta.slice(i);
	}
	//Util::printCube(delta_input, "delta_input:");
	propBackpropagation();
	delta_input.zeros();
}

#else
void LRNLayer::initialize(lrn_dim lrn_d) {
	this->type = LayerType::LRN;
	this->lrn_d = lrn_d;

	checkCUDNN(cudnnCreateLRNDescriptor(&lrnDesc));
	checkCUDNN(cudnnSetLRNDescriptor(lrnDesc, lrn_d.local_size, lrn_d.alpha, lrn_d.beta, lrn_d.k));
}

void LRNLayer::_shape(bool recursive) {
	out_dim = in_dim;

	if(recursive) {
		HiddenLayer::_shape();
	}

	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*in_dim.batchsize()));
}

void LRNLayer::_clearShape() {
	checkCudaErrors(cudaFree(d_delta_input));

	HiddenLayer::_clearShape();
}


LRNLayer::~LRNLayer() {
	//checkCudaErrors(cudaFree(d_delta));
	checkCudaErrors(cudaFree(d_delta_input));

	checkCUDNN(cudnnDestroyLRNDescriptor(lrnDesc));
}


// (1 + alpha/n * sigma(i)(xi^2))^beta
void LRNLayer::feedforward(UINT idx, const DATATYPE *input, const char *end) {
	Util::printMessage("LRNLayer::feedforward()---"+string(name));
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	this->d_input = input;

	Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");

	checkCUDNN(cudnnLRNCrossChannelForward(Cuda::cudnnHandle,
			lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&alpha, inputTensorDesc, d_input,
			&beta, outputTensorDesc, d_output));

	//if(Util::validPage()) {
		//Util::setPrint(true);
		//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, this->name+string("/d_output:"));
		Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, 1, 1, this->name+string("/d_output:"));
		//Util::setPrint(false);
	//}

	propFeedforward(d_output, end);
}


void LRNLayer::backpropagation(UINT idx, DATATYPE *next_delta_input) {
	Util::printMessage("LRNLayer::backpropagation()---"+string(name));
	if(!isLastNextLayerRequest(idx)) throw Exception();

	//DATATYPE *next_delta_input = next_layer->getDeltaInput();
	Util::printDeviceData(next_delta_input, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "next_delta_input:");

	checkCUDNN(cudnnLRNCrossChannelBackward(Cuda::cudnnHandle,
			lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
			&alpha, outputTensorDesc, d_output, outputTensorDesc, next_delta_input, inputTensorDesc, d_input,
			&beta, inputTensorDesc, d_delta_input));

	Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_delta_input:");

	propBackpropagation();
}
#endif












