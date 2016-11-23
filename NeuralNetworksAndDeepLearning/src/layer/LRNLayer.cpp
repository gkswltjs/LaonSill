/*
 * LRNLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "LRNLayer.h"
#include "Util.h"

using namespace std;

template <typename Dtype>
LRNLayer<Dtype>::LRNLayer() {
	this->type = Layer<Dtype>::LRN;
}

template <typename Dtype>
LRNLayer<Dtype>::LRNLayer(Builder* builder) : HiddenLayer<Dtype>(builder) {
	initialize(builder->_lrnDim);
}

template <typename Dtype>
LRNLayer<Dtype>::LRNLayer(const string name, lrn_dim lrn_d) : HiddenLayer<Dtype>(name) {
	initialize(lrn_d);
}

template <typename Dtype>
void LRNLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	this->_inputShape[0] = inputShape;
	this->_outputData[0]->shape(inputShape);

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	/*
	this->setInDimension(this->_inputData[0]->getShape());
	this->out_dim = this->in_dim;

	if(recursive) {
		HiddenLayer<Dtype>::_shape();
	}
	*/
}

template <typename Dtype>
void LRNLayer<Dtype>::_clearShape() {
	HiddenLayer<Dtype>::_clearShape();
}

/*
template <typename Dtype>
void LRNLayer<Dtype>::_save(ofstream &ofs) {
	HiddenLayer<Dtype>::_save(ofs);
	ofs.write((char *)&lrn_d, sizeof(lrn_dim));
}

template <typename Dtype>
void LRNLayer<Dtype>::_load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*>& layerMap) {
	HiddenLayer<Dtype>::_load(ifs, layerMap);

	lrn_dim lrn_d;
	ifs.read((char *)&lrn_d, sizeof(lrn_dim));

	initialize(lrn_d);
	LRNLayer<Dtype>::_shape(false);
}
*/















#ifndef GPU_MODE
LRNLayer<Dtype>::~LRNLayer() {}

void LRNLayer<Dtype>::initialize(lrn_dim lrn_d) {
	this->type = Layer<Dtype>::LRN;

	this->lrn_d = lrn_d;
	this->z.set_size(size(input));
	this->delta_input.set_size(size(output));
	this->delta_input.zeros();
}

// (1 + alpha/n * sigma(i)(xi^2))^beta
void LRNLayer<Dtype>::feedforward() {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	uint32_t i, j;
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
			if(in_channel_idx >= 0 && (uint32_t)in_channel_idx < this->input.n_slices) {
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

void LRNLayer<Dtype>::backpropagation(uint32_t idx, HiddenLayer *next_layer) {
	if(!isLastNextLayerRequest(idx)) throw Exception();

	uint32_t i, j;
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
			if(in_channel_idx >= 0 && (uint32_t)in_channel_idx < input.n_slices) {
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

#endif




template class LRNLayer<float>;


