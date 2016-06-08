/*
 * LRNLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "LRNLayer.h"
#include "../Util.h"


LRNLayer::LRNLayer(string name, io_dim in_dim, lrn_dim lrn_d) : HiddenLayer(name, in_dim, in_dim) {
	this->type = LayerType::LRN;
	this->lrn_d = lrn_d;
	this->z.set_size(size(input));
	this->delta_input.set_size(size(output));
	this->delta_input.zeros();
}

LRNLayer::~LRNLayer() {}


// (1 + alpha/n * sigma(i)(xi^2))^beta
void LRNLayer::feedforward(UINT idx, const rcube &input) {
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

	Layer::feedforward(idx, this->output);

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
	HiddenLayer::backpropagation(idx, this);
	delta_input.zeros();
}




























