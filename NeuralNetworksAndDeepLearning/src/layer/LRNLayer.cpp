/*
 * LRNLayer.cpp
 *
 *  Created on: 2016. 5. 25.
 *      Author: jhkim
 */

#include "LRNLayer.h"
#include "../Util.h"

LRNLayer::LRNLayer(io_dim in_dim, lrn_dim lrn_d) : HiddenLayer(in_dim, in_dim) {
	this->lrn_d = lrn_d;
	this->z.set_size(size(input));
	this->delta_input.set_size(size(output));
}

LRNLayer::~LRNLayer() {}


// (1 + alpha/n * sigma(i)(xi^2))^beta
void LRNLayer::feedforward(const cube &input) {
	int i, j;
	int top_pad = (lrn_d.local_size-1)/2;
	int in_channel_idx;

	Util::convertCube(input, this->input);
	cube sq = square(this->input);
	mat temp(this->input.n_rows, this->input.n_cols);

	Util::printCube(this->input, "input:");
	Util::printCube(sq, "sq:");

	for(i = 0; i < this->input.n_slices; i++) {
		temp.zeros();
		for(j = 0; j < lrn_d.local_size; j++) {
			in_channel_idx = i - top_pad + j;
			if(in_channel_idx >= 0 && in_channel_idx < this->input.n_slices) {
				temp += sq.slice(in_channel_idx);
			}
		}
		Util::printMat(temp, "temp:");
		z.slice(i) = 1+(lrn_d.alpha/lrn_d.local_size)*temp;
		Util::printMat(z.slice(i), "z:");
		temp = pow(z.slice(i), -lrn_d.beta);
		Util::printMat(temp, "pow temp:");

		this->output.slice(i) = this->input.slice(i) % temp;
		Util::printMat(this->output.slice(i), "output:");
	}

	Layer::feedforward(this->output);
}



void LRNLayer::backpropagation(HiddenLayer *next_layer) {

	int i, j, k, l, m;
	int top_pad = (lrn_d.local_size-1)/2;
	int in_channel_idx;
	double c = -2*lrn_d.alpha*lrn_d.beta/lrn_d.local_size;
	double sum;
	mat temp(input.n_rows, input.n_cols);

	cube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);

	Util::printCube(input, "input:");

	// TODO for debug 삭제해야 함
	// w_next_delta = randu<cube>(size(output));
	Util::printCube(w_next_delta, "w_next_delta:");
	Util::printCube(z, "z:");

	for(i = 0; i < input.n_slices; i++) {
		temp.zeros();
		for(j = 0; j < lrn_d.local_size; j++) {
			in_channel_idx = i - top_pad + j;
			if(in_channel_idx >= 0 && in_channel_idx < input.n_slices) {
				temp += pow(z.slice(in_channel_idx), -lrn_d.beta-1)%input.slice(in_channel_idx);
			}
		}
		delta_input.slice(i) = c * temp % input.slice(i) + pow(z.slice(i), -lrn_d.beta);
	}
	Util::printCube(delta_input, "delta_input:");



	HiddenLayer::backpropagation(this);
}




























