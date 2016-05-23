/*
 * Layer.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef LAYER_LAYER_H_
#define LAYER_LAYER_H_

#include "LayerConfig.h"
#include "../Util.h"
#include <armadillo>

using namespace arma;


class Layer {
public:
	Layer(int n_in, int n_out) {
		this->in_dim.rows = n_in;
		this->out_dim.rows = n_out;
		this->input.set_size(n_in, 1, 1);
		this->output.set_size(n_out, 1, 1);
	}
	Layer(io_dim in_dim, io_dim out_dim) {
		this->in_dim = in_dim;
		this->out_dim = out_dim;
		this->input.set_size(in_dim.rows, in_dim.cols, in_dim.channels);
		this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);
	}
	virtual ~Layer() {}

	cube &getInput() { return this->input; }
	cube &getOutput() { return this->output; }

	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void feedforward(const cube &input)=0;




protected:
	io_dim in_dim;
	io_dim out_dim;

	/**
	 * activation이자 레이어의 output
	 */
	cube input;
	cube output;

};

#endif /* LAYER_LAYER_H_ */





























