/*
 * Layer.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef LAYER_LAYER_H_
#define LAYER_LAYER_H_

#include "LayerConfig.h"
#include <armadillo>

using namespace arma;


class Layer {
public:
	Layer(int n_in, int n_out) {
		this->in_dim.rows = n_in;
		this->out_dim.rows = n_out;
	}
	Layer(io_dim in_dim, io_dim out_dim) {
		this->in_dim = in_dim;
		this->out_dim = out_dim;
	}
	virtual ~Layer() {}

	//int getNIn() const { return this->n_in; }
	//int getNOut() const { return this->n_out; }
	cube &getOutput() { return this->output; }



	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void feedforward(const cube &input)=0;




protected:
	void convertInputDim(const cube &input, cube &converted) {
		converted = reshape(input, in_dim.rows, in_dim.cols, in_dim.channels, 1);
		//cube C = reshape(A, 5*4*3, 1, 1, 1);
	}

	io_dim in_dim;
	io_dim out_dim;

	/**
	 * activation이자 레이어의 output
	 */
	cube output;

};

#endif /* LAYER_LAYER_H_ */
