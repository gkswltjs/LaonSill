/*
 * Layer.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef LAYER_LAYER_H_
#define LAYER_LAYER_H_

#include <armadillo>

using namespace arma;


class Layer {
public:
	Layer(int n_in, int n_out) {
		this->n_in = n_in;
		this->n_out = n_out;
	}
	virtual ~Layer() {}

	int getNIn() const { return this->n_in; }
	int getNOut() const { return this->n_out; }
	vec &getOutput() { return this->output; }



	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void feedforward(const vec &input)=0;



protected:

	int n_in;
	int n_out;

	/**
	 * activation이자 레이어의 output
	 */
	vec output;

};

#endif /* LAYER_LAYER_H_ */
