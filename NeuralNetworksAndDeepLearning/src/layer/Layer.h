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
	Layer() {}
	virtual ~Layer() {}

	vec &getActivation() { return this->activation; }



	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void feedforward(const vec &input)=0;



protected:
	/**
	 * activation이자 레이어의 output
	 */
	vec activation;

};

#endif /* LAYER_LAYER_H_ */
