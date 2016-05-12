/*
 * OutputLayer.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef LAYER_OUTPUTLAYER_H_
#define LAYER_OUTPUTLAYER_H_

#include "HiddenLayer.h"
#include <armadillo>

using namespace arma;


class OutputLayer : public HiddenLayer {
public:
	OutputLayer() {};
	virtual ~OutputLayer() {};

	/**
	 * 현재 레이어가 최종 레이어인 경우 δL을 계산
	 * @param target: 현재 데이터에 대한 목적값
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void cost(const vec &target, const vec &input)=0;

};

#endif /* LAYER_OUTPUTLAYER_H_ */
