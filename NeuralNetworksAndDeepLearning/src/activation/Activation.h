/*
 * Activation.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef ACTIVATION_ACTIVATION_H_
#define ACTIVATION_ACTIVATION_H_

#include "../layer/LayerConfig.h"
#include "../Util.h"


enum class ActivationType {
	None, Sigmoid, Softmax, ReLU
};


class Activation {
public:
	Activation() {};
	virtual ~Activation() {};

	ActivationType getType() const { return type; }

	/**
	 * activation function에 따라 layer weight의 초기화하는 방법이 다름
	 */
	//virtual void initialize_weight(int n_in, rmat &weight)=0;
	//virtual void initialize_weight(int n_in, rcube &weight)=0;


	/**
	 * activation function
	 */
	virtual void activate(const rcube &z, rcube &activation)=0;

	/**
	 * activation derivation
	 * 실제 weighted sum값을 이용하여 계산하여야 하지만
	 * 현재까지 activation으로도 계산이 가능하여 파라미터를 activation으로 지정
	 * weighted sum값이 필요한 케이스에 수정이 필요
	 */
	virtual void d_activate(const rcube &activation, rcube &da)=0;

protected:
	ActivationType type;

};

#endif /* ACTIVATION_ACTIVATION_H_ */
