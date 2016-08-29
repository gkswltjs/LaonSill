/**
 * @file ActivationFactory.h
 * @date 2016/6/7
 * @author jhkim
 * @brief
 * @details
 */


#ifndef ACTIVATION_ACTIVATIONFACTORY_H_
#define ACTIVATION_ACTIVATIONFACTORY_H_

#include <stddef.h>

#include "Activation.h"
#include "ReLU.h"
#include "Sigmoid.h"
#include "Softmax.h"




/**
 * @brief 주어진 Activation 타입에 따라 Activation 객체를 생성하여 반환
 * @details 주어진 Activation 타입에 따라 Activation 객체를 생성하여 반환하고
 *          사용이 완료된 Activation 객체를 소멸시키는 역할을 함.
 * @todo (객체를 생성한 곳에서 삭제한다는 원칙에 따라 만들었으나 수정이 필요)
 */
class ActivationFactory {
public:
	ActivationFactory() {}
	virtual ~ActivationFactory() {}

#ifndef GPU_MODE
public:
	static Activation *create(Activation::Type activationType) {
		switch(activationType) {
		case Activation::Sigmoid: return new Sigmoid();
		case Activation::Softmax: return new Softmax();
		case Activation::ReLU: return new ReLU();
		case Activation::None:
		default: return 0;
		}
	}

	static void destory(Activation *&activation_fn) {
		if(activation_fn) {
			delete activation_fn;
			activation_fn = NULL;
		}
	}
#else
public:
	/**
	 * @details 주어진 Activation 타입에 따라 Activation 객체를 생성하여 반환.
	 * @param activationType 생성하고자 하는 Activation 객체의 타입.
	 * @return 생성한 Activation 객체.
	 */
	static Activation* create(Activation::Type activationType) {
		switch(activationType) {
		case Activation::Sigmoid: return new Sigmoid();
		case Activation::Softmax: return new Softmax();
		case Activation::ReLU: return new ReLU();
		case Activation::NoActivation:
		default: return 0;
		}
	}

	/**
	 * @details ActivationFactory에서 생성한 Activation 객체를 소멸.
	 * @param activation_fn Activation 객체에 대한 포인터 참조자.
	 */
	static void destory(Activation *&activation_fn) {
		if(activation_fn) {
			delete activation_fn;
			activation_fn = NULL;
		}
	}
#endif
};

#endif /* ACTIVATION_ACTIVATIONFACTORY_H_ */
