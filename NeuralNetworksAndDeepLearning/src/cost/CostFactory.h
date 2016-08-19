/**
 * @file	CostFactory.h
 * @date	2016/6/7
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef COST_COSTFACTORY_H_
#define COST_COSTFACTORY_H_

#include "Cost.h"
#include "CrossEntropyCost.h"
#include "LogLikelihoodCost.h"
#include "QuadraticCost.h"


/**
 * @brief 주어진 Cost 타입에 따라 Cost 객체를 생성하여 반환
 * @details 주어진 Cost 타입에 따라 Cost 객체를 생성하여 반환하고
 *          사용이 완료된 Cost 객체를 소멸시키는 역할을 함.
 * @todo (객체를 생성한 곳에서 삭제한다는 원칙에 따라 만들었으나 수정이 필요)
 */
class CostFactory {
public:
	CostFactory() {}
	virtual ~CostFactory() {}

	/**
	 * @details 주어진 Cost 타입에 따라 Cost 객체를 생성하여 반환.
	 * @param activationType 생성하고자 하는 Activation 객체의 타입.
	 * @return 생성한 Activation 객체.
	 */
	static Cost *create(Cost::Type costType) {
		switch(costType) {
		case Cost::CrossEntropy: return new CrossEntropyCost();
		case Cost::LogLikelihood: return new LogLikelihoodCost();
		case Cost::Quadratic: return new QuadraticCost();
		case Cost::None:
		default: return 0;
		}
	}

	/**
	 * @details CostFactory에서 생성한 Cost 객체를 소멸.
	 * @param cost_fn Cost 객체에 대한 포인터 참조자.
	 */
	static void destroy(Cost *&cost_fn) {
		if(cost_fn) {
			delete cost_fn;
			cost_fn = NULL;
		}
	}
};

#endif /* COST_COSTFACTORY_H_ */
