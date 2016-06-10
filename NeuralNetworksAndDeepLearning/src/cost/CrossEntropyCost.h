/*
 * CrossEntropyCost.h
 *
 *  Created on: 2016. 4. 25.
 *      Author: jhkim
 */

#ifndef COST_CROSSENTROPYCOST_H_
#define COST_CROSSENTROPYCOST_H_


#include "Cost.h"


class CrossEntropyCost : public Cost {

public:
	CrossEntropyCost() {
		this->type = CostType::CrossEntropy;
	}
	virtual ~CrossEntropyCost() {}

	double fn(const rvec *pA, const rvec *pY) {
		/*
		Util::printVec(pA, "activation");
		Util::printVec(pY, "y");

		rvec left = (-1 * (*pY)) % log(*pA);
		rvec right = (1 - (*pY)) % log(1 - (*pA));
		rvec result = left - right;
		Util::printVec(&left, "left");
		Util::printVec(&right, "right");
		Util::printVec(&result, "result");
		return sum(left - right);
		*/
		return 0.0;
	}

	void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) {
		//Util::printVec(z, "activation");
		//Util::printVec(activation, "y");
		delta.slice(0) = activation.slice(0) - target;
		//Util::printVec(delta, "result");
	}
};






#endif /* COST_CROSSENTROPYCOST_H_ */
