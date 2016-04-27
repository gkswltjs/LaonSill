/*
 * CrossEntropyCost.h
 *
 *  Created on: 2016. 4. 25.
 *      Author: jhkim
 */

#ifndef COST_CROSSENTROPYCOST_H_
#define COST_CROSSENTROPYCOST_H_


#include <armadillo>

#include "Cost.h"


using namespace arma;

class CrossEntropyCost : public Cost {

public:
	CrossEntropyCost() {}
	virtual ~CrossEntropyCost() {}

	double fn(const vec *pA, const vec *pY) {
		Util::printVec(pA, "activation");
		Util::printVec(pY, "y");

		vec left = (-1 * (*pY)) % log(*pA);
		vec right = (1 - (*pY)) % log(1 - (*pA));
		vec result = left - right;
		Util::printVec(&left, "left");
		Util::printVec(&right, "right");
		Util::printVec(&result, "result");
		return sum(left - right);
	}

	vec delta(const vec *pZ, const vec *pA, const vec *pY) {
		Util::printVec(pA, "activation");
		Util::printVec(pY, "y");
		vec result = *pA - *pY;
		Util::printVec(&result, "result");

		return (*pA - *pY);
	}
};






#endif /* COST_CROSSENTROPYCOST_H_ */
