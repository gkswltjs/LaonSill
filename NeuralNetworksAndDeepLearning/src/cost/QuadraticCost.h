/*
 * QuadraticCost.h
 *
 *  Created on: 2016. 4. 25.
 *      Author: jhkim
 */

#ifndef COST_QUADRATICCOST_H_
#define COST_QUADRATICCOST_H_

#include <armadillo>

#include "Cost.h"


using namespace arma;



class QuadraticCost : public Cost {
public:
	QuadraticCost() {}
	virtual ~QuadraticCost() {}

	double fn(const vec *pA, const vec *pY) {
		return 0.5*sum(square(*pA - *pY));
	}

	void d_cost(const cube &z, const cube &activation, const vec &target, cube &delta) {
		delta.slice(0) = activation.slice(0) - target;
	}
};

#endif /* COST_QUADRATICCOST_H_ */
