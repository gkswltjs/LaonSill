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

	void d_cost(const vec &z, const vec &activation, const vec &target, vec &delta) {
		delta = activation - target;
	}
};

#endif /* COST_QUADRATICCOST_H_ */
