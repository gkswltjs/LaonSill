/*
 * QuadraticCost.h
 *
 *  Created on: 2016. 4. 25.
 *      Author: jhkim
 */

#ifndef COST_QUADRATICCOST_H_
#define COST_QUADRATICCOST_H_



#include "Cost.h"


class QuadraticCost : public Cost {
public:
	QuadraticCost() {}
	virtual ~QuadraticCost() {}

	double fn(const rvec *pA, const rvec *pY) {
		return 0.5*sum(square(*pA - *pY));
	}

	void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) {
		delta.slice(0) = activation.slice(0) - target;
	}
};

#endif /* COST_QUADRATICCOST_H_ */
