/*
 * CostAbstract.h
 *
 *  Created on: 2016. 4. 25.
 *      Author: jhkim
 */

#ifndef COST_COST_H_
#define COST_COST_H_

#include <armadillo>

using namespace arma;


class Cost {
public:
	Cost() {}
	virtual ~Cost() {}

	virtual double fn(const vec *pA, const vec *pY) = 0;
	virtual vec delta(const vec *pZ, const vec *pA, const vec *pY) = 0;
};

#endif /* COST_COST_H_ */
