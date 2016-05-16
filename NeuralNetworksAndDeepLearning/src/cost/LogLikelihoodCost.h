/*
 * LogLikelihoodCost.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef COST_LOGLIKELIHOODCOST_H_
#define COST_LOGLIKELIHOODCOST_H_

#include "Cost.h"
#include <armadillo>

using namespace arma;


class LogLikelihoodCost : public Cost {
public:
	LogLikelihoodCost() {}
	virtual ~LogLikelihoodCost() {}

	double fn(const vec *pA, const vec *pY) {
		return 0.0;
	}

	void d_cost(const cube &z, const cube &activation, const vec &target, cube &delta) {
		delta.slice(0) = activation.slice(0) - target;
	}
};

#endif /* COST_LOGLIKELIHOODCOST_H_ */
