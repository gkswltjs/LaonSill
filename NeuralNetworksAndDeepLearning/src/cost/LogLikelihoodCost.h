/*
 * LogLikelihoodCost.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef COST_LOGLIKELIHOODCOST_H_
#define COST_LOGLIKELIHOODCOST_H_

#include "Cost.h"


class LogLikelihoodCost : public Cost {
public:
	LogLikelihoodCost() {
		this->type = CostType::LogLikelihood;
	}
	virtual ~LogLikelihoodCost() {}

	double fn(const rvec *pA, const rvec *pY) {
		return 0.0;
	}

	void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) {
		Util::printCube(activation, "activation:");
		Util::printVec(target, "target:");

		delta.slice(0) = activation.slice(0) - target;
		Util::printCube(delta, "delta:");
	}
};

#endif /* COST_LOGLIKELIHOODCOST_H_ */
