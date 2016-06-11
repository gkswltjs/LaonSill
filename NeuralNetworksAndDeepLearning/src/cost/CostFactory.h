/*
 * CostFactory.h
 *
 *  Created on: 2016. 6. 7.
 *      Author: jhkim
 */

#ifndef COST_COSTFACTORY_H_
#define COST_COSTFACTORY_H_

#include "Cost.h"
#include "CrossEntropyCost.h"
#include "LogLikelihoodCost.h"
#include "QuadraticCost.h"



class CostFactory {
public:
	CostFactory() {}
	virtual ~CostFactory() {}

	static Cost *create(CostType costType) {
		switch(costType) {
		case CostType::CrossEntropy: return new CrossEntropyCost();
		case CostType::LogLikelihood: return new LogLikelihoodCost();
		case CostType::Quadratic: return new QuadraticCost();
		case CostType::None:
		default: return 0;
		}
	}

	static void destroy(Cost *&cost_fn) {
		if(cost_fn) {
			delete cost_fn;
			cost_fn = NULL;
		}
	}
};

#endif /* COST_COSTFACTORY_H_ */
