/*
 * LogLikelihoodCost.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef COST_LOGLIKELIHOODCOST_H_
#define COST_LOGLIKELIHOODCOST_H_

#include "Cost.h"
#include "../cuda/Cuda.h"


class LogLikelihoodCost : public Cost {
public:
	LogLikelihoodCost();
	virtual ~LogLikelihoodCost();

#if CPU_MODE
	double fn(const rvec *pA, const rvec *pY);
	void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta);
#else
	double fn(const rvec *pA, const rvec *pY);
	void d_cost(const DATATYPE *z, DATATYPE *activation, const UINT *target, DATATYPE *delta, UINT numLabels, UINT batchsize);
#endif

};


#endif /* COST_LOGLIKELIHOODCOST_H_ */


