/*
 * CostAbstract.h
 *
 *  Created on: 2016. 4. 25.
 *      Author: jhkim
 */

#ifndef COST_COST_H_
#define COST_COST_H_

#include "../Util.h"

enum class CostType {
	None, CrossEntropy, LogLikelihood, Quadratic
};





class Cost {
public:
	Cost() {}
	virtual ~Cost() {}
	CostType getType() const { return type; }

#if CPU_MODE
public:
	virtual double fn(const rvec *pA, const rvec *pY) = 0;
	virtual void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) = 0;
#else
public:
	virtual double fn(const rvec *pA, const rvec *pY) = 0;
	virtual void d_cost(const DATATYPE *z, DATATYPE *activation, const UINT *target, DATATYPE *delta, UINT numLabels, UINT size) = 0;
#endif

protected:
	CostType type;
};

#endif /* COST_COST_H_ */
