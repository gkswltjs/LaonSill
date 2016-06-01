/*
 * CostAbstract.h
 *
 *  Created on: 2016. 4. 25.
 *      Author: jhkim
 */

#ifndef COST_COST_H_
#define COST_COST_H_

#include "../Util.h"


class Cost {
public:
	Cost() {}
	virtual ~Cost() {}

	virtual double fn(const rvec *pA, const rvec *pY) = 0;
	virtual void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) = 0;
};

#endif /* COST_COST_H_ */
