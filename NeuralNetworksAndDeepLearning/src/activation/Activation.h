/*
 * Activation.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef ACTIVATION_ACTIVATION_H_
#define ACTIVATION_ACTIVATION_H_

#include <armadillo>

using namespace arma;


class Activation {
public:
	Activation() {};
	virtual ~Activation() {};

	virtual void activate(const vec &z, vec &activation)=0;
	virtual void d_activate(const vec &activation, vec &da)=0;
};

#endif /* ACTIVATION_ACTIVATION_H_ */
