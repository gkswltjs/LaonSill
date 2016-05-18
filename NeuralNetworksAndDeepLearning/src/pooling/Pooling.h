/*
 * Pooling.h
 *
 *  Created on: 2016. 5. 16.
 *      Author: jhkim
 */

#ifndef POOLING_POOLING_H_
#define POOLING_POOLING_H_


#include <armadillo>

using namespace arma;


class Pooling {
public:
	Pooling() {}
	virtual ~Pooling() {}

	virtual void pool(const pool_dim &pool_d, const cube &input, ucube &pool_map, cube &output)=0;
	virtual void d_pool(const pool_dim &pool_d, const cube &input, ucube &pool_map, cube &output)=0;

};

#endif /* POOLING_POOLING_H_ */
