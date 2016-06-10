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

enum class PoolingType {
	None, Max, Avg
};

class Pooling {
public:
	Pooling() {}
	virtual ~Pooling() {}
	PoolingType getType() const { return this->type; }

	virtual void pool(const pool_dim &pool_d, const rcube &input, ucube &pool_map, rcube &output)=0;
	virtual void d_pool(const pool_dim &pool_d, const rcube &input, ucube &pool_map, rcube &output)=0;

protected:
	PoolingType type;


};

#endif /* POOLING_POOLING_H_ */
