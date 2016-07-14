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

#if CPU_MODE
	virtual void pool(const pool_dim &pool_d, const rcube &input, ucube &pool_map, rcube &output)=0;
	virtual void d_pool(const pool_dim &pool_d, const rcube &input, ucube &pool_map, rcube &output)=0;
#else
	cudnnPoolingDescriptor_t getPoolDesc() const { return poolDesc; }

	virtual void pool(const cudnnTensorDescriptor_t xDesc, const DATATYPE *x,
			const cudnnTensorDescriptor_t yDesc, DATATYPE *y)=0;
	virtual void d_pool(const cudnnTensorDescriptor_t yDesc, const DATATYPE *y, const DATATYPE *dy,
			const cudnnTensorDescriptor_t xDesc, const DATATYPE *x, DATATYPE *dx)=0;

#endif

protected:
	PoolingType type;
	cudnnPoolingDescriptor_t poolDesc;
	const float alpha = 1.0f, beta = 0.0f;

};

#endif /* POOLING_POOLING_H_ */
