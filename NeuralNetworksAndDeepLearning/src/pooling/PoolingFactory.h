/*
 * PoolingFactory.h
 *
 *  Created on: 2016. 6. 7.
 *      Author: jhkim
 */

#ifndef POOLING_POOLINGFACTORY_H_
#define POOLING_POOLINGFACTORY_H_

#include "AvgPooling.h"
#include "MaxPooling.h"





class PoolingFactory {
public:
	PoolingFactory() {}
	virtual ~PoolingFactory() {}

#if CPU_MODE
	static Pooling *create(PoolingType poolingType) {
		switch(poolingType) {
		case PoolingType::Max: return new MaxPooling();
		case PoolingType::Avg: return new AvgPooling();
		case PoolingType::None:
		default: return 0;
		}
	}
#else
	static Pooling *create(PoolingType poolingType, pool_dim pool_d) {
		switch(poolingType) {
		case PoolingType::Max: return new MaxPooling(pool_d);
		case PoolingType::Avg: return new AvgPooling(pool_d);
		case PoolingType::None:
		default: return 0;
		}
	}
#endif

	static void destroy(Pooling *&pooling_fn) {
		if(pooling_fn) {
			delete pooling_fn;
			pooling_fn = NULL;
		}
	}
};

#endif /* POOLING_POOLINGFACTORY_H_ */
