/**
 * @file DQNCost.h
 * @date 2016-12-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DQNCOST_H
#define DQNCOST_H 

#include "Cuda.h"

#include "common.h"
#include "Cost.h"

template<typename Dtype>
class DQNCost : public Cost<Dtype> {
public: 
    DQNCost() {}
    virtual ~DQNCost() {}

#ifndef GPU_MODE
	double fn(const rvec *pA, const rvec *pY);
	void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta);
#else
	/**
	 * @details C = -ln(ayL)
	 */
	double forward(const Dtype* output, const Dtype* target,
			const uint32_t numLabels, const uint32_t batchsize);
	/**
	 * @details dC/da = -1/ayL
	 */
	void backward(const Dtype* z, const Dtype* activation,
			const Dtype* target, Dtype* delta, uint32_t numLabels, uint32_t batchsize);
#endif

};
#endif /* DQNCOST_H */
