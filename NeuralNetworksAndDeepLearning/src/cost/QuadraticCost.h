/**
 * @file	QuadraticCost.h
 * @date	2016/4/25
 * @author	jhkim
 * @brief
 * @details
 */


#ifndef COST_QUADRATICCOST_H_
#define COST_QUADRATICCOST_H_


#include "../common.h"
#include "Cost.h"



template <typename Dtype>
class QuadraticCost : public Cost<Dtype> {
public:
	QuadraticCost() {
		this->type = Cost<Dtype>::Quadratic;
	}
	virtual ~QuadraticCost() {}

#ifndef GPU_MODE
public:
	double fn(const rvec *pA, const rvec *pY) {
		return 0.5*sum(square(*pA - *pY));
	}
	void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) {
		delta.slice(0) = activation.slice(0) - target;
	}
#else
	double forward(const Dtype* output, const uint32_t* target, const uint32_t numLabels, const uint32_t batchsize) {
		//return 0.5*sum(square(*pA - *pY));
		return 0.0;
	}
	void backward(const Dtype* z, const Dtype* activation, const uint32_t* target, Dtype* delta, uint32_t numLabels, uint32_t batchsize) {
		/*
		checkCudaErrors(cudaMemcpyAsync(delta, activation, sizeof(Dtype)*size, cudaMemcpyDeviceToDevice));
		uint32_t i;
		for(i = 0; i < size/numLabels; i++) {
			delta[i*numLabels+target[i]]-=1;
		}
		*/
	}
#endif

};

template class QuadraticCost<float>;

#endif /* COST_QUADRATICCOST_H_ */
