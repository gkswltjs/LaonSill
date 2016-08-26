/**
 * @file	QuadraticCost.h
 * @date	2016/4/25
 * @author	jhkim
 * @brief
 * @details
 */


#ifndef COST_QUADRATICCOST_H_
#define COST_QUADRATICCOST_H_



#include "Cost.h"




class QuadraticCost : public Cost {
public:
	QuadraticCost() {
		this->type = Cost::Quadratic;
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
	double forward(const DATATYPE* output, const uint32_t* target, const uint32_t numLabels, const uint32_t batchsize) {
		//return 0.5*sum(square(*pA - *pY));
		return 0.0;
	}
	void backward(const DATATYPE *z, const DATATYPE *activation, const UINT *target, DATATYPE *delta, UINT numLabels, UINT batchsize) {
		/*
		Cuda::refresh();
		checkCudaErrors(cudaMemcpyAsync(delta, activation, sizeof(DATATYPE)*size, cudaMemcpyDeviceToDevice));
		UINT i;
		for(i = 0; i < size/numLabels; i++) {
			delta[i*numLabels+target[i]]-=1;
		}
		*/
	}
#endif

};

#endif /* COST_QUADRATICCOST_H_ */
