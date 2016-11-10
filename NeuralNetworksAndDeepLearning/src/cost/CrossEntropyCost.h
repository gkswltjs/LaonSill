/**
 * @file	CostEntropyCost.h
 * @date	2016/4/25
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef COST_CROSSENTROPYCOST_H_
#define COST_CROSSENTROPYCOST_H_

#include "../common.h"
#include "Cost.h"

/**
 * @details
 */
template <typename Dtype>
class CrossEntropyCost : public Cost<Dtype> {

public:
	CrossEntropyCost() {
		this->type = Cost<Dtype>::CrossEntropy;
	}
	virtual ~CrossEntropyCost() {}

#ifndef GPU_MODE
public:
	double fn(const rvec *pA, const rvec *pY) {
		/*
		Util::printVec(pA, "activation");
		Util::printVec(pY, "y");

		rvec left = (-1 * (*pY)) % log(*pA);
		rvec right = (1 - (*pY)) % log(1 - (*pA));
		rvec result = left - right;
		Util::printVec(&left, "left");
		Util::printVec(&right, "right");
		Util::printVec(&result, "result");
		return sum(left - right);
		*/
		return 0.0;
	}

	void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) {
		//Util::printVec(z, "activation");
		//Util::printVec(activation, "y");
		delta.slice(0) = activation.slice(0) - target;
		//Util::printVec(delta, "result");
	}
#else
	double forward(const Dtype* output, const Dtype* target,
			const uint32_t numLabels, const uint32_t batchsize) {
		return 0.0;
	}

	void backward(const Dtype* z, const Dtype* activation, const Dtype* target,
			Dtype* delta, uint32_t numLabels, uint32_t size) {
		checkCudaErrors(cudaMemcpyAsync(delta, activation, sizeof(Dtype)*size, cudaMemcpyDeviceToDevice));
		uint32_t i;
		uint32_t label;
		for(i = 0; i < size/numLabels; i++) {
			label = static_cast<uint32_t>(target[i]+0.1);
			delta[i*numLabels+label]-=1;
		}
	}
#endif
};

template class CrossEntropyCost<float>;

#endif /* COST_CROSSENTROPYCOST_H_ */
