/**
 * @file	LogLikelihoodCost.h
 * @date	2016/5/12
 * @author	jhkim
 * @brief
 * @details
 */


#ifndef COST_LOGLIKELIHOODCOST_H_
#define COST_LOGLIKELIHOODCOST_H_

#include "Cuda.h"

#include "common.h"
#include "Cost.h"


/**
 * @brief 주어진 결과값으로 LogLikeliCost를 구한다.
 * @details 최종 Cost를 C, Output값을 a, 정답 레이블을 y, L을 마지막 레이어 인덱스라고 할 때
 *          Cost C는 다음과 같이 계산된다.
 *          C = -ln(ayL)
 *          즉, 최종 레이어 출력값 중 정답레이블에 해당하는 element에 대한 negative log값이다.
 */
template <typename Dtype>
class LogLikelihoodCost : public Cost<Dtype> {
public:
	LogLikelihoodCost();
	virtual ~LogLikelihoodCost();

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


#endif /* COST_LOGLIKELIHOODCOST_H_ */


