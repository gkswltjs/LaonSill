/**
 * @file	Cost.h
 * @date	2016/4/25
 * @author	jhkim
 * @brief
 * @details
 */



#ifndef COST_COST_H_
#define COST_COST_H_

#include "../Util.h"


/**
 * @brief	Cost 구현 클래스의 베이스 추상 클래스.
 * @details	Cost 클래스를 상속받아 Cost를 구현하는 클래스를 생성할 수 있음.
 */
class Cost {
public:
	Cost() {}
	virtual ~Cost() {}

	/**
	 * @brief Cost 타입 열거형 선언.
	 * @details 지원하는 Cost 타입 열거,
	 *          현재 CrossEntropy, LogLikelihood, Quadratic 함수를 지원.
	 */
	enum Type {
		NoCost,				// Cost를 사용하지 않음. Undefined.
		CrossEntropy, 		// Cost에 CrossEntropy 함수를 적용.
		LogLikelihood, 		// Cost에 LogLikelihood 함수를 적용.
		Quadratic			// Cost에 Quadratic 함수를 적용.
	};

	/**
	 * @details 현재 Cost 객체의 타입을 조회.
	 * @return 현재 Cost 객체 타입.
	 */
	Cost::Type getType() const { return type; }

#ifndef GPU_MODE
	virtual double fn(const rvec *pA, const rvec *pY) = 0;
	virtual void d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) = 0;
#else
	/**
	 * @details activation값와 정답값을 이용하여 cost를 계산.
	 * @param pA activation값 장치 메모리 포인터.
	 * @param pY 정답값 장치 메모리 포인터.
	 * @return 계산된 cost값.
	 * @todo 사용중이지 않기 때문에 사용시 정비가 필요, 현재 이 method의 결과는 무효함.
	 */
	virtual double fn(const DATATYPE *pA, const DATATYPE *pY) = 0;

	/**
	 * @details cost에 대해 activation 입력으로 미분 결과를 계산
	 * @param z activation 입력값 장치 메모리 포인터.
	 * @param activation activation 결과값 장치 메모리 포인터.
	 * @param target 학습 데이터에 대한 정답값 장치 메모리 포인터.
	 * @param delta cost에 대해 acitvation 입력값(z)으로 미분 결과값 장치 메모리 포인터.
	 * @param numLabels 정답값의 레이블 수.
	 * @param batchsize 데이터 배치 수.
	 */
	virtual void d_cost(const DATATYPE *z, const DATATYPE *activation, const UINT *target, DATATYPE *delta, UINT numLabels, UINT batchsize) = 0;
#endif

protected:
	Cost::Type type;				///< 현재 Cost 객체의 Cost 타입.
};

#endif /* COST_COST_H_ */
