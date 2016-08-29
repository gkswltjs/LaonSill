/*
 * LearnableLayer.h
 *
 *  Created on: 2016. 8. 20.
 *      Author: jhkim
 */

#ifndef LEARNABLELAYER_H_
#define LEARNABLELAYER_H_

#include "../Util.h"

/**
 * @brief 학습하는 레이어에서 구현해야하는 베이스 추상 클래스,
 *        인터페이스 역할을 한다.
 */
class LearnableLayer {
public:
	virtual ~LearnableLayer() {}

	/**
	 * @details 학습한 파라미터 그레디언트를 파라미터에 업데이트한다.
	 */
	virtual void update() = 0;
	/**
	 * @details 파라미터들의 제곱의 합을 구한다.
	 * @return 파라미터들의 제곱의 합
	 */
	virtual double sumSquareParamsData() = 0;
	/**
	 * @details 파라미터 그레디언트들의 제곱의 합을 구한다.
	 * @return 파라미터 그레디언트들의 제곱의 합
	 */
	virtual double sumSquareParamsGrad() = 0;
	/**
	 * @details 파라미터 그레디언트를 스케일링한다.
	 * @param 파라미터 그레디언트를 스케일링할 스케일 값
	 */
	virtual void scaleParamsGrad(float scale) = 0;
};



#endif /* LEARNABLELAYER_H_ */
