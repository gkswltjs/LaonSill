/**
 * @file	Evaluation.h
 * @date	2016/7/27
 * @author	jhkim
 * @brief
 * @details
 */





#ifndef EVALUATION_H_
#define EVALUATION_H_


#include "../dataset/DataSet.h"


/**
 * @brief Evaluation 최상위 추상클래스
 * @details 네트워크 추정과 정답을 이용하여 적절한 결과 통계값을 구하고자 할 때
 *          해당 클래스를 상속하여 기능을 구현한다.
 *          결과 통계값에는 예측 정확도, Cost값 등이 있을 수 있다.
 * @todo accurateCoutn, cost와 같은 구체적인 통계값에 대한 의존성을 구현 클래스로 옮겨야 한다.
 */
template <typename Dtype>
class Evaluation {
public:
	Evaluation() {
		this->accurateCount = 0;
		this->cost = 0.0f;
	}
	virtual ~Evaluation() {}

	int getAccurateCount() const { return accurateCount; }
	double getCost() const { return cost; }

	/**
	 * @details 내부 멤버 변수들을 초기화한다.
	 */
	void reset() {
		this->accurateCount = 0;
		this->cost = 0.0f;
	}

	/**
	 * @details 네트워크 추정과 정답을 전달받아 원하는 결과 통계값을 구하는 추상 메쏘드.
	 * @param num_labels 정답의 레이블 수 (카테고리 수)
	 * @param batches 네트워크 추정시 사용한 배치의 수
	 * @param output 네트워크가 추정 호스트 메모리 포인터
	 * @param y 정답값 호스트 메모리 포인터
	 */
	//virtual void evaluate(const int num_labels, const int batches, const Dtype* output, const uint32_t* y)=0;
	virtual void evaluate(const int num_labels, const int batches, const Dtype* output, DataSet<Dtype>* dataSet, const uint32_t baseIndex)=0;

protected:
	int accurateCount;			///< 네트워크 추정에서 정답을 맞춘 갯수
	double cost;				///< 네트워크 추정의 전체 Cost의 합
};

template class Evaluation<float>;

#endif /* EVALUATION_H_ */
