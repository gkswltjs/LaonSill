/**
 * @file	Top1Evaluation.h
 * @date	2016/7/27
 * @author	jhkim
 * @brief
 * @details
 */


#ifndef TOP1EVALUATION_H_
#define TOP1EVALUATION_H_

#include "Evaluation.h"
#include <limits>

/**
 * @brief 네트워크 1위 추정과 정답의 일치율을 평가하는 Evaluation 클래스
 * @details
 */
template <typename Dtype>
class Top1Evaluation : public Evaluation<Dtype> {
public:
	Top1Evaluation() {}
	virtual ~Top1Evaluation() {}

	/*
	virtual void evaluate(const int num_labels, const int batches, const Dtype *output, const uint32_t *y) {
		//cout << "Top1Evaluation: " << endl;
		for(int j = 0; j < batches; j++) {
			Dtype maxValue = -std::numeric_limits<Dtype>::max();
			int maxIndex = 0;
			for(int i = 0; i < num_labels; i++) {
				if(output[num_labels*j+i] > maxValue) {
					maxValue = output[num_labels*j+i];
					maxIndex = i;
				}
				// cost
				if(i == y[j]) this->cost += std::abs(output[num_labels*j+i]-1);
				else this->cost += std::abs(output[num_labels*j+i]);
			}
			if(maxIndex == y[j]) this->accurateCount++;
		}
	}
	*/

	virtual void evaluate(const int num_labels, const int batches, const Dtype *output, DataSet<Dtype>* dataSet, const uint32_t baseIndex) {
		//cout << "Top1Evaluation: " << endl;
		for(int j = 0; j < batches; j++) {
			Dtype maxValue = -std::numeric_limits<Dtype>::max();
			int maxIndex = 0;
			for(int i = 0; i < num_labels; i++) {
				if(output[num_labels*j+i] > maxValue) {
					maxValue = output[num_labels*j+i];
					maxIndex = i;
				}
				// cost
				if(i == *dataSet->getTestLabelAt(baseIndex+j)) this->cost += std::abs(output[num_labels*j+i]-1);
				else this->cost += std::abs(output[num_labels*j+i]);
			}
			if(maxIndex == *dataSet->getTestLabelAt(baseIndex+j)) this->accurateCount++;
		}
	}
};


template class Top1Evaluation<float>;

#endif /* TOP1EVALUATION_H_ */
