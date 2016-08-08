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

/**
 * @brief 네트워크 1위 추정과 정답의 일치율을 평가하는 Evaluation 클래스
 * @details
 */
class Top1Evaluation : public Evaluation {
public:
	Top1Evaluation() {}
	virtual ~Top1Evaluation() {}

	virtual void evaluate(const int num_labels, const int batches, const DATATYPE *output, const UINT *y) {
		//cout << "Top1Evaluation: " << endl;
		for(int j = 0; j < batches; j++) {
			DATATYPE maxValue = -100000.0;
			int maxIndex = 0;
			for(int i = 0; i < num_labels; i++) {
				if(output[num_labels*j+i] > maxValue) {
					maxValue = output[num_labels*j+i];
					maxIndex = i;
				}
				// cost
				if(i == y[j]) cost += std::abs(output[num_labels*j+i]-1);
				else cost += std::abs(output[num_labels*j+i]);
			}
			if(maxIndex == y[j]) accurateCount++;
			//cout << "result for batch " << j << "- target: " << y[j] << ", prediction: " << maxIndex << ", ac cost:" << cost << ", cost:" << cost/(j+1) << endl;
		}
		/*
		Util::setPrint(true);
		Util::printData(output, num_labels, batches, 1, 1, "output:");
		Util::setPrint(false);
		for(int i = 0; i < batches; i++) {
			cout << y[i] << endl;
		}
		exit(1);
		*/
	}

};

#endif /* TOP1EVALUATION_H_ */
