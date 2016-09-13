/**
 * @file	Top5Evaluation.h
 * @date	2016/7/27
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef TOP5EVALUATION_H_
#define TOP5EVALUATION_H_

#include <cmath>

#include "../Util.h"
#include "Evaluation.h"
#include <limits>



/**
 * @brief 네트워크가 추정한 상위 5개와 정답의 일치율을 평가하는 Evaluation 클래스
 * @details
 */
template <typename Dtype>
class Top5Evaluation : public Evaluation<Dtype> {

public:
	Top5Evaluation() {}
	virtual ~Top5Evaluation() {}

	/*
	virtual void evaluate(const int num_labels, const int batches, const Dtype *output, const uint32_t *y) {
		labelScoreList.resize(num_labels);

		//cout << "Top5Evaluation: " << endl;
		for(int j = 0; j < batches; j++) {
			//cout << "result for batch " << j << "- target: " << y[j] << endl;
			for(int i = 0; i < num_labels; i++) {
				labelScoreList[i].label = i;
				labelScoreList[i].score = output[num_labels*j+i];
			}

			//cout << "prediction: ";
			std::sort(labelScoreList.begin(), labelScoreList.end(), greater<LabelScore>());
			//for(int i = 0; i < labelScoreList.size(); i++) {
			//	cout << i << ": " << labelScoreList[i].label << ", " << labelScoreList[i].score << endl;
			//}

			for(int i = 0; i < 5; i++) {
				//cout << labelScoreList[i].label << ", ";
				if(y[j] == labelScoreList[i].label) {
					this->accurateCount++;
					break;
				}
			}
			//cout << endl;
		}
		//exit(1);
	}
	*/

	virtual void evaluate(const int num_labels, const int batches, const Dtype *output, DataSet<Dtype>* dataSet, const uint32_t baseIndex) {
		labelScoreList.resize(num_labels);
		for(int j = 0; j < batches; j++) {
			for(int i = 0; i < num_labels; i++) {
				labelScoreList[i].label = i;
				labelScoreList[i].score = output[num_labels*j+i];
			}
			std::sort(labelScoreList.begin(), labelScoreList.end(), greater<LabelScore>());
			for(int i = 0; i < 5; i++) {
				if(*dataSet->getTestLabelAt(baseIndex+j) == labelScoreList[i].label) {
					this->accurateCount++;
					break;
				}
			}
		}
	}

private:
	/**
	 * @brief 레이블과 해당 레이블의 예측 스코어를 담기 위한 구조체.
	 * @details 레이블의 예측 순위를 구하기 위해 사용하는 내부 구조체.
	 */
	struct LabelScore {
		uint32_t label;					///< 레이블 번호
		Dtype score;				///< 해당 레이블에 대한 네트워크의 추정 스코어

		LabelScore() {}
		LabelScore(uint32_t label, Dtype score) : label(label), score(score) {}

		/**
		 * @details vector 정렬을 위해 operator>를 재정의
		 */
		bool operator> (const LabelScore& right) const {
			return (this->score > right.score);
		}
	};
	std::vector<LabelScore> labelScoreList;				///< 네트워크가 추정한 순위를 정렬하기 위한 버퍼
};


template class Top5Evaluation<float>;

#endif /* TOP5EVALUATION_H_ */
