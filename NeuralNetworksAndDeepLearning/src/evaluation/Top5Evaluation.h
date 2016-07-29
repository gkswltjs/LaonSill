/*
 * Top5Evaluation.h
 *
 *  Created on: 2016. 7. 27.
 *      Author: jhkim
 */

#ifndef TOP5EVALUATION_H_
#define TOP5EVALUATION_H_

#include <cmath>

#include "../Util.h"
#include "Evaluation.h"









class Top5Evaluation : public Evaluation {

public:
	Top5Evaluation() {}
	virtual ~Top5Evaluation() {}

	virtual void evaluate(const int num_labels, const int batches, const DATATYPE *output, const UINT *y) {
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
					accurateCount++;
					break;
				}
			}
			//cout << endl;
		}
		//exit(1);
	}

private:
	struct LabelScore {
		UINT label;
		DATATYPE score;

		LabelScore() {}
		LabelScore(UINT label, DATATYPE score) : label(label), score(score) {}

		bool operator> (const LabelScore& right) const {
			return (this->score > right.score);
		}
	};
	std::vector<LabelScore> labelScoreList;
};

#endif /* TOP5EVALUATION_H_ */
