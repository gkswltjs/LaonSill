/**
 * @file MonitorSet.h
 * @date 2016/4/26
 * @author jhkim
 * @brief
 * @details
 */

#ifndef MONITORSET_H_
#define MONITORSET_H_

#include <vector>

using namespace std;

/**
 * @brief 모티터링 대상이 되는 통계값을 저장하는 저장소 클래스
 * @details 현재 사용하지 않는다.
 * @todo 고정적으로 accuracy, cost 두 가지 통계값을 대상으로 하도록 하고 있음.
 *       범용적인 통계값을 저장할 수 있도록 수정이 필요하다.
 */
class MonitorSet {
public:
	MonitorSet() {}
	virtual ~MonitorSet() {}

	/*
	void add(double validationCost, double validationAccuracy, double trainCost, double trainAccuracy) {
		validationCostSet.push_back(validationCost);
		validationAccuracySet.push_back(validationAccuracy);
		trainCostSet.push_back(trainCost);
		trainAccuracySet.push_back(trainAccuracy);
	}
	*/
	void add(float cost, float accuracy) {
		costSet.push_back(cost);
		accuracySet.push_back(accuracy);
	}

private:
	//vector<double> validationCostSet;
	//vector<double> validationAccuracySet;
	//vector<double> trainCostSet;
	//vector<double> trainAccuracySet;

	vector<float> costSet;
	vector<float> accuracySet;
};

#endif /* MONITORSET_H_ */
