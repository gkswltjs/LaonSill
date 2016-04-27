/*
 * MonitorSet.h
 *
 *  Created on: 2016. 4. 26.
 *      Author: jhkim
 */

#ifndef MONITORSET_H_
#define MONITORSET_H_

#include <vector>

using namespace std;


class MonitorSet {
public:
	MonitorSet() {}
	virtual ~MonitorSet() {}

	void add(double validationCost, double validationAccuracy, double trainCost, double trainAccuracy) {
		validationCostSet.push_back(validationCost);
		validationAccuracySet.push_back(validationAccuracy);
		trainCostSet.push_back(trainCost);
		trainAccuracySet.push_back(trainAccuracy);
	}

private:
	vector<double> validationCostSet;
	vector<double> validationAccuracySet;
	vector<double> trainCostSet;
	vector<double> trainAccuracySet;
};

#endif /* MONITORSET_H_ */
