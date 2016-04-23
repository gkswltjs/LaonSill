/*
 * MockDataSet.cpp
 *
 *  Created on: 2016. 4. 23.
 *      Author: jhkim
 */

#include "MockDataSet.h"
#include "DataSample.h"
#include "../Util.h"

MockDataSet::MockDataSet() {
	// TODO Auto-generated constructor stub

}

MockDataSet::~MockDataSet() {
	// TODO Auto-generated destructor stub
}


void MockDataSet::load() {

	trainDataSize = 10;
	testDataSize = 10;

	double trainData[9*trainDataSize] = {
			0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4,
			0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
			0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
			0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.1,
			0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2,
			0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4,
			0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
			0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
			0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.1,
			0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2,
	};
	int trainTarget[trainDataSize] = {
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9
	};


	for(int i = 0; i < trainDataSize; i++) {
		trainDataSet.push_back(new DataSample(&trainData[i*9], trainTarget[i]));

		Util::printVec(trainDataSet[i]->getData(), "data");
		Util::printVec(trainDataSet[i]->getTarget(), "target");
	}

	for(int i = 0; i < testDataSize; i++) {
		testDataSet.push_back(new DataSample(&trainData[i*9], trainTarget[i]));
	}



}

void MockDataSet::shuffleTrainDataSet() {


}
