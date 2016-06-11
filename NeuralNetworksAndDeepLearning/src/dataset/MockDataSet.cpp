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
	trainDataSize = 10;
	validationDataSize = 10;
	testDataSize = 10;

	trainDataSet = new DataSample[trainDataSize];
	validationDataSet = new DataSample[validationDataSize];
	testDataSet = new DataSample[testDataSize];
}

MockDataSet::~MockDataSet() {
	if(trainDataSet) {
		delete trainDataSet;
		trainDataSet = NULL;
	}
	if(validationDataSet) {
		delete validationDataSet;
		validationDataSet = NULL;
	}
	if(testDataSet) {
		delete testDataSet;
		testDataSet = NULL;
	}
}


void MockDataSet::load() {

	for(int i = 0; i < trainDataSize; i++) {
		trainDataSet[i].init(10, 10, 3, 10);
	}

	for(int i = 0; i < validationDataSize; i++) {
		validationDataSet[i].init(10, 10, 3, 10);
	}

	for(int i = 0; i < testDataSize; i++) {
		testDataSet[i].init(10, 10, 3, 10);
	}


	/*
	double trainData[100] = {
			0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3,
			0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
			0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
			0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0,
			0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0,
			0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3,
			0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
			0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
			0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0,
			0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0,
	};
	int trainTarget[trainDataSize] = {
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9
	};


	for(int i = 0; i < trainDataSize; i++) {
		trainDataSet[i].readData(trainData, trainTarget[i]);
		//trainDataSet.push_back(new DataSample(&trainData[i*9], trainTarget[i]));
		//Util::printVec(trainDataSet[i]->getData(), "data");
		//Util::printVec(trainDataSet[i]->getTarget(), "target");
	}

	for(int i = 0; i < validationDataSize; i++) {
		validationDataSet[i].readData(trainData, trainTarget[i]);
		//validationDataSet.push_back(new DataSample(&trainData[i*9], trainTarget[i]));
	}

	for(int i = 0; i < testDataSize; i++) {
		testDataSet[i].readData(trainData, trainTarget[i]);
		//testDataSet.push_back(new DataSample(&trainData[i*9], trainTarget[i]));
	}
	*/
}




