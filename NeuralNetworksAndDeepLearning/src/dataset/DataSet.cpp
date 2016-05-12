/*
 * DataSet.cpp
 *
 *  Created on: 2016. 4. 21.
 *      Author: jhkim
 */

#include "DataSet.h"

#include "../Network.h"
#include "../exception/Exception.h"
#include "DataSample.h"


DataSet::DataSet() {
	trainDataSet = 0;
	validationDataSet = 0;
	testDataSet = 0;
}

DataSet::~DataSet() {
	if(trainDataSet) delete trainDataSet;
	if(validationDataSet) delete validationDataSet;
	if(testDataSet) delete testDataSet;
}



const DataSample &DataSet::getTrainDataAt(int index) {
	if(index >= trainDataSize) throw Exception();
	return trainDataSet[index];
}

const DataSample &DataSet::getValidationDataAt(int index) {
	if(index >= validationDataSize) throw Exception();
	return validationDataSet[index];
}

const DataSample &DataSet::getTestDataAt(int index) {
	if(index >= testDataSize) throw Exception();
	return testDataSet[index];
}






