/*
 * DataSet.cpp
 *
 *  Created on: 2016. 4. 21.
 *      Author: jhkim
 */

#include "DataSet.h"

#include "../Network.h"
#include "DataSample.h"


DataSet::DataSet() {}
DataSet::~DataSet() {}



const DataSample *DataSet::getTrainDataAt(int index) {
	if(index >= trainDataSize) return 0;
	return trainDataSet[index];
}

const DataSample *DataSet::getValidationDataAt(int index) {
	if(index >= validationDataSize) return 0;
	return validationDataSet[index];
}

const DataSample *DataSet::getTestDataAt(int index) {
	if(index >= testDataSize) return 0;
	return testDataSet[index];
}






