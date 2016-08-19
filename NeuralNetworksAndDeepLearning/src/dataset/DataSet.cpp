/*
 * DataSet.cpp
 *
 *  Created on: 2016. 8. 16.
 *      Author: jhkim
 */

#include "DataSet.h"

DataSet::DataSet() {
	mean[0] = 0;
	mean[1] = 0;
	mean[2] = 0;

	trainDataSet = NULL;
	trainLabelSet = NULL;
	trainSetIndices = NULL;

	validationDataSet = NULL;
	validationLabelSet = NULL;
	validationSetIndices = NULL;

	testDataSet = NULL;
	testLabelSet = NULL;
	testSetIndices = NULL;
}

DataSet::DataSet(uint32_t rows, uint32_t cols, uint32_t channels, uint32_t numTrainData, uint32_t numTestData) {
	this->rows = rows;
	this->cols = cols;
	this->channels = channels;
	this->dataSize = rows*cols*channels;
	this->numTrainData = numTrainData;
	this->numTestData = numTestData;

	trainDataSet = new vector<DATATYPE>(this->dataSize*numTrainData);
	trainLabelSet = new vector<uint32_t>(numTrainData);
	trainSetIndices = new vector<uint32_t>(numTrainData);
	std::iota(trainSetIndices->begin(), trainSetIndices->end(), 0);

	testDataSet = new vector<DATATYPE>(this->dataSize*numTestData);
	testLabelSet = new vector<uint32_t>(numTestData);
	testSetIndices = new vector<uint32_t>(numTestData);
	iota(testSetIndices->begin(), testSetIndices->end(), 0);

	mean[0] = 0;
	mean[1] = 0;
	mean[2] = 0;
}

DataSet::~DataSet() {
	if(trainDataSet) delete trainDataSet;
	if(trainLabelSet) delete trainLabelSet;
	if(trainSetIndices) delete trainSetIndices;

	if(validationDataSet) delete validationDataSet;
	if(validationLabelSet) delete validationLabelSet;
	if(validationSetIndices) delete validationSetIndices;

	if(testDataSet) delete testDataSet;
	if(testLabelSet) delete testLabelSet;
	if(testSetIndices) delete testSetIndices;
}


void DataSet::setMean(const vector<DATATYPE>& means) {
	for(uint32_t i = 0; i < means.size(); i++) {
		this->mean[i] = means[i];
	}
}


const DATATYPE* DataSet::getTrainDataAt(int index) {
	if(index >= numTrainData) throw Exception();
	return &(*trainDataSet)[dataSize*(*trainSetIndices)[index]];
}

const uint32_t* DataSet::getTrainLabelAt(int index) {
	if(index >= numTrainData) throw Exception();
	return &(*trainLabelSet)[(*trainSetIndices)[index]];
}

const DATATYPE* DataSet::getValidationDataAt(int index) {
	if(index >= numValidationData) throw Exception();
	return &(*validationDataSet)[dataSize*(*validationSetIndices)[index]];
}

const uint32_t* DataSet::getValidationLabelAt(int index) {
	if(index >= numValidationData) throw Exception();
	return &(*validationLabelSet)[(*validationSetIndices)[index]];
}

const DATATYPE* DataSet::getTestDataAt(int index) {
	if(index >= numTestData) throw Exception();
	return &(*testDataSet)[dataSize*(*testSetIndices)[index]];
}

const uint32_t* DataSet::getTestLabelAt(int index) {
	if(index >= numTestData) throw Exception();
	return &(*testLabelSet)[(*testSetIndices)[index]];
}


void DataSet::zeroMean(bool hasMean) {
	//cout << "mean_0: " << mean[0] << ", mean_1: " << mean[1] << ", mean_2: " << mean[2] << endl;
	uint32_t di, ci, hi, wi;
	double sum[3] = {0.0, 0.0, 0.0};

	if(!hasMean) {
		for(di = 0; di < numTrainData; di++) {
			for(ci = 0; ci < channels; ci++) {
				for(hi = 0; hi < rows; hi++) {
					for(wi = 0; wi < cols; wi++) {
						sum[ci] += (*trainDataSet)[wi+hi*cols+ci*cols*rows+di*cols*rows*channels];
					}
				}
			}
			//cout << "mean_0: " << mean[0] << ", mean_1: " << mean[1] << ", mean_2: " << mean[2] << endl;
		}

		cout << "sum_0: " << sum[0] << ", sum_1: " << sum[1] << ", sum_2: " << sum[2] << endl;
		cout << "rows: " << rows << ", cols: " << cols << ", numTrainData: " << numTrainData << endl;
		for(ci = 0; ci < channels; ci++) {
			mean[ci] = (DATATYPE)(sum[ci] / (rows*cols*numTrainData));
		}
		cout << "mean_0: " << mean[0] << ", mean_1: " << mean[1] << ", mean_2: " << mean[2] << endl;
	}

	for(di = 0; di < numTrainData; di++) {
		for(ci = 0; ci < channels; ci++) {
			for(hi = 0; hi < rows; hi++) {
				for(wi = 0; wi < cols; wi++) {
					(*trainDataSet)[wi+hi*cols+ci*cols*rows+di*cols*rows*channels] -= mean[ci];
				}
			}
		}
	}

	for(di = 0; di < numTestData; di++) {
		for(ci = 0; ci < channels; ci++) {
			for(hi = 0; hi < rows; hi++) {
				for(wi = 0; wi < cols; wi++) {
					(*testDataSet)[wi+hi*cols+ci*cols*rows+di*cols*rows*channels] -= mean[ci];
				}
			}
		}
	}
}


void DataSet::shuffleTrainDataSet() {
	random_shuffle(&(*trainSetIndices)[0], &(*trainSetIndices)[numTrainData]);
}

void DataSet::shuffleValidationDataSet() {
	random_shuffle(&(*validationSetIndices)[0], &(*validationSetIndices)[numValidationData]);
}

void DataSet::shuffleTestDataSet() {
	random_shuffle(&(*testSetIndices)[0], &(*testSetIndices)[numTestData]);
}























