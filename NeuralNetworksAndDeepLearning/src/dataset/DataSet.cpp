/*
 * DataSet.cpp
 *
 *  Created on: 2016. 8. 16.
 *      Author: jhkim
 */

#include "DataSet.h"

using namespace std;

template <typename Dtype>
DataSet<Dtype>::DataSet() {
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

template <typename Dtype>
DataSet<Dtype>::DataSet(uint32_t rows, uint32_t cols, uint32_t channels, uint32_t numTrainData, uint32_t numTestData) {
	this->rows = rows;
	this->cols = cols;
	this->channels = channels;
	this->dataSize = rows*cols*channels;
	this->numTrainData = numTrainData;
	this->numTestData = numTestData;

	trainDataSet = new vector<Dtype>(this->dataSize*numTrainData);
	trainLabelSet = new vector<Dtype>(numTrainData);
	trainSetIndices = new vector<uint32_t>(numTrainData);
	iota(trainSetIndices->begin(), trainSetIndices->end(), 0);

	testDataSet = new vector<Dtype>(this->dataSize*numTestData);
	testLabelSet = new vector<Dtype>(numTestData);
	testSetIndices = new vector<uint32_t>(numTestData);
	iota(testSetIndices->begin(), testSetIndices->end(), 0);

	mean[0] = 0;
	mean[1] = 0;
	mean[2] = 0;
}

template <typename Dtype>
DataSet<Dtype>::~DataSet() {
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

template <typename Dtype>
void DataSet<Dtype>::setMean(const vector<Dtype>& means) {
	for(uint32_t i = 0; i < means.size(); i++) {
		this->mean[i] = means[i];
	}
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getTrainDataAt(int index) {
	if(index >= numTrainData) {
		cout << "train data index over numTrainData ... " << endl;
		exit(1);
	}
	return &(*trainDataSet)[dataSize*(*trainSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getTrainLabelAt(int index) {
	if(index >= numTrainData) {
		cout << "train label index over numTrainData ... " << endl;
		exit(1);
	}
	return &(*trainLabelSet)[(*trainSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getValidationDataAt(int index) {
	if(index >= numValidationData) {
		cout << "validation data index over numValidationData ... " << endl;
		exit(1);
	}
	return &(*validationDataSet)[dataSize*(*validationSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getValidationLabelAt(int index) {
	if(index >= numValidationData) {
		cout << "validation label index over numValidationData ... " << endl;
		exit(1);
	}
	return &(*validationLabelSet)[(*validationSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getTestDataAt(int index) {
	if(index >= numTestData) {
		cout << "test data index over numTestData ... " << endl;
		exit(1);
	}
	return &(*testDataSet)[dataSize*(*testSetIndices)[index]];
}

template <typename Dtype>
const Dtype* DataSet<Dtype>::getTestLabelAt(int index) {
	if(index >= numTestData) {
		cout << "test label index over numTestData ... " << endl;
		exit(1);
	}
	return &(*testLabelSet)[(*testSetIndices)[index]];
}

template <typename Dtype>
void DataSet<Dtype>::zeroMean(bool hasMean) {
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
			mean[ci] = (Dtype)(sum[ci] / (rows*cols*numTrainData));
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

template <typename Dtype>
void DataSet<Dtype>::shuffleTrainDataSet() {
	random_shuffle(&(*trainSetIndices)[0], &(*trainSetIndices)[numTrainData]);
}

template <typename Dtype>
void DataSet<Dtype>::shuffleValidationDataSet() {
	random_shuffle(&(*validationSetIndices)[0], &(*validationSetIndices)[numValidationData]);
}

template <typename Dtype>
void DataSet<Dtype>::shuffleTestDataSet() {
	random_shuffle(&(*testSetIndices)[0], &(*testSetIndices)[numTestData]);
}


template class DataSet<float>;















