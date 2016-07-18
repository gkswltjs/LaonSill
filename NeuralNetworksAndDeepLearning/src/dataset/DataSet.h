/*
 * DataSet.h
 *
 *  Created on: 2016. 4. 21.
 *      Author: jhkim
 */

#ifndef DATASET_DATASET_H_
#define DATASET_DATASET_H_

#include <armadillo>
#include <vector>
#include <string>
#include "ImageInfo.h"
#include "DataSample.h"
#include "../exception/Exception.h"

using namespace std;
using namespace arma;


class DataSet {
public:
	DataSet() {
		mean_r = 0;
		mean_g = 0;
		mean_b = 0;
	}
	DataSet(UINT rows, UINT cols, UINT channels, UINT numTrainData, UINT numTestData) {
		this->rows = rows;
		this->cols = cols;
		this->channels = channels;
		this->dataSize = rows*cols*channels;
		this->numTrainData = numTrainData;
		this->numTestData = numTestData;

		trainDataSet = new vector<DATATYPE>(this->dataSize*numTrainData);
		trainLabelSet = new vector<UINT>(numTrainData);
		testDataSet = new vector<DATATYPE>(this->dataSize*numTestData);
		testLabelSet = new vector<UINT>(numTestData);

		mean_r = 0;
		mean_g = 0;
		mean_b = 0;
	}
	virtual ~DataSet() {
		if(trainDataSet) delete trainDataSet;
		if(trainLabelSet) delete trainLabelSet;
		if(validationDataSet) delete validationDataSet;
		if(validationLabelSet) delete validationLabelSet;
		if(testDataSet) delete testDataSet;
		if(testLabelSet) delete testLabelSet;
	}

	UINT getRows() const { return this->rows; }
	UINT getCols() const { return this->cols; }
	UINT getChannels() const { return this->channels; }
	UINT getNumTrainData() const { return this->numTrainData; }
	UINT getNumValidationData() const { return this->numValidationData; }
	UINT getNumTestData() const { return this->numTestData; }

	const DATATYPE *getTrainDataAt(int index) {
		if(index >= numTrainData) throw Exception();
		return &(*trainDataSet)[dataSize*index];
	}
	const UINT *getTrainLabelAt(int index) {
		if(index >= numTrainData) throw Exception();
		return &(*trainLabelSet)[index];
	}
	const DATATYPE *getValidationDataAt(int index) {
		if(index >= numValidationData) throw Exception();
		return &(*validationDataSet)[dataSize*index];
	}
	const UINT *getValidationLabelAt(int index) {
		if(index >= numValidationData) throw Exception();
		return &(*validationLabelSet)[index];
	}
	const DATATYPE *getTestDataAt(int index) {
		if(index >= numTestData) throw Exception();
		return &(*testDataSet)[dataSize*index];
	}
	const UINT *getTestLabelAt(int index) {
		if(index >= numTestData) throw Exception();
		return &(*testLabelSet)[index];
	}

	const vector<DATATYPE> *getTrainDataSet() const { return this->trainDataSet; }
	const vector<DATATYPE> *getValidationDataSet() const { return this->validationDataSet; }
	const vector<DATATYPE> *getTestDataSet() const { return this->testDataSet; }

	virtual void load() = 0;
	virtual void shuffleTrainDataSet() = 0;
	virtual void shuffleValidationDataSet() = 0;
	virtual void shuffleTestDataSet() = 0;
	void zeroMean() {
		cout << "mean_r: " << mean_r << ", mean_g: " << mean_g << ", mean_b: " << mean_b << endl;

		UINT i;
		for(i = 0; i < numTrainData; i++) {
			mean_r += (*trainDataSet)[i*channels+0];
			mean_g += (*trainDataSet)[i*channels+1];
			mean_b += (*trainDataSet)[i*channels+2];
		}


		mean_r /= numTrainData;
		mean_g /= numTrainData;
		mean_b /= numTrainData;

		cout << "mean_r: " << mean_r << ", mean_g: " << mean_g << ", mean_b: " << mean_b << endl;

		for(i = 0; i < numTrainData; i++) {
			(*trainDataSet)[i*channels+0] -= mean_r;
			(*trainDataSet)[i*channels+1] -= mean_g;
			(*trainDataSet)[i*channels+2] -= mean_b;
		}
	}

private:



protected:
	UINT rows;
	UINT cols;
	UINT channels;
	UINT dataSize;
	UINT numTrainData;
	UINT numValidationData;
	UINT numTestData;



	//DataSample *trainDataSet;
	//DataSample *validationDataSet;
	//DataSample *testDataSet;
	vector<DATATYPE> *trainDataSet;
	vector<UINT> *trainLabelSet;
	vector<DATATYPE> *validationDataSet;
	vector<UINT> *validationLabelSet;
	vector<DATATYPE> *testDataSet;
	vector<UINT> *testLabelSet;

	DATATYPE mean_r;
	DATATYPE mean_g;
	DATATYPE mean_b;

	//vector<const DataSample *> trainDataSet;
	//vector<const DataSample *> validationDataSet;
	//vector<const DataSample *> testDataSet;

};

#endif /* DATASET_DATASET_H_ */
