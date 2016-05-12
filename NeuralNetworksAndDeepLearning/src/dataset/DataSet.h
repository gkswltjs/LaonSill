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

class DataSample;

using namespace std;
using namespace arma;






class DataSet {
public:
	DataSet();
	virtual ~DataSet();

	int getTrainDataSize() const { return this->trainDataSize; }
	int getValidationDataSize() const { return this->validationDataSize; }
	int getTestDataSize() const { return this->testDataSize; }

	const DataSample &getTrainDataAt(int index);
	const DataSample &getValidationDataAt(int index);
	const DataSample &getTestDataAt(int index);

	const DataSample *getTrainDataSet() const { return this->trainDataSet; }
	const DataSample *getValidationDataSet() const { return this->validationDataSet; }
	const DataSample *getTestDataSet() const { return this->testDataSet; }

	virtual void load() = 0;
	virtual void shuffleTrainDataSet() = 0;
	virtual void shuffleValidationDataSet() = 0;
	virtual void shuffleTestDataSet() = 0;

private:



protected:
	int trainDataSize;
	int validationDataSize;
	int testDataSize;

	DataSample *trainDataSet;
	DataSample *validationDataSet;
	DataSample *testDataSet;

	//vector<const DataSample *> trainDataSet;
	//vector<const DataSample *> validationDataSet;
	//vector<const DataSample *> testDataSet;

};

#endif /* DATASET_DATASET_H_ */
