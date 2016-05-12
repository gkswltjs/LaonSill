/*
 * MnistDataSet.h
 *
 *  Created on: 2016. 4. 23.
 *      Author: jhkim
 */

#ifndef MNISTDATASET_DATASET_H_
#define MNISTDATASET_DATASET_H_

#include <armadillo>
#include <vector>
#include <string>

#include "DataSet.h"
#include "DataSample.h"
#include "ImageInfo.h"

using namespace std;
using namespace arma;



class MnistDataSet : public DataSet {
public:
	MnistDataSet(double validationSetRatio);
	virtual ~MnistDataSet();

	virtual void load();
	void shuffleTrainDataSet();
	void shuffleValidationDataSet();
	void shuffleTestDataSet();


private:
	int loadDataSetFromResource(string resources[2], DataSample *&dataSet, int offset, int size);
	//int loadDataSetFromResource(string resources[2], vector<const DataSample *> &dataSet);

	double validationSetRatio;
};

#endif /* MNISTDATASET_DATASET_H_ */
















