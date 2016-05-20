/*
 * Cifar10DataSet.h
 *
 *  Created on: 2016. 5. 19.
 *      Author: jhkim
 */

#ifndef DATASET_CIFAR10DATASET_H_
#define DATASET_CIFAR10DATASET_H_

#include <armadillo>
#include <vector>
#include <string>

#include "DataSet.h"
#include "DataSample.h"
#include "ImageInfo.h"

using namespace std;
using namespace arma;



class Cifar10DataSet : public DataSet {
public:
	Cifar10DataSet();
	virtual ~Cifar10DataSet();

	virtual void load();
	void shuffleTrainDataSet();
	void shuffleValidationDataSet();
	void shuffleTestDataSet();


private:
	int loadDataSetFromResource(string *resources, int numResources, DataSample *&dataSet, int dataSize);
	double validationSetRatio;
};

#endif /* DATASET_CIFAR10DATASET_H_ */
