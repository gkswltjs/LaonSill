/*
 * MnistDataSet.h
 *
 *  Created on: 2016. 4. 23.
 *      Author: jhkim
 */

#ifndef MNISTDATASET_DATASET_H_
#define MNISTDATASET_DATASET_H_

#include <string>
#include <vector>

#include "UbyteDataSet.h"

using namespace std;
using namespace arma;



class MnistDataSet : public UbyteDataSet {
public:
	MnistDataSet(double validationSetRatio) :
		UbyteDataSet(
				"/home/jhkim/data/learning/mnist/train-images.idx3-ubyte",
				"/home/jhkim/data/learning/mnist/train-labels.idx1-ubyte",
				"/home/jhkim/data/learning/mnist/t10k-images.idx3-ubyte",
				"/home/jhkim/data/learning/mnist/t10k-labels.idx1-ubyte",
				validationSetRatio) {
		this->channels = 1;
	}
	virtual ~MnistDataSet() {}

	/*
	virtual void load();
	void shuffleTrainDataSet();
	void shuffleValidationDataSet();
	void shuffleTestDataSet();
	*/

	/*
private:
#if CPU_MODE
	int loadDataSetFromResource(string resources[2], DataSample *&dataSet, int offset, int size);
#else
	int loadDataSetFromResource(string resources[2], vector<DATATYPE> *&dataSet, vector<UINT> *&labelSet, int offset, int size);
#endif

	double validationSetRatio;
	*/
};

#endif /* MNISTDATASET_DATASET_H_ */
















