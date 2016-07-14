/*
 * UbyteDataSet.h
 *
 *  Created on: 2016. 7. 13.
 *      Author: jhkim
 */

#ifndef UBYTEDATASET_H_
#define UBYTEDATASET_H_

#include <armadillo>
#include <vector>
#include <string>

#include "DataSet.h"
#include "DataSample.h"
#include "ImageInfo.h"

using namespace std;
using namespace arma;



class UbyteDataSet : public DataSet {
public:
	UbyteDataSet(const char *train_image, const char *train_label,
			const char *test_image, const char *test_label, double validationSetRatio);
	virtual ~UbyteDataSet();

	virtual void load();
	void shuffleTrainDataSet();
	void shuffleValidationDataSet();
	void shuffleTestDataSet();


#if CPU_MODE
protected:
	int loadDataSetFromResource(string resources[2], DataSample *&dataSet, int offset, int size);
#else
protected:
	int loadDataSetFromResource(const char *data_path, const char *label_path, vector<DATATYPE> *&dataSet, vector<UINT> *&labelSet, int offset, int size);
#endif

protected:
	char train_image[256];
	char train_label[256];
	char test_image[256];
	char test_label[256];
	double validationSetRatio;
};

#endif /* UBYTEDATASET_H_ */
