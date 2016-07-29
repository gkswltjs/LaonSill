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
	UbyteDataSet(
			string train_image,
			string train_label,
			int numTrainFile,
			string test_image,
			string test_label,
			int numTestFile,
			int channel=0,
			double validationSetRatio=0.0
			);
	virtual ~UbyteDataSet();

	virtual void load();

	virtual const DATATYPE *getTrainDataAt(int index);
	virtual const UINT *getTrainLabelAt(int index);
	virtual const DATATYPE *getValidationDataAt(int index);
	virtual const UINT *getValidationLabelAt(int index);
	virtual const DATATYPE *getTestDataAt(int index);
	virtual const UINT *getTestLabelAt(int index);


	void shuffleTrainDataSet();
	void shuffleValidationDataSet();
	void shuffleTestDataSet();


#if CPU_MODE
protected:
	int loadDataSetFromResource(string resources[2], DataSample *&dataSet, int offset, int size);
#else
protected:
	int load(int type, int page=0);
	int loadDataSetFromResource(
			string data_path,
			string label_path,
			vector<DATATYPE> *&dataSet,
			vector<UINT> *&labelSet,
			int offset,
			int size);
#endif

protected:
	string train_image;
	string train_label;
	int numTrainFile;
	string test_image;
	string test_label;
	int numTestFile;
	double validationSetRatio;

	int trainFileIndex;
	int testFileIndex;
	int numImagesInFile;

	vector<uint8_t> *bufDataSet;
};

#endif /* UBYTEDATASET_H_ */








































