/*
 * MockDataSet.h
 *
 *  Created on: 2016. 4. 23.
 *      Author: jhkim
 */

#ifndef DATASET_MOCKDATASET_H_
#define DATASET_MOCKDATASET_H_

#include "DataSet.h"

class MockDataSet : public DataSet {
public:
	MockDataSet(UINT rows, UINT cols, UINT channels, UINT numTrainData, UINT numTestData, UINT numLabels);
	virtual ~MockDataSet();

	virtual void load();
	void shuffleTrainDataSet() {}
	void shuffleValidationDataSet() {}
	void shuffleTestDataSet() {}

private:
	UINT numLabels;

};

#endif /* DATASET_MOCKDATASET_H_ */
