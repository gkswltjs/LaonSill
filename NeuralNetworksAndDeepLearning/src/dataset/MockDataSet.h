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
	MockDataSet();
	virtual ~MockDataSet();

	virtual void load();
	virtual void shuffleTrainDataSet();
};

#endif /* DATASET_MOCKDATASET_H_ */
