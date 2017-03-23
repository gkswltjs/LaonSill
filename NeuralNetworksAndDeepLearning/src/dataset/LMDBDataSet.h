/*
 * LMDBDataSet.h
 *
 *  Created on: Mar 22, 2017
 *      Author: jkim
 */

#ifndef LMDBDATASET_H_
#define LMDBDATASET_H_

#include "common.h"
#include "DataSet.h"
#include "lmdb++.h"

template <typename Dtype>
class LMDBDataSet : public DataSet<Dtype> {
public:
	LMDBDataSet(const std::string& source);
	virtual ~LMDBDataSet();

	virtual void load();

	virtual const Dtype* getTrainDataAt(int index);
	virtual const Dtype* getTrainLabelAt(int index);
	virtual const Dtype* getValidationDataAt(int index);
	virtual const Dtype* getValidationLabelAt(int index);
	virtual const Dtype* getTestDataAt(int index);
	virtual const Dtype* getTestLabelAt(int index);

	virtual void shuffleTrainDataSet();
	virtual void shuffleValidationDataSet();
	virtual void shuffleTestDataSet();

private:
	std::string 	_source;

	lmdb::env* 		_env;
	lmdb::txn* 		_rtxn;
	lmdb::dbi* 		_dbi;
	lmdb::cursor*	_cursor;


	Dtype*			_trainData;
	Dtype*			_trainLabel;
	bool			_isRetrieved;

	/*
	int				_height;
	int				_width;
	int				_channel;
	int				_dataSize;
	*/

};

#endif /* LMDBDATASET_H_ */
