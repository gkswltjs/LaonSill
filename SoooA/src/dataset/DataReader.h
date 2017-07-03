/*
 * DataReader.h
 *
 *  Created on: Jun 30, 2017
 *      Author: jkim
 */

#ifndef DATAREADER_H_
#define DATAREADER_H_

#include <string>
#include <queue>

#include "SDF.h"

template <typename T>
class DataReader {
public:
	DataReader(const std::string& source);
	virtual ~DataReader();

	T* getNextData();
	T* peekNextData();

	int getNumData();

private:
	SDF db;
	int numData;

	std::queue<T*> data_queue;

};

#endif /* DATAREADER_H_ */
