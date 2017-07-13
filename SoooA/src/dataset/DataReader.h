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
    void fillNextData(T* data);

	int getNumData();

private:
	SDF db;
	int numData;

	std::queue<T*> data_queue;

    /***************************************************************************
     * callback functions - will be registered by Input Data Provider module
     * *************************************************************************/
public:
    static void allocElem(void** elemPtr);
    static void deallocElem(void* elemPtr);
    static void fillElem(void* reader, void* elemPtr);
};

#endif /* DATAREADER_H_ */
