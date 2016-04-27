/*
 * DataSet.cpp
 *
 *  Created on: 2016. 4. 21.
 *      Author: jhkim
 */

#include "MnistDataSet.h"

#include <iostream>


#include "../Util.h"
#include "DataSample.h"
#include "ImageInfo.h"




MnistDataSet::MnistDataSet(double validationSetRatio = 0.0) {
	this->validationSetRatio = validationSetRatio;
}

MnistDataSet::~MnistDataSet() {}

void MnistDataSet::load() {

	string filenames[2][2] = {
			{"./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte"},
			{"./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte"}
	};

	trainDataSize = loadDataSetFromResource(filenames[0], trainDataSet, 0, 50000);
	//validationDataSize = loadDataSetFromResource(filenames[0], validationDataSet, 50000, 10000);
	testDataSize = loadDataSetFromResource(filenames[1], testDataSet, 0, 0);

	//trainDataSize = loadDataSetFromResource(filenames[0], trainDataSet);
	//testDataSize = loadDataSetFromResource(filenames[1], testDataSet);

}



/*
int MnistDataSet::loadDataSetFromResource(string resources[2], vector<const DataSample *> &dataSet) {

	// LOAD IMAGE DATA
	ImageInfo dataInfo(resources[0]);
	dataInfo.load();

	// READ IMAGE DATA META DATA
	unsigned char *dataPtr = dataInfo.getBufferPtrAt(0);
	int dataMagicNumber = Util::pack4BytesToInt(dataPtr);
	if(dataMagicNumber != 0x00000803) return -1;
	dataPtr += 4;
	int dataSize = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumRows = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumCols = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;


	// LOAD LABEL DATA
	ImageInfo targetInfo(resources[1]);
	targetInfo.load();

	// READ LABEL DATA META DATA
	unsigned char *targetPtr = targetInfo.getBufferPtrAt(0);
	int targetMagicNumber = Util::pack4BytesToInt(targetPtr);
	if(targetMagicNumber != 0x00000801) return -1;
	targetPtr += 4;
	//int labelSize = Util::pack4BytesToInt(targetPtr);
	targetPtr += 4;



	int dataArea = dataNumRows * dataNumCols;
	for(int i = 0; i < dataSize; i++) {
		const DataSample *dataSample = new DataSample(dataPtr, dataArea, targetPtr, 10);
		dataSet.push_back(dataSample);

		if(i < 10) {
			//Util::printVec(dataSample->getData(), "data");
			//Util::printVec(dataSample->getTarget(), "target");
		}
	}
	return dataSize;
}
*/



int MnistDataSet::loadDataSetFromResource(string resources[2], vector<const DataSample *> &dataSet, int offset, int size) {

	// LOAD IMAGE DATA

	ImageInfo dataInfo(resources[0]);
	dataInfo.load();

	// READ IMAGE DATA META DATA
	unsigned char *dataPtr = dataInfo.getBufferPtrAt(0);
	int dataMagicNumber = Util::pack4BytesToInt(dataPtr);
	if(dataMagicNumber != 0x00000803) return -1;
	dataPtr += 4;
	int dataSize = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumRows = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumCols = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;



	// LOAD LABEL DATA
	ImageInfo targetInfo(resources[1]);
	targetInfo.load();

	// READ LABEL DATA META DATA
	unsigned char *targetPtr = targetInfo.getBufferPtrAt(0);
	int targetMagicNumber = Util::pack4BytesToInt(targetPtr);
	if(targetMagicNumber != 0x00000801) return -1;
	targetPtr += 4;
	//int labelSize = Util::pack4BytesToInt(targetPtr);
	targetPtr += 4;


	if(offset >= dataSize) return 0;

	dataPtr += offset;
	targetPtr += offset;

	int stop = dataSize;
	if(size > 0) stop = min(dataSize, offset+size);

	int dataArea = dataNumRows * dataNumCols;
	for(int i = offset; i < stop; i++) {
		const DataSample *dataSample = new DataSample(dataPtr, dataArea, targetPtr, 10);
		dataSet.push_back(dataSample);
	}
	return stop-offset;
}


void MnistDataSet::shuffleTrainDataSet() {
	random_shuffle(trainDataSet.begin(), trainDataSet.end());
}











































