/*
 * Cifar10DataSet.cpp
 *
 *  Created on: 2016. 5. 19.
 *      Author: jhkim
 */

#include "Cifar10DataSet.h"

Cifar10DataSet::Cifar10DataSet() {}

Cifar10DataSet::~Cifar10DataSet() {
	if(trainDataSet) delete trainDataSet;
	if(validationDataSet) delete validationDataSet;
	if(testDataSet) delete testDataSet;
}

void Cifar10DataSet::load() {

	string filenames[2][5] = {
			{"./data/cifar-10/data_batch_1.bin", "./data/cifar-10/data_batch_2.bin", "./data/cifar-10/data_batch_3.bin",
					"./data/cifar-10/data_batch_4.bin", "./data/cifar-10/data_batch_5.bin"},
			{"./data/cifar-10/test_batch.bin", "", "", "", ""}
	};

	trainDataSize = loadDataSetFromResource(filenames[0], 5, trainDataSet, 50000);
	//validationDataSize = loadDataSetFromResource(filenames[0], validationDataSet, 50000, 10000);
	testDataSize = loadDataSetFromResource(filenames[1], 1, testDataSet, 10000);

	//trainDataSize = loadDataSetFromResource(filenames[0], trainDataSet);
	//testDataSize = loadDataSetFromResource(filenames[1], testDataSet);

}




int Cifar10DataSet::loadDataSetFromResource(string *resources, int numResources, DataSample *&dataSet, int dataSize) {
	// LOAD IMAGE DATA
	if(dataSet) delete dataSet;
	dataSet = new DataSample[dataSize];

	for(int i = 0; i < numResources; i++) {
		ImageInfo dataInfo(resources[i]);
		dataInfo.load();

		// READ IMAGE DATA META DATA
		unsigned char *dataPtr = dataInfo.getBufferPtrAt(0);
		for(int j = 0; j < 10000; j++) {
			//const DataSample *dataSample = new DataSample(dataPtr, dataArea, targetPtr, 10);
			//dataSet.push_back(dataSample);
			//dataSet[i-offset].readData(dataPtr, dataNumRows, dataNumCols, 1, targetPtr, 10);
			dataSet[i*10000+j].readData(dataPtr, 32, 32, 3);
		}
	}
	return dataSize;
}


void Cifar10DataSet::shuffleTrainDataSet() {
	random_shuffle(&trainDataSet[0], &trainDataSet[trainDataSize]);
}

void Cifar10DataSet::shuffleValidationDataSet() {
	random_shuffle(&validationDataSet[0], &validationDataSet[validationDataSize]);
}

void Cifar10DataSet::shuffleTestDataSet() {
	random_shuffle(&testDataSet[0], &testDataSet[testDataSize]);
}


