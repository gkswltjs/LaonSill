/*
 * MockDataSet.cpp
 *
 *  Created on: 2016. 4. 23.
 *      Author: jhkim
 */

#include "MockDataSet.h"
#include "DataSample.h"
#include "../Util.h"

MockDataSet::MockDataSet(UINT rows, UINT cols, UINT channels, UINT numTrainData, UINT numTestData, UINT numLabels)
	: DataSet(rows, cols, channels, numTrainData, numTestData), numLabels(numLabels) {}

MockDataSet::~MockDataSet() {}

void MockDataSet::load() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> ud(-0.1, 0.1);

	UINT i;
	// load train data
	for(i = 0; i < dataSize*numTrainData; i++) {
		(*trainDataSet)[i] = static_cast<DATATYPE>(ud(gen));
	}
	for(i = 0; i < numTrainData; i++) {
		UINT label = static_cast<UINT>((ud(gen)+0.1)*numLabels*5);
		cout << "label: " << label << endl;
		(*trainLabelSet)[i] = static_cast<UINT>(label);
	}

	// load test data
	for(i = 0; i < dataSize*numTestData; i++) {
		(*testDataSet)[i] = static_cast<DATATYPE>(ud(gen));
	}
	for(i = 0; i < numTestData; i++) {
		UINT label = static_cast<UINT>((ud(gen)+0.1)*numLabels*5);
				cout << "label: " << label << endl;
		(*testLabelSet)[i] = static_cast<UINT>(label);
	}

	/*
	double trainData[100] = {
			0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3,
			0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
			0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
			0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0,
			0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0,
			0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3,
			0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
			0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
			0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0,
			0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0,
	};
	int trainTarget[trainDataSize] = {
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9
	};


	for(int i = 0; i < trainDataSize; i++) {
		trainDataSet[i].readData(trainData, trainTarget[i]);
		//trainDataSet.push_back(new DataSample(&trainData[i*9], trainTarget[i]));
		//Util::printVec(trainDataSet[i]->getData(), "data");
		//Util::printVec(trainDataSet[i]->getTarget(), "target");
	}

	for(int i = 0; i < validationDataSize; i++) {
		validationDataSet[i].readData(trainData, trainTarget[i]);
		//validationDataSet.push_back(new DataSample(&trainData[i*9], trainTarget[i]));
	}

	for(int i = 0; i < testDataSize; i++) {
		testDataSet[i].readData(trainData, trainTarget[i]);
		//testDataSet.push_back(new DataSample(&trainData[i*9], trainTarget[i]));
	}
	*/
}




