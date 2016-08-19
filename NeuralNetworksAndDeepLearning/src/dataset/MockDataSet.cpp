/*
 * MockDataSet.cpp
 *
 *  Created on: 2016. 4. 23.
 *      Author: jhkim
 */

#include "MockDataSet.h"
#include "../Util.h"
#include <random>

MockDataSet::MockDataSet(uint32_t rows, uint32_t cols, uint32_t channels, uint32_t numTrainData, uint32_t numTestData, uint32_t numLabels)
	: DataSet(rows, cols, channels, numTrainData, numTestData), numLabels(numLabels) {}

MockDataSet::~MockDataSet() {}

void MockDataSet::load() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> ud(-0.1, 0.1);

	uint32_t i;
	// load train data
	for(i = 0; i < dataSize*numTrainData; i++) {
		(*trainDataSet)[i] = static_cast<DATATYPE>(ud(gen));
	}
	for(i = 0; i < numTrainData; i++) {
		uint32_t label = static_cast<uint32_t>((ud(gen)+0.1)*numLabels*5);
		(*trainLabelSet)[i] = static_cast<uint32_t>(label);
	}

	// load test data
	for(i = 0; i < dataSize*numTestData; i++) {
		(*testDataSet)[i] = static_cast<DATATYPE>(ud(gen));
	}
	for(i = 0; i < numTestData; i++) {
		uint32_t label = static_cast<uint32_t>((ud(gen)+0.1)*numLabels*5);
		(*testLabelSet)[i] = static_cast<uint32_t>(label);
	}
}




