/*
 * MockDataSet.cpp
 *
 *  Created on: 2016. 4. 23.
 *      Author: jhkim
 */

#include "MockDataSet.h"
#include "../Util.h"
#include <random>

template <typename Dtype>
MockDataSet<Dtype>::MockDataSet(uint32_t rows, uint32_t cols, uint32_t channels, uint32_t numTrainData, uint32_t numTestData, uint32_t numLabels)
	: DataSet<Dtype>(rows, cols, channels, numTrainData, numTestData), numLabels(numLabels) {}

template <typename Dtype>
MockDataSet<Dtype>::~MockDataSet() {}

template <typename Dtype>
void MockDataSet<Dtype>::load() {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> ud(-0.1, 0.1);

	uint32_t i;
	// load train data
	for(i = 0; i < this->dataSize*this->numTrainData; i++) {
		(*this->trainDataSet)[i] = static_cast<Dtype>(ud(gen));
	}
	for(i = 0; i < this->numTrainData; i++) {
		uint32_t label = static_cast<uint32_t>((ud(gen)+0.1)*numLabels*5);
		(*this->trainLabelSet)[i] = static_cast<uint32_t>(label);
	}

	// load test data
	for(i = 0; i < this->dataSize*this->numTestData; i++) {
		(*this->testDataSet)[i] = static_cast<Dtype>(ud(gen));
	}
	for(i = 0; i < this->numTestData; i++) {
		uint32_t label = static_cast<uint32_t>((ud(gen)+0.1)*numLabels*5);
		(*this->testLabelSet)[i] = static_cast<uint32_t>(label);
	}
}



template class MockDataSet<float>;
