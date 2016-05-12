/*
 * DataSample.h
 *
 *  Created on: 2016. 4. 21.
 *      Author: jhkim
 */

#ifndef DATASET_DATASAMPLE_H_
#define DATASET_DATASAMPLE_H_

#include "../Util.h"
#include <armadillo>

using namespace arma;




class DataSample {
public:
	DataSample() {};

	/*
	DataSample(const double *trainData, const int trainTarget) {
		readDAta(trainData, trainTarget);
	}
	DataSample(unsigned char *&dataPtr, int dataBytes, unsigned char *&targetPtr, int targetBytes) {
		readData(dataPtr, dataBytes, targetPtr, targetBytes);
	}
	*/
	virtual ~DataSample() {};

	const vec &getData() const { return this->data; }
	const vec &getTarget() const { return this->target; }
	//void setData(vec *data) { this->data = data; }
	//void setTarget(vec *target) { this->target = target; }

	void readData(const double *trainData, const int trainTarget) {
		this->data.set_size(9, 1);
		this->target.set_size(10, 1);

		for(int i = 0; i < 9; i++) data.row(i) = trainData[i];

		this->target.fill(0.0);
		this->target.row(trainTarget) = 1.0;
	}

	void readData(unsigned char *&dataPtr, int dataBytes, unsigned char *&targetPtr, int targetBytes) {
		this->data.set_size(dataBytes, 1);
		Util::printVec(this->data, "data");

		for(int j = 0; j < dataBytes; j++) {
			this->data.row(j) = (*dataPtr)/255.0;
			dataPtr++;
		}

		this->target.set_size(targetBytes, 1);
		this->target.fill(0.0f);
		this->target.row(*targetPtr) = 1.0;
		targetPtr++;
	}

private:
	int trainDataSize;
	int testDataSize;
	int inputSize;
	int outputSize;

	vec data;
	vec target;
};

#endif /* DATASET_DATASAMPLE_H_ */
