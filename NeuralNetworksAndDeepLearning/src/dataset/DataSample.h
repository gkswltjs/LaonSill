/*
 * DataSample.h
 *
 *  Created on: 2016. 4. 21.
 *      Author: jhkim
 */

#ifndef DATASET_DATASAMPLE_H_
#define DATASET_DATASAMPLE_H_

#include <armadillo>

using namespace arma;




class DataSample {
public:
	DataSample(const double *trainData, const int trainTarget) {
		this->data = new vec(9);
		this->target = new vec(10);

		for(int i = 0; i < 9; i++) {
			data->row(i) = trainData[i];
		}

		this->target->fill(0.0);
		this->target->row(trainTarget) = 1.0;
	}
	DataSample(unsigned char *&dataPtr, int dataBytes, unsigned char *&targetPtr, int targetBytes) {
		this->data = new vec(dataBytes);
		for(int j = 0; j < dataBytes; j++) {
			this->data->row(j) = (*dataPtr)/255.0;
			dataPtr++;
		}

		this->target = new vec(10);
		this->target->fill(0.0f);
		this->target->row(*targetPtr) = 1.0;
		targetPtr++;
	}
	virtual ~DataSample() {
		delete data;
		delete target;
	};


	const vec *getData() const { return this->data; }
	const vec *getTarget() const { return this->target; }
	void setData(vec *data) { this->data = data; }
	void setTarget(vec *target) { this->target = target; }

private:
	int trainDataSize;
	int testDataSize;
	int inputSize;
	int outputSize;

	vec *data;
	vec *target;
};

#endif /* DATASET_DATASAMPLE_H_ */
