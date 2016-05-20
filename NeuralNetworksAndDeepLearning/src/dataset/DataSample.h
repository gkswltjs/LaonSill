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
	DataSample() {}

	/*
	DataSample(const double *trainData, const int trainTarget) {
		readDAta(trainData, trainTarget);
	}
	DataSample(unsigned char *&dataPtr, int dataBytes, unsigned char *&targetPtr, int targetBytes) {
		readData(dataPtr, dataBytes, targetPtr, targetBytes);
	}
	*/
	virtual ~DataSample() {}

	const cube &getData() const { return this->data; }
	const vec &getTarget() const { return this->target; }
	//void setData(vec *data) { this->data = data; }
	//void setTarget(vec *target) { this->target = target; }

	void readData(const double *trainData, const int trainTarget) {
		this->data.set_size(12, 12, 3);
		this->target.set_size(10, 1);

		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 12; i++) {
				for(int j = 0; j < 12; j++) {
					data.slice(k)(i, j) = trainData[i*12+j];
				}
			}
		}
		this->target.fill(0.0);
		this->target.row(trainTarget) = 1.0;
	}

	void readData(unsigned char *&dataPtr, int rows, int cols, int channels, unsigned char *&targetPtr, int targetBytes) {
		this->data.set_size(rows, cols, channels);
		Util::printVec(this->data, "data");

		for(int i = 0; i < channels; i++) {
			for(int j = 0; j < rows; j++) {
				for(int k = 0; k < cols; k++) {
					this->data.slice(i)(j, k) = (*dataPtr)/255.0;
					dataPtr++;
				}
			}
		}

		this->target.set_size(targetBytes, 1);
		this->target.fill(0.0f);
		this->target.row(*targetPtr) = 1.0;
		targetPtr++;
	}

	void readData(unsigned char *&dataPtr, int rows, int cols, int channels) {
			this->data.set_size(rows, cols, channels);

			this->target.set_size(10, 1);
			this->target.fill(0.0f);
			this->target.row(*dataPtr) = 1.0;
			//Util::printVec(target, "target:");

			dataPtr++;

			for(int i = 0; i < channels; i++) {
				for(int j = 0; j < rows; j++) {
					for(int k = 0; k < cols; k++) {
						this->data.slice(i)(j, k) = (*dataPtr)/255.0;
						dataPtr++;
					}
				}
			}
			//Util::printCube(data, "data:");
		}

private:
	cube data;
	vec target;
};

#endif /* DATASET_DATASAMPLE_H_ */
