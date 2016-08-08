/**
 * @file DataSample.h
 * @date 2016/4/21
 * @author jhkim
 * @brief deprecated. not used.
 * @details
 */


#ifndef DATASET_DATASAMPLE_H_
#define DATASET_DATASAMPLE_H_

#include "../Util.h"
#include <armadillo>

using namespace arma;




class DataSample {


#if CPU_MODE
public:
	DataSample() {}

	void init(int rows, int cols, int channels, int classes) {
		data = randu<rcube>(rows, cols, channels);
		target = rvec(classes, 1);
		target(0, 0) = 1.0;
	}

	/*
	DataSample(const double *trainData, const int trainTarget) {
		readDAta(trainData, trainTarget);
	}
	DataSample(unsigned char *&dataPtr, int dataBytes, unsigned char *&targetPtr, int targetBytes) {
		readData(dataPtr, dataBytes, targetPtr, targetBytes);
	}
	*/
	virtual ~DataSample() {}

	const rcube &getData() const { return this->data; }
	const rvec &getTarget() const { return this->target; }
	//void setData(vec *data) { this->data = data; }
	//void setTarget(vec *target) { this->target = target; }

	void readData(const double *trainData, const int trainTarget) {
		this->data.set_size(10, 10, 1);
		this->target.set_size(10, 1);

		for(int k = 0; k < 1; k++) {
			for(int i = 0; i < 10; i++) {
				for(int j = 0; j < 10; j++) {
					//data.slice(k)(i, j) = trainData[i*10+j];
					C_MEMPTR(data, i, j, k) = trainData[i*10+j];
				}
			}
		}
		this->target.zeros();
		this->target.row(trainTarget) = 1.0;
	}

	void readData(unsigned char *&dataPtr, int rows, int cols, int channels, unsigned char *&targetPtr, int targetBytes) {
		this->data.set_size(rows, cols, channels);
		//Util::printVec(this->data, "data");

		for(int i = 0; i < channels; i++) {
			for(int j = 0; j < rows; j++) {
				for(int k = 0; k < cols; k++) {
					//this->data.slice(i)(j, k) = (*dataPtr)/255.0;
					C_MEMPTR(this->data, j, k, i) = (*dataPtr)/255.0;
					dataPtr++;
				}
			}
		}

		this->target.set_size(targetBytes, 1);
		this->target.zeros();
		this->target.row(*targetPtr) = 1.0;
		targetPtr++;
	}

	void readData(unsigned char *&dataPtr, int rows, int cols, int channels) {
			this->data.set_size(rows, cols, channels);

			this->target.set_size(10, 1);
			this->target.zeros();
			this->target.row(*dataPtr) = 1.0;
			//Util::printVec(target, "target:");

			dataPtr++;

			for(int i = 0; i < channels; i++) {
				for(int j = 0; j < rows; j++) {
					for(int k = 0; k < cols; k++) {
						//this->data.slice(i)(j, k) = (*dataPtr)/255.0;
						C_MEMPTR(this->data, j, k, i) = (*dataPtr)/255.0;
						dataPtr++;
					}
				}
			}
			//Util::printCube(data, "data:");
		}

private:
	rcube data;
	rvec target;

#else
public:
	DataSample() {}

	void init(int rows, int cols, int channels, int classes) {
		data = new DATATYPE[rows*cols*channels];

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> ud(-0.1, 0.1);

		UINT i;
		for(i = 0; i < rows*cols*channels; i++) {
			data[i] = static_cast<DATATYPE>(ud(gen));
		}

		target = 5;

		//data = randu<rcube>(rows, cols, channels);
		//target = rvec(classes, 1);
		//target(0, 0) = 1.0;
	}

	virtual ~DataSample() {
		delete [] data;
	}

	const DATATYPE *getData() const { return this->data; }
	const int getTarget() const { return this->target; }


private:
	DATATYPE *data;
	int target;
#endif


};

#endif /* DATASET_DATASAMPLE_H_ */
