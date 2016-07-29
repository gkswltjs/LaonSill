/*
 * Evaluation.h
 *
 *  Created on: 2016. 7. 27.
 *      Author: jhkim
 */

#ifndef EVALUATION_H_
#define EVALUATION_H_

class Evaluation {
public:
	Evaluation() {
		this->accurateCount = 0;
		this->cost = 0.0f;
	}
	virtual ~Evaluation() {}

	int getAccurateCount() const { return accurateCount; }
	double getCost() const { return cost; }

	void reset() {
		this->accurateCount = 0;
		this->cost = 0.0f;
	}

	virtual void evaluate(const int num_labels, const int batches, const DATATYPE *d_output, const UINT *y)=0;

protected:
	int accurateCount;
	double cost;
};

#endif /* EVALUATION_H_ */
