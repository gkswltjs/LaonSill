/*
 * Network.h
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include <armadillo>

class DataSample;

class DataSet;

using namespace std;
using namespace arma;



class Network {
public:
	Network(int sizes[], int sizeCount, DataSet *dataSet);
	virtual ~Network();




	void feedforward();
	void sgd(int epochs, int miniBatchSize, double eta);
	void updateMiniBatch(int nthMiniBatch, int miniBatchSize, double eta, vector<mat *> &nabla_w, vector<vec *> &nabla_b);
	void backprop(const DataSample *dataSample, vector<mat *> &nabla_w, vector<vec *> &nabla_b);



	int evaluate();
	vec feedforward(const vec *x);
	int testEvaluateResult(const vec &evaluateResult, const vec *y);


private:
	void initializeParameters(vector<mat *> &weights, vector<vec *> &biases, bool init);
	void deallocParameters(vector<mat *> weights, vector<vec *> biases);
	vec costDerivative(const vec *outputActivation, const vec *y);
	vec sigmoid(const vec *activation);
	vec sigmoidPrime(const vec *z);


	DataSet *dataSet;

	int numLayers;
	int *sizes;
	vector<vec *> biases;
	vector<mat *> weights;

	vec (* actFuncPtr) (vec);

};

#endif /* NETWORK_H_ */
