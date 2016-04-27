/*
 * Network.h
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include <armadillo>

#include "cost/Cost.h"
#include "monitor/NetworkListener.h"

class DataSample;

class DataSet;

using namespace std;
using namespace arma;



class Network {
public:
	Network(int sizes[], int sizeCount, DataSet *dataSet, Cost *cost, NetworkListener *networkListener);
	virtual ~Network();

	void sgd(int epochs, int miniBatchSize, double eta, double lambda);

	void save(string filename);
	void load(string filename);

private:
	void updateMiniBatch(int nthMiniBatch, int miniBatchSize, double eta, double lambda, vector<mat *> &nabla_w, vector<vec *> &nabla_b);
	void backprop(const DataSample *dataSample, vector<mat *> &nabla_w, vector<vec *> &nabla_b);

	void feedforward();
	vec feedforward(const vec *x);
	int testEvaluateResult(const vec &evaluateResult, const vec *y);

	void defaultWeightInitializer(vector<mat *> &weights, vector<vec *> &biases, bool init);
	void deallocParameters(vector<mat *> weights, vector<vec *> biases);
	vec costDerivative(const vec *outputActivation, const vec *y);
	vec sigmoid(const vec *activation);
	vec sigmoidPrime(const vec *z);
	double totalCost(const vector<const DataSample *> &dataSet, double lambda);
	double accuracy(const vector<const DataSample *> &dataSet);
	int evaluate();


	Cost *cost;
	DataSet *dataSet;
	NetworkListener *networkListener;

	int numLayers;
	int *sizes;
	vector<vec *> biases;
	vector<mat *> weights;
};

#endif /* NETWORK_H_ */
