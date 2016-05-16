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
#include "activation/Activation.h"
#include "monitor/NetworkListener.h"
#include "layer/Layer.h"

class DataSample;

class DataSet;

using namespace std;
using namespace arma;



class Network {
public:
	Network(Layer **layers, int numLayers, DataSet *dataSet, NetworkListener *networkListener);
	virtual ~Network();

	void sgd(int epochs, int miniBatchSize, double eta, double lambda);

	void save(string filename);
	void load(string filename);

private:
	void updateMiniBatch(int nthMiniBatch, int miniBatchSize, double eta, double lambda);
	void backprop(const DataSample &dataSample);

	//void feedforward();
	void feedforward(const cube &input);
	int testEvaluateResult(const vec &output, const vec &y);

	//void defaultWeightInitializer(vector<mat *> &weights, vector<vec *> &biases, bool init);
	//void deallocParameters(vector<mat *> weights, vector<vec *> biases);
	vec costDerivative(const vec *outputActivation, const vec *y);
	vec sigmoid(const vec *activation);
	vec sigmoidPrime(const vec *z);
	double totalCost(const vector<const DataSample *> &dataSet, double lambda);
	double accuracy(const vector<const DataSample *> &dataSet);
	int evaluate();


	Cost *cost_fn;
	Activation *activation_fn;
	DataSet *dataSet;
	NetworkListener *networkListener;

	Layer **layers;
	//InputLayer *inputLayer;
	//HiddenLayer **hiddenLayers;
	//OutputLayer *outputLayer;

	int numLayers;
	//vector<vec *> biases;
	//vector<mat *> weights;
};

#endif /* NETWORK_H_ */
