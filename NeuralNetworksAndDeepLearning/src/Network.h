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
#include "layer/InputLayer.h"
#include "layer/HiddenLayer.h"
#include "layer/OutputLayer.h"
#include "layer/LayerConfig.h"

class DataSample;

class DataSet;

using namespace std;
using namespace arma;



class Network {
public:
	Network(InputLayer *inputLayer, DataSet *dataSet, NetworkListener *networkListener);
	virtual ~Network();

	void sgd(int epochs, int miniBatchSize, double eta, double lambda);


	//void setInputLayer(InputLayer *inputLayer) { this->inputLayer = inputLayer; }
	void addOutputLayer(OutputLayer *outputLayer) { this->outputLayers.push_back(outputLayer); }







	/**
	 * 어디로 옮기면 좋을까
	 */
	static void addLayerRelation(Layer *prevLayer, HiddenLayer *nextLayer) {
		int nextLayerIdx = nextLayer->getPrevLayerSize();
		int prevLayerIdx = prevLayer->getNextLayerSize();

		prevLayer->addNextLayer(next_layer_relation(nextLayer, nextLayerIdx));

		// prev layer가 hidden layer가 아닌 경우 prev layers에 추가할 필요 없음
		HiddenLayer *pLayer = dynamic_cast<HiddenLayer *>(prevLayer);
		if(pLayer) nextLayer->addPrevLayer(prev_layer_relation(pLayer, prevLayerIdx));
	}





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

	//Layer **layers;



	InputLayer *inputLayer;
	vector<OutputLayer *> outputLayers;



	//InputLayer *inputLayer;
	//HiddenLayer **hiddenLayers;
	//OutputLayer *outputLayer;

	int numLayers;
	//vector<vec *> biases;
	//vector<mat *> weights;
};

#endif /* NETWORK_H_ */












































