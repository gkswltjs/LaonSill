/*
 * Network.h
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include <armadillo>

#include "../cost/Cost.h"
#include "../activation/Activation.h"
#include "../monitor/NetworkListener.h"
#include "../layer/Layer.h"
#include "../layer/InputLayer.h"
#include "../layer/HiddenLayer.h"
#include "../layer/OutputLayer.h"
#include "../layer/LayerConfig.h"

class DataSample;

class DataSet;

using namespace std;
using namespace arma;




class Network {
public:
	Network(NetworkListener *networkListener=0) {
		this->networkListener = networkListener;
		this->maxAccuracy = 0.0;
		this->minCost = 100.0;
		this->saveConfigured = false;
	}
	Network(InputLayer *inputLayer, OutputLayer *outputLayer, DataSet *dataSet, NetworkListener *networkListener);
	virtual ~Network();

	void addOutputLayer(OutputLayer *outputLayer) { this->outputLayers.push_back(outputLayer); }
	void setDataSet(DataSet *dataSet, UINT batches);
	InputLayer *getInputLayer() const { return this->inputLayer; }

	void sgd(int epochs);
	void test();


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

	void saveConfig(const char *savePrefix);
	void save(const char* filename);
	void load(const char* filename);
	void shape();
	void reshape(io_dim in_dim=io_dim(0,0,0,0));
	void backprop(const DataSample &dataSample);
	Layer *findLayer(const char *name);
	DATATYPE getDataSetMean(UINT channel);


#if CPU_MODE
public:
	void feedforward(const rcube &input, const char *end=0);
#else
public:
	void feedforward(const DATATYPE *input, const char *end=0);
#endif



protected:
	void updateMiniBatch(int nthMiniBatch);


	double totalCost(const vector<const DataSample *> &dataSet, double lambda);
	double accuracy(const vector<const DataSample *> &dataSet);
	int evaluate(int &accurateCnt, float &cost);

	DataSet *dataSet;
	NetworkListener *networkListener;

	InputLayer *inputLayer;
	vector<OutputLayer *> outputLayers;

	io_dim in_dim;

	char savePrefix[200];
	bool saveConfigured;
	double maxAccuracy;
	double minCost;

#if CPU_MODE
protected:
	int testEvaluateResult(const rvec &output, const rvec &y);
#else
protected:
	int testEvaluateResult(const int num_labels, const DATATYPE *output, const UINT *y, int &accurateCnt, float &cost);
#endif


};





#endif /* NETWORK_H_ */












































