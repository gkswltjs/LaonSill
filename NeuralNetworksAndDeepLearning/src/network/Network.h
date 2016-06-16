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
//#include "../layer/InputLayer.h"
//#include "../layer/HiddenLayer.h"
//#include "../layer/OutputLayer.h"
//#include "../layer/LayerConfig.h"

class DataSample;

class DataSet;

using namespace std;
using namespace arma;



#if CPU_MODE


class Network {
public:
	Network() {}
	Network(InputLayer *inputLayer, OutputLayer *outputLayer, DataSet *dataSet, NetworkListener *networkListener);
	virtual ~Network();

	void addOutputLayer(OutputLayer *outputLayer) { this->outputLayers.push_back(outputLayer); }
	void setDataSet(DataSet *dataSet) { this->dataSet = dataSet; }

	void sgd(int epochs, int miniBatchSize);
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

	void save(string filename);
	void load(string filename);

protected:
	void updateMiniBatch(int nthMiniBatch, int miniBatchSize);
	void backprop(const DataSample &dataSample);

	//void feedforward();
	void feedforward(const rcube &input);
	int testEvaluateResult(const rvec &output, const rvec &y);
	double totalCost(const vector<const DataSample *> &dataSet, double lambda);
	double accuracy(const vector<const DataSample *> &dataSet);
	int evaluate();

	DataSet *dataSet;
	NetworkListener *networkListener;

	InputLayer *inputLayer;
	vector<OutputLayer *> outputLayers;
};


#else



#endif


#endif /* NETWORK_H_ */












































