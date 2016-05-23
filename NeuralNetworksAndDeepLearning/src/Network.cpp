/*
 * Network.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include "Network.h"

#include <armadillo>
#include <iostream>
#include <vector>

#include "dataset/DataSample.h"
#include "dataset/DataSet.h"
#include "layer/HiddenLayer.h"
#include "layer/OutputLayer.h"
#include "Util.h"
#include "Timer.h"





Network::Network(Layer **layers, int numLayers, DataSet *dataSet, NetworkListener *networkListener) {
	this->layers = layers;
	this->numLayers = numLayers;

	//defaultWeightInitializer(weights, biases, true);

	this->dataSet = dataSet;
	//this->cost_fn = cost_fn;
	//this->activation_fn = activation_fn;
	this->networkListener = networkListener;
}

Network::~Network() {}












void Network::sgd(int epochs, int miniBatchSize, double eta, double lambda) {
	int trainDataSize = dataSet->getTrainDataSize();
	int miniBatchesSize = trainDataSize / miniBatchSize;

	Timer timer;

	for(int i = 0; i < epochs; i++) {
		timer.start();

		dataSet->shuffleTrainDataSet();

		for(int j = 0; j < miniBatchesSize; j++) {
			if((j+1)%100 == 0) cout << "Minibatch " << j+1 << " started." << endl;
			//cout << "Minibatch " << j+1 << " started." << endl;

			for(int k = 1; k < numLayers; k++) {
				(dynamic_cast<HiddenLayer *>(layers[k]))->reset_nabla();
			}
			updateMiniBatch(j, miniBatchSize, eta, lambda);
		}

		timer.stop();


		//dataSet->shuffleTestDataSet();
		if(dataSet->getTestDataSize() > 0) {
			cout << "Epoch " << i+1 << " " << evaluate() << " / " << dataSet->getTestDataSize() << endl;
		} else {
			cout << "Epoch " << i+1 << " complete." << endl;
		}


		/*
		cout << "Epoch " << i+1 << " training complete" << endl;
		if(networkListener) {
			//double validationCost = totalCost(dataSet->getValidationDataSet(), lambda);
			//double validationAccuracy = accuracy(dataSet->getValidationDataSet());
			//double trainCost = totalCost(dataSet->getTrainDataSet(), lambda);
			//double trainAccuracy = accuracy(dataSet->getTrainDataSet());
			double validationCost = 0;
			double validationAccuracy = accuracy(dataSet->getValidationDataSet());
			double trainCost = 0;
			double trainAccuracy = 0;
			networkListener->epochComplete(validationCost, validationAccuracy, trainCost, trainAccuracy);
		}
		*/
	}

}



void Network::updateMiniBatch(int nthMiniBatch, int miniBatchSize, double eta, double lambda) {

	int baseIndex = nthMiniBatch*miniBatchSize;
	for(int i = 0; i < miniBatchSize; i++) {
		backprop(dataSet->getTrainDataAt(baseIndex+i));
	}

	int n = dataSet->getTrainDataSize();
	for(int i = 1; i < numLayers; i++) {
		(dynamic_cast<HiddenLayer *>(layers[i]))->update(eta, lambda, n, miniBatchSize);
	}
}



void Network::backprop(const DataSample &dataSample) {
	int lastLayerIndex = numLayers-1;

	// feedforward
	feedforward(dataSample.getData());

	//Util::printVec(dataSample.getTarget(), "target:");
	//Util::printCube(layers[lastLayerIndex]->getInput(), "input:");

	// backward pass
	(dynamic_cast<OutputLayer *>(layers[lastLayerIndex]))->cost(dataSample.getTarget());

	for(int i = lastLayerIndex-1; i > 0; i--) {
		(dynamic_cast<HiddenLayer *>(layers[i]))->backpropagation(dynamic_cast<HiddenLayer *>(layers[i+1]));
	}
}





void Network::feedforward(const cube &input) {

	//Util::printCube(input, "input:");
	layers[0]->feedforward(input);


	for(int i = 1; i < numLayers; i++) {
		layers[i]->feedforward(layers[i-1]->getOutput());
	}

}






/*
double Network::totalCost(const vector<const DataSample *> &dataSet, double lambda) {
	double cost = 0.0;
	int dataSize = dataSet.size();

	for(int i = 0; i < dataSize; i++) {
		vec activation = feedforward(dataSet[i]->getData());
		cost += this->cost->fn(&activation, dataSet[i]->getTarget());
	}
	cost /= dataSize;

	// add weight decay term of cost
	for(int i = 1; i < numLayers; i++) {
		cost += 0.5*(lambda/dataSize)*accu(square(*weights[i]));
	}
	return cost;
}



double Network::accuracy(const vector<const DataSample *> &dataSet) {
	int total = 0;
	int dataSize = dataSet.size();
	for(int i = 0; i < dataSize; i++) {
		const DataSample *dataSample = dataSet[i];
		Util::printVec(dataSample->getData(), "data");
		Util::printVec(dataSample->getTarget(), "target");
		total += testEvaluateResult(feedforward(dataSample->getData()), dataSample->getTarget());
	}
	return total/(double)dataSize;
}
*/




int Network::testEvaluateResult(const vec &output, const vec &y) {
	//Util::printVec(&evaluateResult, "result");
	//Util::printVec(y, "y");

	uword rrow, yrow;
	output.max(rrow);
	y.max(yrow);

	if(rrow == yrow) return 1;
	else return 0;
}



/*
void Network::save(string filename) {

}

void Network::load(string filename) {
}
*/



int Network::evaluate() {
	int testResult = 0;
	//bool printBak = Util::getPrint();
	//Util::setPrint(true);
	int testDataSize = dataSet->getTestDataSize();
	for(int i = 0; i < testDataSize; i++) {
		//const DataSample *testData = dataSet->getTestDataAt(i);
		const DataSample &testData = dataSet->getTestDataAt(i);
		//Util::printVec(testData->getData(), "data");
		//Util::printVec(testData->getTarget(), "target");

		feedforward(testData.getData());
		testResult += testEvaluateResult(layers[numLayers-1]->getOutput(), testData.getTarget());
	}
	//Util::setPrint(printBak);

	return testResult;
}
















































