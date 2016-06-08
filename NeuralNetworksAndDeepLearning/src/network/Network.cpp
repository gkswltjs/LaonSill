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

#include "../dataset/DataSample.h"
#include "../dataset/DataSet.h"
#include "../layer/HiddenLayer.h"
#include "../layer/OutputLayer.h"
#include "../Util.h"
#include "../Timer.h"





Network::Network(InputLayer *inputLayer, OutputLayer *outputLayer, DataSet *dataSet, NetworkListener *networkListener) {
	this->inputLayer = inputLayer;
	this->outputLayers.push_back(outputLayer);
	this->dataSet = dataSet;
	this->networkListener = networkListener;
}

Network::~Network() {}






void Network::sgd(int epochs, int miniBatchSize) {
	int trainDataSize = dataSet->getTrainDataSize();
	int miniBatchesSize = trainDataSize / miniBatchSize;

	Timer timer1, timer2;

	for(int i = 0; i < epochs; i++) {
		//timer1.start();
		dataSet->shuffleTrainDataSet();

		timer2.start();
		for(int j = 0; j < miniBatchesSize; j++) {
			if((j+1)%100 == 0) {
				cout << "Minibatch " << j+1 << " started: " << timer2.stop(false) << endl;
				timer2.start();
			}
			//cout << "Minibatch " << j+1 << " started: " << timer2.stop(false) << endl;
			//timer2.start();

			//cout << "reset_nabla()" << endl;
			inputLayer->reset_nabla(0);
			updateMiniBatch(j, miniBatchSize);
		}
		//timer1.stop();

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



void Network::updateMiniBatch(int nthMiniBatch, int miniBatchSize) {

	int baseIndex = nthMiniBatch*miniBatchSize;
	for(int i = 0; i < miniBatchSize; i++) {
		backprop(dataSet->getTrainDataAt(baseIndex+i));
	}

	int n = dataSet->getTrainDataSize();

	//cout << "update()" << endl;
	inputLayer->update(0, n, miniBatchSize);
}



void Network::backprop(const DataSample &dataSample) {
	//Timer timer;
	//timer.start();
	// feedforward
	feedforward(dataSample.getData());

	//cout << "time for feed forward: ";
	//timer.stop();

	//timer.start();
	//cout << "backpropagation()" << endl;
	for(UINT i = 0; i < outputLayers.size(); i++) {
		outputLayers[i]->cost(dataSample.getTarget());
	}
	//cout << "time for backward: ";
	//timer.stop();

}





void Network::feedforward(const rcube &input) {
	//cout << "feedforward()" << endl;
	inputLayer->feedforward(0, input);

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




int Network::testEvaluateResult(const rvec &output, const rvec &y) {
	//Util::printVec(&evaluateResult, "result");
	//Util::printVec(y, "y");

	uword rrow, yrow;
	output.max(rrow);
	y.max(yrow);

	if(rrow == yrow) return 1;
	else return 0;
}




void Network::save(string filename) {

	ifstream fin(filename.c_str(), ios::in | ios::ate | ios::binary);

	ofstream ofs(filename.c_str(), ios::binary);

	inputLayer->save(0, ofs);

	ofs.close();
}


void Network::load(string filename) {
}




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
		testResult += testEvaluateResult(outputLayers[0]->getOutput(), testData.getTarget());
	}
	//Util::setPrint(printBak);

	return testResult;
}











