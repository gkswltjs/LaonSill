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
#include <map>

#include "../dataset/DataSample.h"
#include "../dataset/DataSet.h"
#include "../layer/LayerFactory.h"
#include "../layer/HiddenLayer.h"
#include "../layer/OutputLayer.h"
#include "../Util.h"
#include "../Timer.h"





Network::Network(InputLayer *inputLayer, OutputLayer *outputLayer, DataSet *dataSet, NetworkListener *networkListener) {
	this->inputLayer = inputLayer;
	if(outputLayer) this->outputLayers.push_back(outputLayer);
	this->dataSet = dataSet;
	this->networkListener = networkListener;
}

Network::~Network() {
	if(inputLayer) {
		delete inputLayer;
		inputLayer = NULL;
	}
}






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


void Network::test() {
	if(dataSet->getTestDataSize() > 0) {
		cout << "Evaluating ... " << evaluate() << " / " << dataSet->getTestDataSize() << endl;
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
	Timer timer;
	timer.start();

	ofstream ofs(filename.c_str(), ios::out | ios::binary);

	//int inputLayerSize = 1;
	int outputLayerSize = outputLayers.size();

	//ofs.write((char *)&inputLayerSize, sizeof(int));		// input layer size
	//ofs.write((char *)&inputLayer, sizeof(Layer *));		// input layer address
	ofs.write((char *)&outputLayerSize, sizeof(int));		// output layer size

	for(UINT i = 0; i < outputLayers.size(); i++) {
		cout << "outputLayer: " << outputLayers[i] << endl;
		ofs.write((char *)&outputLayers[i], sizeof(Layer *));
	}

	inputLayer->save(0, ofs);

	ofs.close();

	cout << "time elapsed to save network: " << timer.stop(false) << endl;
}


void Network::load(string filename) {
	ifstream ifs(filename.c_str(), ios::in | ios::binary);

	//UINT inputLayerSize;
	//ifs.read((char *)&inputLayerSize, sizeof(UINT));

	//Layer *inputLayer;
	//ifs.read((char *)&inputLayer, sizeof(Layer *));

	UINT outputLayerSize;
	ifs.read((char *)&outputLayerSize, sizeof(UINT));

	for(UINT i = 0; i < outputLayerSize; i++) {
		OutputLayer *outputLayer;
		ifs.read((char *)&outputLayer, sizeof(OutputLayer *));
		outputLayers.push_back(outputLayer);
		//cout << "loaded outputLayer: " << outputLayer << endl;
	}

	map<Layer *, Layer *> layerMap;
	this->inputLayer = new InputLayer();
	this->inputLayer->load(ifs, layerMap);

	// restore output layers of network
	for(UINT i = 0; i < outputLayerSize; i++) {
		outputLayers[i] = (OutputLayer *)layerMap.find((Layer *)outputLayers[i])->second;
	}


	/*
	map<Layer *, Layer *> layerMap;
	while(true) {
		LayerType layerType;
		ifs.read((char *)&layerType, sizeof(int));
		Layer *address;
		ifs.read((char *)&address, sizeof(Layer *));

		if(address == 0) break;

		Layer *layer = LayerFactory::create(layerType);
		layerMap.insert(pair<Layer *, Layer *>(address, layer));
	}
	cout << "map size: " << layerMap.size() << endl;

	ifs.seekg(1000);
	Layer *layerKey;
	ifs.read((char *)&layerKey, sizeof(Layer *));

	while(ifs) {
		Layer *layer = layerMap.find(layerKey)->second;
		if(!layer) throw Exception();

		layer->load(ifs, layerMap);

		vector<next_layer_relation> &nextLayers = layer->getNextLayers();
		for(UINT i = 0; i < nextLayers.size(); i++) {
			Layer *nextLayer = nextLayers[i].next_layer;
			nextLayers[i].next_layer = layerMap.find(nextLayer)->second;
		}

		// 학습된 네트워크를 load하는 경우 backward pass가 없으므로 불필요
		//HiddenLayer *hiddenLayer = dynamic_cast<HiddenLayer *>(layer);
		//if(hiddenLayer) {
		//	vector<prev_layer_relation> &prevLayers = hiddenLayer->getPrevLayers();
		//	for(UINT i = 0; i < prevLayers.size(); i++) {
		//		Layer *prevLayer = prevLayers[i].prev_layer;
		//		prevLayers[i].prev_layer = dynamic_cast<HiddenLayer *>(layerMap.find(prevLayer)->second);
		//	}
		//}

		ifs.read((char *)&layerKey, sizeof(Layer *));
	}
	*/

	ifs.close();
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











