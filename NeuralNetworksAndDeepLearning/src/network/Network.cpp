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

void Network::setDataSet(DataSet *dataSet, UINT batches) {
	this->dataSet = dataSet;
	this->in_dim.rows = dataSet->getRows();
	this->in_dim.cols = dataSet->getCols();
	this->in_dim.channels = dataSet->getChannels();
	this->in_dim.batches = batches;
}

void Network::shape() {
	inputLayer->shape(0, in_dim);
}

void Network::reshape() {
	inputLayer->reshape(0, in_dim);
}


void Network::sgd(int epochs) {
	int trainDataSize = dataSet->getNumTrainData();
	int miniBatchesSize = trainDataSize / in_dim.batches;

	Timer timer1, timer2;
	for(int i = 0; i < epochs; i++) {
		timer1.start();
		// TODO do not invoke, due to data-label separation
		//dataSet->shuffleTrainDataSet();
		timer2.start();
		for(int j = 0; j < miniBatchesSize; j++) {
			//if((j+1)%100 == 0) {
			//	cout << "Minibatch " << j+1 << " started: " << timer2.stop(false) << endl;
			//	timer2.start();
			//}
			//cout << "Minibatch " << j+1 << " started: " << timer2.stop(false) << endl;
			//timer2.start();

			//cout << "reset_nabla()" << endl;
			inputLayer->reset_nabla(0);
			updateMiniBatch(j);
		}
		//timer1.stop();

		//dataSet->shuffleTestDataSet();
		int accurateCnt = 0;
		float cost = 0.0f;
		float accuracy = 0.0f;
		int numTestData = dataSet->getNumTestData();
		evaluate(accurateCnt, cost);
		cost /= numTestData;
		accuracy = (float)accurateCnt/numTestData;

		if(dataSet->getNumTestData() > 0) {
			cout << "Epoch " << i+1 << " " << accurateCnt << " / " << numTestData
					<< ", accuracy: " << accuracy << ", cost: " << cost << " :" << timer1.stop(false) << endl;

			if(networkListener) {
				networkListener->epochComplete(cost, accuracy);
			}
		}
		else { cout << "Epoch " << i+1 << " complete: " << timer1.stop(false) << endl; }
		//if(accuracy < 0.15) break;
	}
}





void Network::test() {
	Timer timer;
	int accurateCnt = 0;
	float cost = 0.0f;
	float accuracy = 0.0f;
	int numTestData = dataSet->getNumTestData();
	evaluate(accurateCnt, cost);
	cost /= numTestData;
	accuracy = (float)accurateCnt/numTestData;

	if(dataSet->getNumTestData() > 0) {
		timer.start();
		cout << accurateCnt << " / " << numTestData << ", accuracy: " << accuracy << ", cost: " << cost << " :" << timer.stop(false) << endl;
	}
}









void Network::save(string filename) {
	Timer timer;
	timer.start();

	ofstream ofs(filename.c_str(), ios::out | ios::binary);
	//int inputLayerSize = 1;
	int outputLayerSize = outputLayers.size();

	ofs.write((char *)&in_dim, sizeof(io_dim));
	//ofs.write((char *)&inputLayerSize, sizeof(int));		// input layer size
	//ofs.write((char *)&inputLayer, sizeof(Layer *));		// input layer address
	ofs.write((char *)&outputLayerSize, sizeof(UINT));		// output layer size
	for(UINT i = 0; i < outputLayers.size(); i++) {
		ofs.write((char *)&outputLayers[i], sizeof(Layer *));
	}
	inputLayer->save(0, ofs);
	ofs.close();

	cout << "time elapsed to save network: " << timer.stop(false) << endl;
}


void Network::load(string filename) {
	ifstream ifs(filename.c_str(), ios::in | ios::binary);
	UINT outputLayerSize;

	ifs.read((char *)&in_dim, sizeof(in_dim));
	ifs.read((char *)&outputLayerSize, sizeof(UINT));
	for(UINT i = 0; i < outputLayerSize; i++) {
		OutputLayer *outputLayer;
		ifs.read((char *)&outputLayer, sizeof(OutputLayer *));
		outputLayers.push_back(outputLayer);
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




int Network::evaluate(int &accurateCnt, float &cost) {
#if CPU_MODE
	int testResult = 0;
	//bool printBak = Util::getPrint();
	//Util::setPrint(true);
	int testDataSize = dataSet->getNumTestData();
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
#else
	Cuda::refresh();

	//int accurateCnt = 0;
	//float cost = 0;
	//bool printBak = Util::getPrint();
	//Util::setPrint(true);

	int testBatchesSize = dataSet->getNumTestData()/in_dim.batches;
	for(int i = 0; i < testBatchesSize; i++) {

		// FEED FORWARD
		DATATYPE *d_testData;
		checkCudaErrors(cudaMalloc(&d_testData, sizeof(DATATYPE)*inputLayer->getInputDimension()*in_dim.batches));
		//checkCudaErrors(cudaMemcpyAsync(d_testData, dataSet->getTestDataAt(i*in_dim.batches),
		checkCudaErrors(cudaMemcpyAsync(d_testData, dataSet->getTestDataAt(i*in_dim.batches),
				sizeof(DATATYPE)*inputLayer->getInputDimension()*in_dim.batches, cudaMemcpyHostToDevice));

		feedforward(d_testData);
		checkCudaErrors(cudaFree(d_testData));

		//UINT *d_testLabel;
		//checkCudaErrors(cudaMalloc(&d_testLabel, sizeof(UINT)));
		//checkCudaErrors(cudaMemcpyAsync(d_testLabel, dataSet->getTestLabelAt(i),
		//		sizeof(UINT), cudaMemcpyHostToDevice));

		testEvaluateResult(outputLayers[0]->getOutput(), dataSet->getTestLabelAt(i*in_dim.batches), accurateCnt, cost);
		//checkCudaErrors(cudaFree(d_testLabel));
	}

	//Util::setPrint(false);
	return accurateCnt;
#endif
}






#if CPU_MODE




void Network::feedforward(const rcube &input) {
	//cout << "feedforward()" << endl;
	inputLayer->feedforward(0, input);

}



void Network::updateMiniBatch(int nthMiniBatch, int miniBatchSize) {

	int baseIndex = nthMiniBatch*miniBatchSize;
	for(int i = 0; i < miniBatchSize; i++) {
		backprop(dataSet->getTrainDataAt(baseIndex+i));
	}

	int n = dataSet->getTrainData();

	//cout << "update()" << endl;
	//inputLayer->update(0, n, miniBatchSize);
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
		//outputLayers[i]->cost(dataSample.getTarget());
	}
	//cout << "time for backward: ";
	//timer.stop();

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


#else





void Network::feedforward(const DATATYPE *input) {
	//cout << "feedforward()" << endl;
	inputLayer->feedforward(0, input);
}


void Network::updateMiniBatch(int nthMiniBatch) {
	Cuda::refresh();

	int baseIndex = nthMiniBatch*in_dim.batches;

	// FEED FORWARD
	DATATYPE *d_trainData;
	checkCudaErrors(cudaMalloc(&d_trainData, sizeof(DATATYPE)*inputLayer->getInputDimension()*in_dim.batches));
	checkCudaErrors(cudaMemcpyAsync(d_trainData, dataSet->getTrainDataAt(baseIndex),
			sizeof(DATATYPE)*inputLayer->getInputDimension()*in_dim.batches, cudaMemcpyHostToDevice));

	feedforward(d_trainData);

	// BACK PROPAGATION
	UINT *d_trainLabel;
	checkCudaErrors(cudaMalloc(&d_trainLabel, sizeof(UINT)*in_dim.batches));
	checkCudaErrors(cudaMemcpyAsync(d_trainLabel, dataSet->getTrainLabelAt(baseIndex),
				sizeof(UINT)*in_dim.batches, cudaMemcpyHostToDevice));

	for(UINT i = 0; i < outputLayers.size(); i++) {
		outputLayers[i]->cost(d_trainLabel);
	}
	checkCudaErrors(cudaFree(d_trainLabel));
	checkCudaErrors(cudaFree(d_trainData));
	//cudaError_t

	// UPDATE
	//cout << "update()" << endl;
	int n = dataSet->getNumTrainData();
	inputLayer->update(0, n, in_dim.batches);
}





int Network::testEvaluateResult(const DATATYPE *d_output, const UINT *y, int &accurateCnt, float &cost) {
	const int num_labels = 10;
	DATATYPE *output = new DATATYPE[num_labels*in_dim.batches];

	//Util::setPrint(true);
	//Util::printDeviceData(d_output, num_labels, 1, 1, 1, "d_output:");
	//cout << "y for 0: " << y[0] << ", y for 1: " << y[1] << endl;
	//Util::setPrint(false);
	checkCudaErrors(cudaMemcpyAsync(output, d_output,	sizeof(DATATYPE)*num_labels*in_dim.batches, cudaMemcpyDeviceToHost));

	for(int j = 0; j < in_dim.batches; j++) {
		DATATYPE maxValue = -100000;
		int maxIndex = 0;
		for(int i = 0; i < num_labels; i++) {
			//cout << output[i] << ", ";
			if(output[num_labels*j+i] > maxValue) {
				maxValue = output[num_labels*j+i];
				maxIndex = i;
			}

			// cost
			if(i == y[j]) cost += std::abs(output[num_labels*j+i]-1);
			else cost += std::abs(output[num_labels*j+i]);
		}
		//cout << endl << "maxIndex: " << maxIndex << ", y: " << y[0] << endl;

		if(maxIndex == y[j]) accurateCnt++;
	}

	delete [] output;

	return accurateCnt;

	//Util::printVec(&evaluateResult, "result");
	//Util::printVec(y, "y");

	/*
	uword rrow, yrow;
	output.max(rrow);
	y.max(yrow);

	if(rrow == yrow) return 1;
	else return 0;
	*/
}



#endif















