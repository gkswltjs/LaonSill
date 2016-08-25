/*
 * Network.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include "Network.h"

#include <iostream>
#include <vector>
#include <map>

#include "../dataset/DataSet.h"
#include "../layer/LayerFactory.h"
#include "../layer/HiddenLayer.h"
#include "../layer/OutputLayer.h"
#include "../Util.h"


Network::Network(NetworkConfig* config)
	: config(config) {
	DataSet* dataSet = config->_dataSet;
	this->in_dim.rows = dataSet->getRows();
	this->in_dim.cols = dataSet->getCols();
	this->in_dim.channels = dataSet->getChannels();
	this->in_dim.batches = config->_batchSize;

	//this->d_trainLabel = NULL;

	//this->trainData = new Data();
}

/*
Network::Network(NetworkListener *networkListener)
	: Network(0, 0, 0, networkListener) {}

Network::Network(InputLayer *inputLayer, OutputLayer *outputLayer, DataSet *dataSet, NetworkListener *networkListener) {
	this->inputLayer = inputLayer;
	if(outputLayer) this->outputLayers.push_back(outputLayer);
	this->dataSet = dataSet;
	if(networkListener) this->networkListeners.push_back(networkListener);
	this->maxAccuracy = 0.0;
	this->minCost = 100.0;
	this->saveConfigured = false;

	this->dataSetMean[0] = 0.0f;
	this->dataSetMean[1] = 0.0f;
	this->dataSetMean[2] = 0.0f;

	this->clipGradientsLevel = 1000.0f;
}
*/


Network::~Network() {
	if(config->_inputLayer) {
		delete config->_inputLayer;
		config->_inputLayer = NULL;
	}
	//checkCudaErrors(cudaFree(d_trainLabel));
	//checkCudaErrors(cudaFree(d_trainData));

	//delete trainData;
}



void Network::sgd(int epochs) {
	DataSet* dataSet = config->_dataSet;
	vector<Evaluation*>& evaluations = config->_evaluations;
	vector<NetworkListener*>& networkListeners = config->_networkListeners;

	const uint32_t trainDataSize = dataSet->getNumTrainData();
	const uint32_t numBatches = trainDataSize / in_dim.batches;

	Timer timer1;
	Timer timer2;
	for(uint32_t epochIndex = 0; epochIndex < epochs; epochIndex++) {
		Util::train = true;

		//dataSet->shuffleTrainDataSet();
		timer1.start();
		timer2.start();
		for(uint32_t batchIndex = 0; batchIndex < numBatches; batchIndex++) {
			//if((batchIndex+1)%100 == 0) {
			//	cout << "Minibatch " << batchIndex+1 << " started: " << timer2.stop(false) << endl;
			//	timer2.start();
			//}
			//cout << "Minibatch " << batchIndex+1 << " started: " << timer2.stop(false) << endl;
			//timer2.start();
#ifndef GPU_MODE
			inputLayer->reset_nabla(0);
#endif
			trainBatch(batchIndex);
			// UPDATE
			applyUpdate();
		}
		Util::train = false;

		const uint32_t numTestData = dataSet->getNumTestData();
		if(numTestData > 0) {
			evaluateTestSet();

			const float cost = evaluations[0]->getCost() / numTestData;
			const uint32_t accurateCnt = evaluations[0]->getAccurateCount();
			const float accuracy = (float)accurateCnt/numTestData;

			//save();
			cout << "Epoch " << epochIndex+1 << " " << accurateCnt << " / " << numTestData <<
					", accuracy: " << accuracy << ", cost: " << cost <<
					" :" << timer1.stop(false) << endl;

			for(uint32_t nl = 0; nl < networkListeners.size(); nl++) {
				networkListeners[nl]->epochComplete(
						evaluations[nl]->getCost()/numTestData,
						(float)evaluations[nl]->getAccurateCount()/numTestData);
			}
		}
		else {
			cout << "Epoch " << epochIndex+1 << " complete: " << timer1.stop(false) << endl;
		}
	}
}



void Network::evaluateTestSet() {
#ifndef GPU_MODE
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
	DataSet* dataSet = config->_dataSet;
	vector<Evaluation*>& evaluations = config->_evaluations;

	for(int i = 0; i < evaluations.size(); i++) {
		evaluations[i]->reset();
	}

	const uint32_t numBatches = dataSet->getNumTestData()/in_dim.batches;
	for(uint32_t batchIndex = 0; batchIndex < numBatches; batchIndex++) {
		evaluateTestData(batchIndex);
	}
#endif
}

void Network::evaluateTestData(uint32_t batchIndex) {
	config->_inputLayer->feedforward(config->_dataSet->getTestDataAt(batchIndex*in_dim.batches));

	const uint32_t numLabels = config->_outputLayers[0]->getOutDimension().rows;
	Data* networkOutput = config->_outputLayers[0]->getOutput();
	const uint32_t* y = config->_dataSet->getTestLabelAt(batchIndex*in_dim.batches);

	networkOutput->print_data("networkOutput:");
	const DATATYPE* output = networkOutput->host_data();
	for(int i = 0; i < config->_evaluations.size(); i++) {
		config->_evaluations[i]->evaluate(numLabels, in_dim.batches, output, y);
	}
}

void Network::test() {
	Util::train = false;
	DataSet* dataSet = config->_dataSet;
	vector<Evaluation*>& evaluations = config->_evaluations;

	Timer timer;
	float numTestData = (float)dataSet->getNumTestData();
	evaluateTestSet();
	int accurateCnt = evaluations[0]->getAccurateCount();
	float cost = evaluations[0]->getCost() / numTestData;
	float accuracy = accurateCnt / numTestData;

	if(dataSet->getNumTestData() > 0) {
		timer.start();
		cout << accurateCnt << " / " << numTestData << ", accuracy: " << accuracy << ", cost: " << cost << " :" << timer.stop(false) << endl;
	}
}


#ifndef GPU_MODE

void Network::feedforward(const rcube &input, const char *end) {
	//cout << "feedforward()" << endl;
	inputLayer->feedforward(0, input, end);

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

/*
void Network::feedforward(const DATATYPE *input, const char *end) {
	trainData->set_data(input);
	config->_inputLayer->feedforward(0, trainData, end);
}
*/

void Network::trainBatch(uint32_t batchIndex) {
	DataSet* dataSet = config->_dataSet;
	InputLayer* inputLayer = config->_inputLayer;
	vector<OutputLayer*> outputLayers = config->_outputLayers;

	int baseIndex = batchIndex*in_dim.batches;

	// FORWARD PASS
	config->_inputLayer->feedforward(dataSet->getTrainDataAt(baseIndex));

	// BACKWARD PASS
	//checkCudaErrors(cudaMemcpyAsync(d_trainLabel, dataSet->getTrainLabelAt(baseIndex),
	//			sizeof(UINT)*in_dim.batches, cudaMemcpyHostToDevice));

	for(UINT i = 0; i < outputLayers.size(); i++) {
		//outputLayers[i]->cost(d_trainLabel);
		outputLayers[i]->cost(dataSet->getTrainLabelAt(baseIndex));
	}
}

#endif

void Network::applyUpdate() {
	clipGradients();

	uint32_t numLearnableLayers = config->_learnableLayers.size();
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		config->_learnableLayers[i]->update();
	}
}

void Network::clipGradients() {
	const float clipGradientsLevel = config->_clipGradientsLevel;
	const double sumsqParamsGrad = computeSumSquareParamsGrad();
	const double sumsqParamsData = computeSumSquareParamsData();

	const double l2normParamsGrad = std::sqrt(sumsqParamsGrad);
	const double l2normParamsData = std::sqrt(sumsqParamsData);

	if(clipGradientsLevel < 0.0001) {
		cout << "Gradient clipping: no scaling down gradients (L2 norm " << l2normParamsGrad <<
				", Weight: " << l2normParamsData << " <= " << clipGradientsLevel << ")" << endl;
	} else {
		if(l2normParamsGrad > clipGradientsLevel) {
			const DATATYPE scale_factor = clipGradientsLevel / (l2normParamsGrad*1);

			cout << "Gradient clipping: scaling down gradients (L2 norm " << l2normParamsGrad <<
					", Weight: " << l2normParamsData << " > " << clipGradientsLevel <<
					") by scale factor " << scale_factor << endl;
			scaleParamsGrad(scale_factor);
		} else {
			cout << "Gradient clipping: no scaling down gradients (L2 norm " << l2normParamsGrad <<
					", Weight: " << l2normParamsData << " <= " << clipGradientsLevel << ")" << endl;
		}
	}
}

double Network::computeSumSquareParamsData() {
	uint32_t numLearnableLayers = config->_learnableLayers.size();
	double sumsq = 0.0;
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		sumsq += config->_learnableLayers[i]->sumSquareParamsData();
	}
	return sumsq;
}

double Network::computeSumSquareParamsGrad() {
	uint32_t numLearnableLayers = config->_learnableLayers.size();
	double sumsq = 0.0;
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		sumsq += config->_learnableLayers[i]->sumSquareParamsGrad();
	}
	return sumsq;
}

void Network::scaleParamsGrad(DATATYPE scale) {
	uint32_t numLearnableLayers = config->_learnableLayers.size();
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		config->_learnableLayers[i]->scaleParamsGrad(scale);
	}
}

void Network::shape(io_dim in_dim) {
	if(in_dim.unitsize() > 0) {
		this->in_dim = in_dim;
	}
	config->_inputLayer->shape(0, this->in_dim);

	//checkCudaErrors(Util::ucudaMalloc(&d_trainData, sizeof(DATATYPE)*config->_inputLayer->getInputSize()*this->in_dim.batches));
	//checkCudaErrors(Util::ucudaMalloc(&d_trainLabel, sizeof(UINT)*this->in_dim.batches));

	//trainData->reshape({this->in_dim.batches, this->in_dim.channels, this->in_dim.rows, this->in_dim.cols});
}

void Network::reshape(io_dim in_dim) {
	if(in_dim.unitsize() > 0) {
		this->in_dim = in_dim;
	}
	config->_inputLayer->reshape(0, this->in_dim);
}

/*
void Network::saveConfig(const char* savePrefix) {
	strcpy(this->savePrefix, savePrefix);
	this->saveConfigured = true;
}
*/

void Network::save(const char* filename) {
	//if(saveConfigured && cost < minCost) {
	//	minCost = cost;
	//	char savePath[256];
	//	sprintf(savePath, "%s%02d.network", savePrefix, i+1);
	//	save(savePath);

	InputLayer* inputLayer = config->_inputLayer;
	vector<OutputLayer*>& outputLayers = config->_outputLayers;

	Timer timer;
	timer.start();

	ofstream ofs(filename, ios::out | ios::binary);
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

void Network::load(const char* filename) {
	InputLayer* inputLayer = config->_inputLayer;
	vector<OutputLayer*>& outputLayers = config->_outputLayers;

	ifstream ifs(filename, ios::in | ios::binary);
	UINT outputLayerSize;

	ifs.read((char *)&in_dim, sizeof(in_dim));
	ifs.read((char *)&outputLayerSize, sizeof(UINT));
	for(UINT i = 0; i < outputLayerSize; i++) {
		OutputLayer *outputLayer;
		ifs.read((char *)&outputLayer, sizeof(OutputLayer *));
		outputLayers.push_back(outputLayer);
	}

	map<Layer *, Layer *> layerMap;
	config->_inputLayer = new InputLayer();
	config->_inputLayer->load(ifs, layerMap);

	// restore output layers of network
	for(UINT i = 0; i < outputLayerSize; i++) {
		outputLayers[i] = (OutputLayer *)layerMap.find((Layer *)outputLayers[i])->second;
	}

	/*
	map<Layer *, Layer *> layerMap;
	while(true) {
		Layer::Type layerType;
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
			Layer *nextLayer = nextLayers[i];
			nextLayers[i] = layerMap.find(nextLayer)->second;
		}

		// 학습된 네트워크를 load하는 경우 backward pass가 없으므로 불필요
		//HiddenLayer *hiddenLayer = dynamic_cast<HiddenLayer *>(layer);
		//if(hiddenLayer) {
		//	vector<prev_layer_relation> &prevLayers = hiddenLayer->getPrevLayers();
		//	for(UINT i = 0; i < prevLayers.size(); i++) {
		//		Layer *prevLayer = prevLayers[i];
		//		prevLayers[i] = dynamic_cast<HiddenLayer *>(layerMap.find(prevLayer)->second);
		//	}
		//}

		ifs.read((char *)&layerKey, sizeof(Layer *));
	}
	*/

	ifs.close();
}

/*
DATATYPE Network::getDataSetMean(UINT channel) {
	return dataSetMean[channel];
}

void Network::setDataSetMean(DATATYPE *dataSetMean) {
	for(int i = 0; i < 3; i++) {
		this->dataSetMean[i] = dataSetMean[i];
	}
}
*/

Layer* Network::findLayer(const string name) {
	//return config->_inputLayer->find(0, name);
	map<string, Layer*>& nameLayerMap = config->_nameLayerMap;
	map<string, Layer*>::iterator it = nameLayerMap.find(name);
	if(it != nameLayerMap.end()) {
		return it->second;
	} else {
		return 0;
	}
}




