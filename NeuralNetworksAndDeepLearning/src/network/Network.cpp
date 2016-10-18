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
#include <cfloat>

#include "../dataset/DataSet.h"
#include "../layer/LayerFactory.h"
#include "../layer/HiddenLayer.h"
#include "../layer/OutputLayer.h"
#include "../Util.h"


template <typename Dtype>
Network<Dtype>::Network(NetworkConfig<Dtype>* config)
	: config(config) {
	DataSet<Dtype>* dataSet = config->_dataSet;
	//this->in_dim.rows = dataSet->getRows();
	//this->in_dim.cols = dataSet->getCols();
	//this->in_dim.channels = dataSet->getChannels();
	//this->in_dim.batches = config->_batchSize;
}

/*
Network<Dtype>::Network(NetworkListener *networkListener)
	: Network(0, 0, 0, networkListener) {}

Network<Dtype>::Network(InputLayer<Dtype>* inputLayer, OutputLayer<Dtype>*outputLayer, DataSet *dataSet, NetworkListener *networkListener) {
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

template <typename Dtype>
Network<Dtype>::~Network() {
	if(config->_inputLayer) {
		delete config->_inputLayer;
		config->_inputLayer = NULL;
	}
	//checkCudaErrors(cudaFree(d_trainLabel));
	//checkCudaErrors(cudaFree(d_trainData));

	//delete trainData;
}


template <typename Dtype>
void Network<Dtype>::sgd(int epochs) {
	DataSet<Dtype>* dataSet = config->_dataSet;
	vector<Evaluation<Dtype>*>& evaluations = config->_evaluations;
	vector<NetworkListener*>& networkListeners = config->_networkListeners;

	const uint32_t trainDataSize = dataSet->getNumTrainData();
	const uint32_t numBatches = trainDataSize / config->_inDim.batches;

	Timer timer1;
	Timer timer2;


	//iterations = 0;
	for(uint32_t epochIndex = 0; epochIndex < epochs; epochIndex++) {
		config->_status = NetworkStatus::Train;

		dataSet->shuffleTrainDataSet();
		timer1.start();
		timer2.start();
		for(uint32_t batchIndex = 0; batchIndex < numBatches; batchIndex++) {
			config->_iterations++;
			//iterations++;
			//Util::printMessage("iteration: " + to_string(iterations));

			if((batchIndex+1)%100 == 0) {
				cout << "Minibatch " << batchIndex+1 << " started: " << timer2.stop(false) << endl;
				timer2.start();
			}
			//cout << "Minibatch " << batchIndex+1 << " started: " << timer2.stop(false) << endl;
			//timer2.start();
#ifndef GPU_MODE
			inputLayer->reset_nabla(0);
#endif
			trainBatch(batchIndex);

			//if(iterations >= 1000) {
				//checkAbnormalParam();
			//}
			// UPDATE
			applyUpdate();

			if(config->doTest()) {
				config->_status = NetworkStatus::Test;
				const uint32_t numTestData = dataSet->getNumTestData();
				if(numTestData > 0) {
					double cost = evaluateTestSet();
					cost /= numTestData;

					//const float cost = evaluations[0]->getCost() / numTestData;
					const uint32_t accurateCnt = evaluations[0]->getAccurateCount();
					const float accuracy = (float)accurateCnt/numTestData;

					cout << "epoch: " << epochIndex+1 << ", iteration: " << epochIndex*numBatches+batchIndex+1 << " " << accurateCnt << " / " << numTestData <<
							", accuracy: " << accuracy << ", cost: " << cost <<
							" :" << timer1.stop(false) << endl;

					for(uint32_t nl = 0; nl < networkListeners.size(); nl++) {
						networkListeners[nl]->onAccuracyComputed(0, "top1_accuracy", (double)evaluations[0]->getAccurateCount()/numTestData*100);
						networkListeners[nl]->onAccuracyComputed(1, "top5_accuracy", (double)evaluations[1]->getAccurateCount()/numTestData*100);
						//networkListeners[nl]->onCostComputed(0, "cost", evaluations[0]->getCost()/numTestData);
						networkListeners[nl]->onCostComputed(0, "cost", cost);
					}
				}
				config->_status = NetworkStatus::Train;
			}

			if(config->doSave()) {
				save();
			}
		}

		/*
		//if((epochIndex+1) % 1 == 0) {
			config->_status = NetworkStatus::Test;
			const uint32_t numTestData = dataSet->getNumTestData();
			if(numTestData > 0) {
				double cost = evaluateTestSet();
				cost /= numTestData;

				//const float cost = evaluations[0]->getCost() / numTestData;
				const uint32_t accurateCnt = evaluations[0]->getAccurateCount();
				const float accuracy = (float)accurateCnt/numTestData;

				//save();
				cout << "Epoch " << epochIndex+1 << " " << accurateCnt << " / " << numTestData <<
						", accuracy: " << accuracy << ", cost: " << cost <<
						" :" << timer1.stop(false) << endl;

				for(uint32_t nl = 0; nl < networkListeners.size(); nl++) {
					networkListeners[nl]->onAccuracyComputed(0, "top1_accuracy", (double)evaluations[0]->getAccurateCount()/numTestData*100);
					networkListeners[nl]->onAccuracyComputed(1, "top5_accuracy", (double)evaluations[1]->getAccurateCount()/numTestData*100);
					//networkListeners[nl]->onCostComputed(0, "cost", evaluations[0]->getCost()/numTestData);
					networkListeners[nl]->onCostComputed(0, "cost", cost);
				}
			}
			else {
				cout << "Epoch " << epochIndex+1 << " complete: " << timer1.stop(false) << endl;
			}
		//}
		 */

	}
}


template <typename Dtype>
double Network<Dtype>::evaluateTestSet() {
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
	DataSet<Dtype>* dataSet = config->_dataSet;
	vector<Evaluation<Dtype>*>& evaluations = config->_evaluations;
	double cost = 0.0;

	for(int i = 0; i < evaluations.size(); i++) {
		evaluations[i]->reset();
	}

	const uint32_t numBatches = dataSet->getNumTestData()/config->_inDim.batches;
	//cout << "numTestData: " << dataSet->getNumTestData() << ", batches: " << in_dim.batches << ", numBatches: " << numBatches << endl;
	for(uint32_t batchIndex = 0; batchIndex < numBatches; batchIndex++) {
		cost += evaluateTestData(batchIndex);
	}
	return cost;
#endif
}

template <typename Dtype>
double Network<Dtype>::evaluateTestData(uint32_t batchIndex) {
	const uint32_t baseIndex = batchIndex*config->_inDim.batches;

	//config->_inputLayer->feedforward(config->_dataSet->getTestDataAt(batchIndex*in_dim.batches));
	config->_inputLayer->feedforward(config->_dataSet, baseIndex);
	OutputLayer<Dtype>* outputLayer = config->_outputLayers[0];

	const uint32_t numLabels = outputLayer->getOutDimension().rows;
	Data<Dtype>* networkOutput = outputLayer->getOutput();

	//const uint32_t* y = config->_dataSet->getTestLabelAt(batchIndex*in_dim.batches);
	//double cost = outputLayer->cost(y);
	double cost = outputLayer->cost(config->_dataSet, baseIndex);

	networkOutput->print_data("networkOutput:");
	const Dtype* output = networkOutput->host_data();
	for(int i = 0; i < config->_evaluations.size(); i++) {
		config->_evaluations[i]->evaluate(numLabels, config->_inDim.batches, output, config->_dataSet, baseIndex);
	}
	//cout << "cost at " << batchIndex << " " << cost << endl;

	return cost;
}

template <typename Dtype>
void Network<Dtype>::test() {
	config->_status = NetworkStatus::Test;
	DataSet<Dtype>* dataSet = config->_dataSet;
	vector<Evaluation<Dtype>*>& evaluations = config->_evaluations;

	Timer timer;
	float numTestData = (float)dataSet->getNumTestData();
	double cost = evaluateTestSet();
	cost /= numTestData;
	int accurateCnt = evaluations[0]->getAccurateCount();
	float accuracy = accurateCnt / numTestData;

	if(dataSet->getNumTestData() > 0) {
		timer.start();
		cout << accurateCnt << " / " << numTestData << ", accuracy: " << accuracy << ", cost: " << cost << " :" << timer.stop(false) << endl;
	}
}


#ifndef GPU_MODE
template <typename Dtype>
void Network<Dtype>::feedforward(const rcube &input, const char *end) {
	//cout << "feedforward()" << endl;
	inputLayer->feedforward(0, input, end);

}

template <typename Dtype>
void Network<Dtype>::updateMiniBatch(int nthMiniBatch, int miniBatchSize) {

	int baseIndex = nthMiniBatch*miniBatchSize;
	for(int i = 0; i < miniBatchSize; i++) {
		backprop(dataSet->getTrainDataAt(baseIndex+i));
	}

	int n = dataSet->getTrainData();

	//cout << "update()" << endl;
	//inputLayer->update(0, n, miniBatchSize);
}

template <typename Dtype>
void Network<Dtype>::backprop(const DataSample &dataSample) {
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
double Network<Dtype>::totalCost(const vector<const DataSample *> &dataSet, double lambda) {
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

double Network<Dtype>::accuracy(const vector<const DataSample *> &dataSet) {
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

template <typename Dtype>
int Network<Dtype>::testEvaluateResult(const rvec &output, const rvec &y) {
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
void Network<Dtype>::feedforward(const Dtype *input, const char *end) {
	trainData->set_data(input);
	config->_inputLayer->feedforward(0, trainData, end);
}
*/

template <typename Dtype>
void Network<Dtype>::trainBatch(uint32_t batchIndex) {
	DataSet<Dtype>* dataSet = config->_dataSet;
	InputLayer<Dtype>* inputLayer = config->_inputLayer;
	vector<OutputLayer<Dtype>*> outputLayers = config->_outputLayers;

	int baseIndex = batchIndex*config->_inDim.batches;

	// FORWARD PASS
	//config->_inputLayer->feedforward(dataSet->getTrainDataAt(baseIndex));
	config->_inputLayer->feedforward(dataSet, baseIndex);

	// BACKWARD PASS
	for(UINT i = 0; i < outputLayers.size(); i++) {
		//outputLayers[i]->backpropagation(dataSet->getTrainLabelAt(baseIndex));
		outputLayers[i]->backpropagation(dataSet, baseIndex);
	}
}

#endif

template <typename Dtype>
void Network<Dtype>::applyUpdate() {

	/*
	for(uint32_t i = 0; i < config->_layers.size()-1; i++) {
		Layer<Dtype>* layer = config->_layers[i];
		if(layer->_input->is_nan_data()) {
			cout << layer->getName() << " input is nan data " << endl;
		}
	}
	for(uint32_t i = config->_layers.size()-2; i >= 0; i--) {
		Layer<Dtype>* layer = config->_layers[i];
		if(layer->_input->is_nan_grad()) {
			cout << layer->getName() << " input is nan grad " << endl;
		}
	}
	cout << "------------------------" << endl;
	*/

	//checkLearnableParamIsNan();
	clipGradients();

	const uint32_t numLearnableLayers = config->_learnableLayers.size();
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		config->_learnableLayers[i]->update();
	}
}

template <typename Dtype>
void Network<Dtype>::clipGradients() {
	const float clipGradientsLevel = config->_clipGradientsLevel;
	const double sumsqParamsGrad = computeSumSquareParamsGrad();
	const double sumsqParamsData = computeSumSquareParamsData();

	const double l2normParamsGrad = std::sqrt(sumsqParamsGrad);
	const double l2normParamsData = std::sqrt(sumsqParamsData);

	if(clipGradientsLevel < 0.0001) {
		//cout << "Gradient clipping: no scaling down gradients (L2 norm " << l2normParamsGrad <<
		//		", Weight: " << l2normParamsData << " <= " << clipGradientsLevel << ")" << endl;
	} else {
		if(l2normParamsGrad > clipGradientsLevel) {
			const float scale_factor = clipGradientsLevel / (l2normParamsGrad*1);

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

template <typename Dtype>
double Network<Dtype>::computeSumSquareParamsData() {
	uint32_t numLearnableLayers = config->_learnableLayers.size();
	double sumsq = 0.0;
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		double temp = config->_learnableLayers[i]->sumSquareParamsData();
		//if(i >= numLearnableLayers-10) { // && i < numLearnableLayers-1) {
		if(i < 0) {
			config->_networkListeners[0]->onDataSumsqComputed(
					//i-(numLearnableLayers-10),
					i,
					config->_learnableLayers[i]->getName(),
					std::sqrt(temp));
		}
		sumsq += temp;
	}
	return sumsq;
}

template <typename Dtype>
double Network<Dtype>::computeSumSquareParamsGrad() {
	uint32_t numLearnableLayers = config->_learnableLayers.size();
	double sumsq = 0.0;
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		double temp = config->_learnableLayers[i]->sumSquareParamsGrad();
		//if(i < 10) {
		if(i < 0) {
			config->_networkListeners[0]->onGradSumsqComputed(
					i,
					config->_learnableLayers[i]->getName(),
					std::sqrt(temp));
		}
		sumsq += temp;
		//cout << config->_learnableLayers[i]->getName() << ", grad l2-norm: " << std::sqrt(temp) << endl;
	}
	return sumsq;
}

template <typename Dtype>
void Network<Dtype>::scaleParamsGrad(float scale) {
	uint32_t numLearnableLayers = config->_learnableLayers.size();
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		config->_learnableLayers[i]->scaleParamsGrad(scale);
	}
}




/*
template <typename Dtype>
void Network<Dtype>::checkAbnormalParam() {
	const uint32_t numLearnableLayers = config->_learnableLayers.size();
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		double testResult = config->_learnableLayers[i]->testParamAbnormality();

		if(testResult < DBL_MAX-1) {

			Layer<Dtype>* layer = dynamic_cast<Layer<Dtype>*>(config->_learnableLayers[i]);
			cout << layer->getName() << " test failed ... : " << testResult << endl;

			Data<Dtype>::printConfig = 1;

			cout << "checkAbnormalParam ... " << endl;
			FullyConnectedLayer<Dtype>* fullyConnectedLayer = dynamic_cast<FullyConnectedLayer<Dtype>*>(config->_learnableLayers[i]);
			if(fullyConnectedLayer) {
				fullyConnectedLayer->_params[0]->print_grad(fullyConnectedLayer->getName()+" at "+to_string(testResult));
			} else {
				ConvLayer<Dtype>* convLayer = dynamic_cast<ConvLayer<Dtype>*>(config->_learnableLayers[i]);
				if(convLayer) {
					convLayer->_params[0]->print_grad(convLayer->getName()+" at "+to_string(testResult));
				}
			}

			cout << "print input, output of layers ... " << endl;
			const uint32_t numLayers = config->_layers.size();
			for(uint32_t j = 0; j < numLayers; j++) {
				config->_layers[i]->_input->print_data(config->_layers[i]->getName()+": inputData:");
				config->_layers[i]->_output->print_data(config->_layers[i]->getName()+": outputData:");
				config->_layers[i]->_input->print_grad(config->_layers[i]->getName()+": inputGrad:");
				config->_layers[i]->_output->print_grad(config->_layers[i]->getName()+": outputGrad:");
			}

			Data<Dtype>::printConfig = 0;

			exit(1);
		}
	}
}
*/


template <typename Dtype>
void Network<Dtype>::checkLearnableParamIsNan() {

	/*
	const uint32_t numLayers = config->_layers.size();
	for(uint32_t i = 0; i < numLayers; i++) {
		Layer<Dtype>* layer = config->_layers[i];

		if(layer) {
			if(layer->_input->is_nan_data()) {
				cout << layer->getName() << " input data is nan data ... " << endl;
			}
			if(layer->_output->is_nan_data()) {
				cout << layer->getName() << " output data is nan data ... " << endl;
			}
		}
	}


	for(uint32_t i = numLayers-1; i >= 0; i++) {

	}
	*/



	/*
	const uint32_t numLearnableLayers = config->_learnableLayers.size();
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		FullyConnectedLayer<Dtype>* fullyConnectedLayer = dynamic_cast<FullyConnectedLayer<Dtype>*>(config->_learnableLayers[i]);
		if(fullyConnectedLayer) {
			if(fullyConnectedLayer->_params[0]->is_nan_grad()) {
				cout << fullyConnectedLayer->getName() << " is nan grad ... " << endl;
			}
		} else {
			ConvLayer<Dtype>* convLayer = dynamic_cast<ConvLayer<Dtype>*>(config->_learnableLayers[i]);
			if(convLayer) {
				if(convLayer->_params[0]->is_nan_grad()) {
					cout << convLayer->getName() << " is nan grad ... " << endl;
				}
			}
		}
	}
	*/
}

















/*
template <typename Dtype>
void Network<Dtype>::shape(io_dim in_dim) {
	if(in_dim.unitsize() > 0) {
		this->in_dim = in_dim;
	}
	config->_inputLayer->shape(0, this->in_dim);

	//checkCudaErrors(Util::ucudaMalloc(&d_trainData, sizeof(Dtype)*config->_inputLayer->getInputSize()*this->in_dim.batches));
	//checkCudaErrors(Util::ucudaMalloc(&d_trainLabel, sizeof(UINT)*this->in_dim.batches));
	//trainData->reshape({this->in_dim.batches, this->in_dim.channels, this->in_dim.rows, this->in_dim.cols});
}
*/

template <typename Dtype>
void Network<Dtype>::reshape(io_dim in_dim) {
	if(in_dim.unitsize() > 0) {
		this->config->_inDim = in_dim;
	}
	config->_inputLayer->reshape(0, this->config->_inDim);
}

/*
void Network<Dtype>::saveConfig(const char* savePrefix) {
	strcpy(this->savePrefix, savePrefix);
	this->saveConfigured = true;
}
*/

template <typename Dtype>
void Network<Dtype>::save() {
	//if(saveConfigured && cost < minCost) {
	//	minCost = cost;
	//	char savePath[256];
	//	sprintf(savePath, "%s%02d.network", savePrefix, i+1);
	//	save(savePath);


	/*
	InputLayer<Dtype>* inputLayer = config->_inputLayer;
	vector<OutputLayer<Dtype>*>& outputLayers = config->_outputLayers;

	Timer timer;
	timer.start();

	ofstream ofs(filename, ios::out | ios::binary);
	//int inputLayerSize = 1;
	int outputLayerSize = outputLayers.size();

	ofs.write((char *)&in_dim, sizeof(io_dim));
	//ofs.write((char *)&inputLayerSize, sizeof(int));		// input layer size
	//ofs.write((char *)&inputLayer, sizeof(Layer<Dtype>*));		// input layer address
	ofs.write((char *)&outputLayerSize, sizeof(UINT));		// output layer size
	for(UINT i = 0; i < outputLayers.size(); i++) {
		ofs.write((char *)&outputLayers[i], sizeof(Layer<Dtype>*));
	}
	inputLayer->save(0, ofs);
	ofs.close();

	cout << "time elapsed to save network: " << timer.stop(false) << endl;
	*/

	config->save();

}



/*
template <typename Dtype>
void Network<Dtype>::load(const char* filename) {
	InputLayer<Dtype>* inputLayer = config->_inputLayer;
	vector<OutputLayer<Dtype>*>& outputLayers = config->_outputLayers;

	ifstream ifs(filename, ios::in | ios::binary);
	UINT outputLayerSize;

	ifs.read((char *)&in_dim, sizeof(in_dim));
	ifs.read((char *)&outputLayerSize, sizeof(UINT));
	for(UINT i = 0; i < outputLayerSize; i++) {
		OutputLayer<Dtype>* outputLayer;
		ifs.read((char *)&outputLayer, sizeof(OutputLayer<Dtype>*));
		outputLayers.push_back(outputLayer);
	}

	map<Layer<Dtype>*, Layer<Dtype>*> layerMap;
	config->_inputLayer = new InputLayer<Dtype>();
	config->_inputLayer->load(ifs, layerMap);

	// restore output layers of network
	for(UINT i = 0; i < outputLayerSize; i++) {
		outputLayers[i] = (OutputLayer<Dtype>*)layerMap.find((Layer<Dtype>*)outputLayers[i])->second;
	}*/

	/*
	map<Layer<Dtype>*, Layer<Dtype>*> layerMap;
	while(true) {
		Layer::Type layerType;
		ifs.read((char *)&layerType, sizeof(int));
		Layer<Dtype>*address;
		ifs.read((char *)&address, sizeof(Layer<Dtype>*));

		if(address == 0) break;

		Layer<Dtype>*layer = LayerFactory<Dtype>::create(layerType);
		layerMap.insert(pair<Layer<Dtype>*, Layer<Dtype>*>(address, layer));
	}
	cout << "map size: " << layerMap.size() << endl;

	ifs.seekg(1000);
	Layer<Dtype>*layerKey;
	ifs.read((char *)&layerKey, sizeof(Layer<Dtype>*));

	while(ifs) {
		Layer<Dtype>*layer = layerMap.find(layerKey)->second;
		if(!layer) throw Exception();

		layer->load(ifs, layerMap);

		vector<next_layer_relation> &nextLayers = layer->getNextLayers();
		for(UINT i = 0; i < nextLayers.size(); i++) {
			Layer<Dtype>*nextLayer = nextLayers[i];
			nextLayers[i] = layerMap.find(nextLayer)->second;
		}

		// 학습된 네트워크를 load하는 경우 backward pass가 없으므로 불필요
		//HiddenLayer<Dtype>*hiddenLayer = dynamic_cast<HiddenLayer<Dtype>*>(layer);
		//if(hiddenLayer) {
		//	vector<prev_layer_relation> &prevLayers = hiddenLayer->getPrevLayers();
		//	for(UINT i = 0; i < prevLayers.size(); i++) {
		//		Layer<Dtype>*prevLayer = prevLayers[i];
		//		prevLayers[i] = dynamic_cast<HiddenLayer<Dtype>*>(layerMap.find(prevLayer)->second);
		//	}
		//}

		ifs.read((char *)&layerKey, sizeof(Layer<Dtype>*));
	}
	*/
/*
	ifs.close();
}
*/

/*
Dtype Network<Dtype>::getDataSetMean(UINT channel) {
	return dataSetMean[channel];
}

void Network<Dtype>::setDataSetMean(Dtype *dataSetMean) {
	for(int i = 0; i < 3; i++) {
		this->dataSetMean[i] = dataSetMean[i];
	}
}
*/

template <typename Dtype>
Layer<Dtype>* Network<Dtype>::findLayer(const string name) {
	//return config->_inputLayer->find(0, name);
	map<string, Layer<Dtype>*>& nameLayerMap = config->_nameLayerMap;
	typename map<string, Layer<Dtype>*>::iterator it = nameLayerMap.find(name);
	if(it != nameLayerMap.end()) {
		return it->second;
	} else {
		return 0;
	}
}



template class Network<float>;
