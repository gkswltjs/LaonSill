/*
 * Network.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include <vector>
#include <map>
#include <cfloat>

#include "DataSet.h"
#include "Layer.h"
#include "SoftmaxWithLossLayer.h"
#include "LossLayer.h"
#include "Util.h"
#include "Worker.h"
#include "Perf.h"
#include "StdOutLog.h"
#include "Network.h"
#include "SysLog.h"
#include "DebugUtil.h"

using namespace std;


template<typename Dtype>
atomic<int>         Network<Dtype>::networkIDGen;
template<typename Dtype>
map<int, Network<Dtype>*> Network<Dtype>::networkIDMap;
template<typename Dtype>
mutex Network<Dtype>::networkIDMapMutex;

template <typename Dtype>
Network<Dtype>::Network(NetworkConfig<Dtype>* config)
	: config(config) {
	//DataSet<Dtype>* dataSet = config->_dataSet;
	//this->in_dim.rows = dataSet->getRows();
	//this->in_dim.cols = dataSet->getCols();
	//this->in_dim.channels = dataSet->getChannels();
	//this->in_dim.batches = config->_batchSize;
    //
    this->networkID = atomic_fetch_add(&Network<Dtype>::networkIDGen, 1);
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    Network<Dtype>::networkIDMap[this->networkID] = this;
}

template<typename Dtype>
void Network<Dtype>::init() {
    atomic_store(&Network<Dtype>::networkIDGen, 0);
}

template<typename Dtype>
Network<Dtype>* Network<Dtype>::getNetworkFromID(int networkID) {
    Network<Dtype>* network;
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    network = Network<Dtype>::networkIDMap[networkID];
    lock.unlock();
    return network;
}

template <typename Dtype>
InputLayer<Dtype>* Network<Dtype>::getInputLayer() {
    LayersConfig<Dtype>* layersConfig = config->layersConfigs[Worker<Dtype>::consumerIdx];
    return dynamic_cast<InputLayer<Dtype>*>(layersConfig->_firstLayers[0]);
}

template <typename Dtype>
LayersConfig<Dtype>* Network<Dtype>::getLayersConfig() {
    return config->layersConfigs[Worker<Dtype>::consumerIdx];
}

template <typename Dtype>
void Network<Dtype>::setLayersConfig(LayersConfig<Dtype>* layersConfig) {
    config->layersConfigs[Worker<Dtype>::consumerIdx] = layersConfig;
}


template <typename Dtype>
Network<Dtype>::~Network() {
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    Network<Dtype>::networkIDMap.erase(this->networkID);
}

template <typename Dtype>
void Network<Dtype>::sgd_with_timer(int epochs) {
    struct timespec startTime;
    SPERF_START(NETWORK_TRAINING_TESTTIME, &startTime);
    // XXX: 임시
    //epochs = 1000000;
    //epochs = 1;
	sgd(epochs);

    SPERF_END(NETWORK_TRAINING_TESTTIME, startTime, epochs);
    STDOUT_BLOCK(cout << "Total Training Time : " << SPERF_TIME(NETWORK_TRAINING_TESTTIME)
                    << endl;);
}

#define SAVE_PROPOSAL_TARGET_LAYER 0

// batch 단위로 sgd()를 수행하기 위한 함수.
// GAN 디버깅을 위한 임시 함수. 추후에 정리 필요.
template<typename Dtype>
Dtype Network<Dtype>::sgdMiniBatch(uint32_t batchTotalIndex) {
    InputLayer<Dtype>* inputLayer = getLayersConfig()->_inputLayer;
	//vector<vector<Evaluation<Dtype>*>>& evaluations = config->_evaluations;
	vector<NetworkListener*>& networkListeners = config->_networkListeners;

	const uint32_t trainDataSize = inputLayer->getNumTrainData();
	//cout << "trainDataSize: " << trainDataSize << endl;
	const uint32_t numBatches = 
        trainDataSize / config->_batchSize / Worker<Dtype>::consumerCount;

    config->_status = NetworkStatus::Train;

    if (batchTotalIndex == 0)
        inputLayer->shuffleTrainDataSet();

    LayersConfig<Dtype>* layersConfig = getLayersConfig();
    vector<double> costList(config->_lossLayers.size());
    typename map<string, Layer<Dtype>*>::iterator it;

    uint32_t batchIndex = batchTotalIndex * Worker<Dtype>::consumerCount +
        Worker<Dtype>::consumerIdx;
    config->_iterations++;

#ifndef GPU_MODE
    inputLayer->reset_nabla(0);
#endif
    trainBatch(batchIndex);

    // UPDATE
    if (config->_phase == NetworkPhase::TrainPhase) {
        for (uint32_t i = 0; i < config->_lossLayers.size(); i++) {
            it = layersConfig->_nameLayerMap.find(config->_lossLayers[i]);
            assert(it != layersConfig->_nameLayerMap.end());
            LossLayer<Dtype>* lossLayer = dynamic_cast<LossLayer<Dtype>*>(it->second);
            assert(lossLayer != 0);
            costList[i] += lossLayer->cost();
        }
        applyUpdate();
    }

    float lossSum = 0.0;
    // 모든 worker에서 GPU 트레이닝이 끝나길 기다린다.
    // XXX: 예쁘게.. 
    if (Worker<Dtype>::waitPeer()) {
        // 마지막 쓰레드가 메모리를 갱신한다.
        if (config->_phase == NetworkPhase::TrainPhase && config->doTest()) {
            config->_status = NetworkStatus::Test;

            for (uint32_t i = 0; i < config->_lossLayers.size(); i++) {
                float cost = costList[i]/config->_testInterval;
                networkListeners[i]->onCostComputed(0, config->_lossLayers[i], cost);
                costList[i] = 0.0;
                //cout << config->_lossLayers[i] << " cost:" << cost << ",";
                lossSum += cost;
            }
            //cout << endl;

            config->_status = NetworkStatus::Train;
        }

        if(config->_phase == NetworkPhase::TrainPhase && config->doSave()) {
            save();
        }

        Worker<Dtype>::wakeupPeer();
    }

    return lossSum;
}

template <typename Dtype>
Dtype Network<Dtype>::sgd(int epochs) {
    InputLayer<Dtype>* inputLayer = getLayersConfig()->_inputLayer;
	//vector<vector<Evaluation<Dtype>*>>& evaluations = config->_evaluations;
	vector<NetworkListener*>& networkListeners = config->_networkListeners;

	const uint32_t trainDataSize = inputLayer->getNumTrainData();
	//cout << "trainDataSize: " << trainDataSize << endl;
	const uint32_t numBatches = 
        trainDataSize / config->_batchSize / Worker<Dtype>::consumerCount;

    float lossSum = 0.0;
	//iterations = 0;
	for (uint32_t epochIndex = 0; epochIndex < epochs; epochIndex++) {
		config->_status = NetworkStatus::Train;

		STDOUT_BLOCK(cout << "epochIndex: " << epochIndex << ", epochs: " << epochs << endl;);

		inputLayer->shuffleTrainDataSet();

        // GPU가 여러대 있는 경우에 한대의 GPU가 하나의 batch에 해당하는
        // 데이터를 트레이닝한다.
        // XXX: GPU 대수는 numBatches의 최소공약수라 가정한다. 
        //      (나중에 고쳐야 한다.)


		LayersConfig<Dtype>* layersConfig = getLayersConfig();
		vector<double> costList(config->_lossLayers.size());
		typename map<string, Layer<Dtype>*>::iterator it;


#if SAVE_PROPOSAL_TARGET_LAYER
		ofstream ofs(config->_savePathPrefix + "/proposal_target_layer.ptl",
            ios::out | ios::binary);
		const uint32_t numData = numBatches*5;
		ofs.write((char*)&numData, sizeof(uint32_t));
#endif
		double cost;
		for (uint32_t batchTotalIndex = 0; batchTotalIndex < numBatches; batchTotalIndex++) {
            uint32_t batchIndex = batchTotalIndex * Worker<Dtype>::consumerCount +
                Worker<Dtype>::consumerIdx;
			config->_iterations++;

#ifndef GPU_MODE
			inputLayer->reset_nabla(0);
#endif
			trainBatch(batchIndex);



#if SAVE_PROPOSAL_TARGET_LAYER
			saveProposalTargets(ofs);
#endif
			// UPDATE
			if (config->_phase == NetworkPhase::TrainPhase) {
				for (uint32_t i = 0; i < config->_lossLayers.size(); i++) {
					it = layersConfig->_nameLayerMap.find(config->_lossLayers[i]);
					assert(it != layersConfig->_nameLayerMap.end());
					LossLayer<Dtype>* lossLayer = dynamic_cast<LossLayer<Dtype>*>(it->second);
					assert(lossLayer != 0);
					costList[i] += lossLayer->cost();

					//costList[i] = lossLayer->cost();
					//networkListeners[i]->onCostComputed(0, config->_lossLayers[i], costList[i]);
				}
				applyUpdate();
			}

            // 모든 worker에서 GPU 트레이닝이 끝나길 기다린다.
            // XXX: 예쁘게.. 
            if (Worker<Dtype>::waitPeer()) {
                // 마지막 쓰레드가 메모리를 갱신한다.
                if (config->_phase == NetworkPhase::TrainPhase && config->doTest()) {
                    config->_status = NetworkStatus::Test;
					for (uint32_t i = 0; i < config->_lossLayers.size(); i++) {
						float cost = costList[i]/config->_testInterval;
						networkListeners[i]->onCostComputed(i, config->_lossLayers[i], cost);
						costList[i] = 0.0;
						STDOUT_BLOCK(cout << config->_lossLayers[i] << " cost:" << cost << ",";);
                        lossSum += cost;
					}
					cout << endl;
                    config->_status = NetworkStatus::Train;
                }

                DebugUtil<Dtype>::printNetworkEdges(stdout, "ETRI network", layersConfig, 0);


                if(config->_phase == NetworkPhase::TrainPhase && config->doSave()) {
                    save();
                }

                Worker<Dtype>::wakeupPeer();
            }
		}

#if SAVE_PROPOSAL_TARGET_LAYER
		ofs.close();
#endif


	}

    return lossSum;
}


template <typename Dtype>
double Network<Dtype>::evaluateTestSet() {
    InputLayer<Dtype>* inputLayer = getLayersConfig()->_inputLayer;
	//vector<vector<Evaluation<Dtype>*>>& evaluations = config->_evaluations;
	double cost = 0.0;

	//for (uint32_t i = 0; i < evaluations.size(); i++) {
	//	for (uint32_t j = 0; j < evaluations[i].size(); j++)
	//		evaluations[i][j]->reset();
	//}

	vector<double> costList;
	const uint32_t numBatches = inputLayer->getNumTestData()/config->_batchSize;
	for (uint32_t batchIndex = 0; batchIndex < numBatches; batchIndex++) {
		//cost += evaluateTestData(batchIndex);
		evaluateTestData(batchIndex, costList);
	}

	for (uint32_t i = 0; i < config->_lossLayers.size(); i++) {
		cout << costList[i] / numBatches << ", ";
	}
	cout << endl;

	return cost;
}

template <typename Dtype>
double Network<Dtype>::evaluateTestData(uint32_t batchIndex, vector<double>& costList) {
	LayersConfig<Dtype>* layersConfig = getLayersConfig();

#ifndef OUTPUTLAYER
#else
	OutputLayer<Dtype>* lossLayer = layersConfig->_lossLayers[0];
#endif
	int baseIndex = batchIndex*config->_batchSize;

	_feedforward(batchIndex);

	costList.assign(config->_lossLayers.size(), 0.0);

	typename map<string, Layer<Dtype>*>::iterator it;
	for (uint32_t i = 0; i < config->_lossLayers.size(); i++) {
		it = layersConfig->_nameLayerMap.find(config->_lossLayers[i]);
		assert(it != layersConfig->_nameLayerMap.end());

		LossLayer<Dtype>* lossLayer = dynamic_cast<LossLayer<Dtype>*>(it->second);
		assert(lossLayer != 0);

		costList[i] += lossLayer->cost();
	}


	/*
	LossLayer<Dtype>* lossLayer = layersConfig->_lossLayers[0];
	LossLayer<Dtype>* softmaxWithLoss =
			dynamic_cast<SoftmaxWithLossLayer<Dtype>*>(lossLayer);
	assert(softmaxWithLoss);

	const uint32_t numLabels = softmaxWithLoss->prob->getShape(2);
	Data<Dtype>* networkOutput = softmaxWithLoss->prob;

	double cost = lossLayer->cost();

	networkOutput->print_data("networkOutput:");
	const Dtype* output = networkOutput->host_data();
	DataSet<Dtype>* dataSet = layersConfig->_inputLayer->_dataSet;
	for(int i = 0; i < config->_evaluations.size(); i++) {
		config->_evaluations[i]->evaluate(numLabels, config->_batchSize,
				output, dataSet, baseIndex);
	}
	*/

	//return cost;
	return 0.0;
}

template <typename Dtype>
void Network<Dtype>::test() {
	config->_status = NetworkStatus::Test;
	/*
	DataSet<Dtype>* dataSet = getLayersConfig()->_inputLayer->_dataSet;
	vector<vector<Evaluation<Dtype>*>>& evaluations = config->_evaluations;

	Timer timer;
	float numTestData = (float)dataSet->getNumTestData();
	double cost = evaluateTestSet();
	cost /= numTestData;
	int accurateCnt = evaluations[0]->getAccurateCount();
	float accuracy = accurateCnt / numTestData;

	if(dataSet->getNumTestData() > 0) {
		timer.start();
		cout << accurateCnt << " / " << numTestData << ", accuracy: " << accuracy <<
				", cost: " << cost << " :" << timer.stop(false) << endl;
	}
	*/
}


#ifndef GPU_MODE
template <typename Dtype>
void Network<Dtype>::feedforward(const rcube &input, const char *end) {
	//cout << "feedforward()" << endl;
	inputLayer->feedforward(0, input, end);

}

template <typename Dtype>
void Network<Dtype>::updateMiniBatch(int nthMiniBatch, int miniBatchSize) {

#if 0
	int baseIndex = nthMiniBatch*miniBatchSize;
	for(int i = 0; i < miniBatchSize; i++) {
		backprop(dataSet->getTrainDataAt(baseIndex+i));
	}

	int n = dataSet->getTrainData();

	//cout << "update()" << endl;
	//inputLayer->update(0, n, miniBatchSize);
#endif
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
		total += testEvaluateResult(feedforward(dataSample->getData()),
                                    dataSample->getTarget());
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


template <typename Dtype>
void Network<Dtype>::trainBatch(uint32_t batchIndex) {
	_feedforward(batchIndex);

	if (config->_phase == NetworkPhase::TrainPhase)
		_backpropagation(batchIndex);
}

#endif

template <typename Dtype>
void Network<Dtype>::applyUpdate() {
	clipGradients();
	const uint32_t numLearnableLayers = getLayersConfig()->_learnableLayers.size();

    // 모든 worker에서 GPU 트레이닝이 끝나길 기다린다.
    // XXX: 예쁘게.. 
    if (Worker<Dtype>::waitPeer()) {
        typename vector<LayersConfig<Dtype>*>::iterator iter; 
        LayersConfig<Dtype>* firstLayersConfig;
        // 마지막 쓰레드가 learnable layer들의 data를 갱신한다.

        // (1) 변화된 부분을 첫번째 layer에 적용한다.
        for (iter = config->layersConfigs.begin(); iter != config->layersConfigs.end();
            iter++) {
            
            if (iter == config->layersConfigs.begin()) {
                firstLayersConfig = (*iter);
                continue;
            }

            for (uint32_t i = 0; i < numLearnableLayers; i++) {
                LearnableLayer<Dtype>* firstLayer = firstLayersConfig->_learnableLayers[i];
                LearnableLayer<Dtype>* curLayer = (*iter)->_learnableLayers[i];

                curLayer->applyChanges(firstLayer);
            }
        }

        // (2) 첫번째 layer의 값을 다른 layer들에게 동기화 한다.
        if (!Worker<Dtype>::isSingle()) {
            for (iter = config->layersConfigs.begin(); iter != config->layersConfigs.end();
                iter++) {
                if (iter == config->layersConfigs.begin()) {
                    continue;
                }

                for (uint32_t i = 0; i < numLearnableLayers; i++) {
                    LearnableLayer<Dtype>* firstLayer = firstLayersConfig->_learnableLayers[i];
                    LearnableLayer<Dtype>* curLayer = (*iter)->_learnableLayers[i];

                    curLayer->syncParams(firstLayer);
                }
            }
        }

        Worker<Dtype>::wakeupPeer();
    }

    // 각 layer들을 갱신한다.
	for (uint32_t i = 0; i < numLearnableLayers; i++) {
        getLayersConfig()->_learnableLayers[i]->update();
	}
}

/**
 * XXX: 데이터 분산과 관련된 LayersConfigs까지는 고려하지 않았음.
 */
template <typename Dtype>
void Network<Dtype>::syncNetwork(Network<Dtype>* target) {
    // LL : learnable layer
	const uint32_t srcLLCnt = this->getLayersConfig()->_learnableLayers.size();
	const uint32_t dstLLCnt = target->getLayersConfig()->_learnableLayers.size();

    SASSUME((srcLLCnt == dstLLCnt),
        "learnable layer count of target & source network should be same."
        " learnable layer count of source=%d, learnable layer count of target=%d.",
        srcLLCnt, dstLLCnt);

    // LCs : Layers Configs
    const int srcLCsCnt = this->config->layersConfigs.size();
    const int dstLCsCnt = target->config->layersConfigs.size();
    SASSUME((srcLCsCnt == dstLCsCnt),
        "layers configs count of target & source network should be same."
        " layers configs count of target network=%d,"
        " layers configs count of target network=%d.", srcLCsCnt, dstLCsCnt);

    for (int i =0; i < srcLCsCnt; i++) {
        for (uint32_t j = 0; j < srcLLCnt; j++) {
            // LC : Layers Config
            LayersConfig<Dtype>* srcLC = this->config->layersConfigs[i];
            LayersConfig<Dtype>* dstLC = target->config->layersConfigs[i];

            // (1) source network layer의 파라미터 값을 dest network layer의 파라미터 
            //    값으로 동기화 한다.
            dstLC->_learnableLayers[i]->syncParams(srcLC->_learnableLayers[i]);
        }
    }
}

template <typename Dtype>
void Network<Dtype>::clipGradients() {
	const float clipGradientsLevel = config->_clipGradientsLevel;
	const double sumsqParamsGrad = computeSumSquareParamsGrad();
	const double sumsqParamsData = computeSumSquareParamsData();

	const double l2normParamsGrad = sqrt(sumsqParamsGrad);
	const double l2normParamsData = sqrt(sumsqParamsData);

	if (clipGradientsLevel < 0.0001) {
		//cout << "Gradient clipping: no scaling down gradients (L2 norm " << 
        //  l2normParamsGrad << ", Weight: " << l2normParamsData << " <= " << 
        //  clipGradientsLevel << ")" << endl;
	} else {
		if (l2normParamsGrad > clipGradientsLevel) {
			const float scale_factor = clipGradientsLevel / (l2normParamsGrad*1);

			cout << "Gradient clipping: scaling down gradients (L2 norm " << l2normParamsGrad <<
					", Weight: " << l2normParamsData << " > " << clipGradientsLevel <<
					") by scale factor " << scale_factor << endl;
			scaleParamsGrad(scale_factor);
		} else {
			cout << "Gradient clipping: no scaling down gradients (L2 norm "
                << l2normParamsGrad << ", Weight: " << l2normParamsData << " <= "
                << clipGradientsLevel << ")" << endl;
		}
	}
}

template <typename Dtype>
double Network<Dtype>::computeSumSquareParamsData() {
	uint32_t numLearnableLayers = getLayersConfig()->_learnableLayers.size();
	double sumsq = 0.0;
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		double temp = getLayersConfig()->_learnableLayers[i]->sumSquareParamsData();
		//if(i >= numLearnableLayers-10) { // && i < numLearnableLayers-1) {
		if(i < 0) {
			config->_networkListeners[0]->onDataSumsqComputed(
					//i-(numLearnableLayers-10),
					i,
					getLayersConfig()->_learnableLayers[i]->getName(),
					sqrt(temp));
		}
		sumsq += temp;
	}
	return sumsq;
}

template <typename Dtype>
double Network<Dtype>::computeSumSquareParamsGrad() {
	uint32_t numLearnableLayers = getLayersConfig()->_learnableLayers.size();
	double sumsq = 0.0;
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		double temp = getLayersConfig()->_learnableLayers[i]->sumSquareParamsGrad();
		//if(i < 10) {
		if(i < 0) {
			config->_networkListeners[0]->onGradSumsqComputed(
					i,
					getLayersConfig()->_learnableLayers[i]->getName(),
					sqrt(temp));
		}
		sumsq += temp;
		//cout << getLayersConfig()->_learnableLayers[i]->getName() << ", grad l2-norm: " <<
        //  sqrt(temp) << endl;
	}
	return sumsq;
}

template <typename Dtype>
void Network<Dtype>::scaleParamsGrad(float scale) {
	uint32_t numLearnableLayers = getLayersConfig()->_learnableLayers.size();
	for(uint32_t i = 0; i < numLearnableLayers; i++) {
		getLayersConfig()->_learnableLayers[i]->scaleParamsGrad(scale);
	}
}

template <typename Dtype>
void Network<Dtype>::save() {
	config->save();
}

template <typename Dtype>
void Network<Dtype>::loadPretrainedWeights() {
	if (config->_weightsArgs.size() < 1) return;

	const uint32_t numWeightsArgs = config->_weightsArgs.size();

	// load data list from model file
	map<std::string, Data<float>*> dataMap;

	for (uint32_t i = 0; i < numWeightsArgs; i++) {
		ifstream ifs(config->_weightsArgs[i].weightsPath, std::ios::in | std::ios::binary);
		map<string, string>& weightsMap = config->_weightsArgs[i].weightsMap;

		uint32_t numData;
		ifs.read((char*)&numData, sizeof(uint32_t));

		Data<float>::printConfig = true;
		cout << "Load Pretrained Weights ... ----------" << endl;
		for (uint32_t j = 0; j < numData; j++) {
			Data<float>* data = new Data<float>("", true);
			data->load(ifs);

			if (data)
				data->print();

			string dataName;
			if (weightsMap.size() < 1)
				dataName = data->_name;
			else {
				map<string, string>::iterator it;
				it = weightsMap.find(data->_name);
				if (it == weightsMap.end()) {
					dataName = data->_name;
				} else {
					dataName = it->second;
				}
			}

			map<string, Data<float>*>::iterator it;
			it = dataMap.find(dataName);
			if (it != dataMap.end()) {
				cout << dataName << " overwrites ... " << endl;
				delete it->second;
			}

			dataMap[dataName] = data;
			cout << data->_name << " is set to " << dataName << endl;
		}
		cout << "--------------------------------------" << endl;
		Data<float>::printConfig = false;
		ifs.close();
	}

	LayersConfig<Dtype>* layersConfig = getLayersConfig();
	vector<LearnableLayer<Dtype>*> learnableLayers = layersConfig->_learnableLayers;
	const uint32_t numLearnableLayers = learnableLayers.size();

	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		learnableLayers[i]->loadParams(dataMap);
	}

	map<std::string, Data<float>*>::iterator it;
	for (it = dataMap.begin(); it != dataMap.end(); it++)
		delete it->second;
	dataMap.clear();

}

template <typename Dtype>
Layer<Dtype>* Network<Dtype>::findLayer(const string name) {
	//return getLayersConfig()->_inputLayer->find(0, name);
	map<string, Layer<Dtype>*>& nameLayerMap = getLayersConfig()->_nameLayerMap;
	typename map<string, Layer<Dtype>*>::iterator it = nameLayerMap.find(name);
	if(it != nameLayerMap.end()) {
		return it->second;
	} else {
		return 0;
	}
}

template <typename Dtype>
void Network<Dtype>::_feedforward(uint32_t batchIndex) {
	LayersConfig<Dtype>* layersConfig = getLayersConfig();
	InputLayer<Dtype>* inputLayer = layersConfig->_inputLayer;
	int baseIndex = batchIndex*config->_batchSize;

	inputLayer->feedforward(baseIndex);
	for (uint32_t i = 1; i < layersConfig->_layers.size(); i++) {
		//cout << layersConfig->_layers[i]->name << ": feedforward ... " << endl;
		layersConfig->_layers[i]->feedforward();
		//cout << "output sumsq of " << layersConfig->_layers[i]->name << ":\t\t" <<
		//		layersConfig->_layers[i]->_outputData[0]->sumsq_device_data() << endl;
	}
}

template<typename Dtype>
void Network<Dtype>::_forward(string layerName, uint32_t batchIndex) {
	LayersConfig<Dtype>* layersConfig = getLayersConfig();
	typename map<string, Layer<Dtype>*>::iterator it =
        layersConfig->_nameLayerMap.find(layerName);
	SASSERT((it != layersConfig->_nameLayerMap.end()), "invalid layer name. layer name=%s",
        layerName.c_str());

    InputLayer<Dtype>* inputLayer = dynamic_cast<InputLayer<Dtype>*>(it->second);
    if (inputLayer) {
	    int baseIndex = batchIndex*config->_batchSize;
	    inputLayer->feedforward(baseIndex);
    } else {
        Layer<Dtype>* layer = dynamic_cast<Layer<Dtype>*>(it->second);
        SASSERT0(layer != NULL); 
        layer->feedforward();
    }
}

/*
 * @brief       이 함수는 아래의 조건들에 따라서 3가지 동작을 수행한다.
 *              (1) batchCount == 1
 *                - Q network에서 feedforward를 수행
 *                - 결과물을 반환
 *
 *              (2) (batchCount > 1) & isNetQ == false : 
 *                - Q Head Network에서 feedforward를 수행
 *                - 결과값을 반환
 *
 *              (3) (batchCount > 1) & isNetQ == true :
 *                - Q network에서 feedforward를 수행
 *                - 결과값을 반환하지만 caller에서 사용하지 않음
 *                  (결과값은 backward 과정에서 활용)
 *                - (1)과 (3)은 함수의 도작만 본다면 batchCount 외에는 동일
 */
template <typename Dtype>
vector<Data<Dtype>*>& Network<Dtype>::feedForwardDQNNetwork(int batchCount,
        DQNImageLearner<Dtype> *learner, bool isNetQ) {
	LayersConfig<Dtype>* layersConfig = getLayersConfig();
	ALEInputLayer<Dtype>* inputLayer = (ALEInputLayer<Dtype>*)layersConfig->_layers[0];

    // (1) change batch size
    this->config->_batchSize = batchCount;
    inputLayer->reshape();

    // (2) feed forward
    inputLayer->fillData(learner, isNetQ);
	inputLayer->feedforward(0);     // XXX: mini batch의 개수가 1이다. 
                                    //      그래서 batch index값을 0으로만 넣고 있다.
                                    //      여러개의 mini batch를 가질 수 있도록 수정하자.

	for (uint32_t i = 1; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->feedforward();
	}

    // (3) return last layer's output data
    int lastLayerIndex = layersConfig->_layers.size() - 1;
    return layersConfig->_layers[lastLayerIndex]->getOutputData();
}

template<typename Dtype>
void Network<Dtype>::backPropagateDQNNetwork(DQNImageLearner<Dtype> *learner) {
	LayersConfig<Dtype>* layersConfig = getLayersConfig();
	ALEInputLayer<Dtype>* inputLayer = (ALEInputLayer<Dtype>*)layersConfig->_layers[0];
    
    // (1) fill input layer's label
    inputLayer->fillLabel(learner);

    // (2) do back propagation
    _backpropagation(0);
}

template <typename Dtype>
void Network<Dtype>::_backpropagation(uint32_t batchIndex) {
	LayersConfig<Dtype>* layersConfig = getLayersConfig();

	for (int i = layersConfig->_layers.size()-1; i >= 0; i--) {
		Layer<Dtype>* hiddenLayer =
				dynamic_cast<Layer<Dtype>*>(layersConfig->_layers[i]);
		if (hiddenLayer) {
			//cout << layersConfig->_layers[i]->name << ": backpropagation ... " << endl;
			hiddenLayer->backpropagation();
			//cout << "input sumsq of " << hiddenLayer->name << ":\t\t" <<
			//		hiddenLayer->_inputData[0]->sumsq_device_grad() << endl;
		}

		/*
		else {
			cout << layersConfig->_layers[i]->name <<
					" is not a hiddenLayer, so skip backpropagation ... " << endl;
		}
		*/
	}
}

template<typename Dtype>
void Network<Dtype>::_backward(string layerName) {
	LayersConfig<Dtype>* layersConfig = getLayersConfig();
	typename map<string, Layer<Dtype>*>::iterator it = 
        layersConfig->_nameLayerMap.find(layerName);
	SASSERT((it != layersConfig->_nameLayerMap.end()), "invalid layer name. layer name=%s",
        layerName.c_str());

    Layer<Dtype>* hiddenLayer = dynamic_cast<Layer<Dtype>*>(it->second);
    SASSERT0(hiddenLayer != NULL); 
    hiddenLayer->backpropagation();
}

template <typename Dtype>
void Network<Dtype>::saveProposalTargets(ofstream& ofs) {
	LayersConfig<Dtype>* layersConfig = getLayersConfig();

	typename map<string, Layer<Dtype>*>::iterator it =
        layersConfig->_nameLayerMap.find("roi-data");
	assert(it != layersConfig->_nameLayerMap.end());

	Layer<Dtype>* proposalTargetLayer = it->second;
	const uint32_t numOutputs = proposalTargetLayer->_outputs.size();

	for (uint32_t i = 0; i < numOutputs; i++) {
		proposalTargetLayer->_outputData[i]->save(ofs);
	}


	//Data<Dtype>::printConfig = true;
	//proposalTargetLayer->_outputData[0]->print_data({}, false);
	//Data<Dtype>::printConfig = false;
}

template class Network<float>;
