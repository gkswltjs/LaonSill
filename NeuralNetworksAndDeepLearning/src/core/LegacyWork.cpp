/**
 * @file LegacyWork.cpp
 * @date 2016-12-23
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <map>
#include <string>

#include "common.h"
#include "LegacyWork.h"
#include "Network.h"
#include "NetworkConfig.h"
#include "ColdLog.h"
#include "Debug.h"
#include "Evaluation.h"
#include "Top1Evaluation.h"
#include "Top5Evaluation.h"
#include "NetworkMonitor.h"

using namespace std;

template <typename Dtype>
void LegacyWork<Dtype>::buildNetwork(Job* job) {
    // XXX: 현재는 CCN Double Layer만 생성하도록 되어 있다. 수정필요!!!
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(job->getIntValue(0));
    
    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<float>* layersConfig = createCNNSimpleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception5BLayersConfig<float>();

    // (2) network config 정보를 layer들에게 전달한다.
    for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
        layersConfig->_layers[i]->setNetworkConfig(network->config);
    }

    // (3) shape 과정을 수행한다. 
    //io_dim in_dim;
    //in_dim.rows = network->config->_dataSet->getRows();
    //in_dim.cols = network->config->_dataSet->getCols();
    //in_dim.channels = network->config->_dataSet->getChannels();
    //in_dim.batches = network->config->_batchSize;
    //layersConfig->_inputLayer->setInDimension(in_dim);

    for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
    	layersConfig->_layers[i]->shape();
    	//in_dim = layersConfig->_layers[i-1]->getOutDimension();
    }
    //layersConfig->_inputLayer->shape(0, in_dim);

    // (4) network에 layersConfig 정보를 등록한다.
    network->setLayersConfig(layersConfig);
}

template <typename Dtype>
void LegacyWork<Dtype>::trainNetwork(Job *job) {
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(job->getIntValue(0));
    int maxEpochs = job->getIntValue(1);

    COLD_LOG(ColdLog::INFO, true, "training network starts(maxEpoch: %d).", maxEpochs);

    network->sgd_with_timer(maxEpochs);

    // XXX: save() 함수 확인 다시하자.
    //if (consumerIdx == 0)
    //    network->save();
}

template <typename Dtype>
void LegacyWork<Dtype>::cleanupNetwork(Job* job) {
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(job->getIntValue(0));

	LayersConfig<Dtype>* layersConfig = network->getLayersConfig();
	const uint32_t layerSize = layersConfig->_layers.size();
	for(uint32_t i = 0; i < layerSize; i++) {
		delete layersConfig->_layers[i];
	}

	// clean up layer data
	typename map<string, Data<Dtype>*>::iterator it;
	for (it = layersConfig->_layerDataMap.begin();
			it != layersConfig->_layerDataMap.end(); it++) {
		delete it->second;
	}
}

template <typename Dtype>
int LegacyWork<Dtype>::createNetwork() {
    // XXX: network를 어떻게 구성할지에 대한 정보를 받아야 한다.
    //      또한, 그 정보를 토대로 네트워크를 구성해야 한다.
    //      Evaluation과 Dataset, Network Listener도 분리 시켜야 한다.
    const uint32_t batchSize = 50;
	//const uint32_t batchSize = 1000;
	//const uint32_t testInterval = 20;			// 10000(목표 샘플수) / batchSize
	const uint32_t testInterval = 200;			// 10000(목표 샘플수) / batchSize
	//const uint32_t saveInterval = 20000;		// 1000000 / batchSize
	const uint32_t saveInterval = 1000000;		// 1000000 / batchSize
	const uint32_t stepSize = 100000;
	const float baseLearningRate = 0.001f;
	const float weightDecay = 0.0002f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.1;
	const LRPolicy lrPolicy = LRPolicy::Step;

	cout << "batchSize: " << batchSize << endl;
	cout << "testInterval: " << testInterval << endl;
	cout << "saveInterval: " << saveInterval << endl;
	cout << "baseLearningRate: " << baseLearningRate << endl;
	cout << "weightDecay: " << weightDecay << endl;
	cout << "momentum: " << momentum << endl;
	cout << "clipGradientsLevel: " << clipGradientsLevel << endl;

	//DataSet<Dtype>* dataSet = createMnistDataSet<Dtype>();
	//DataSet<Dtype>* dataSet = createMockDataSet<Dtype>();
	//DataSet<Dtype>* dataSet = createImageNet10000DataSet<Dtype>();
	//dataSet->load();

	Evaluation<Dtype>* top1Evaluation = new Top1Evaluation<Dtype>();
	Evaluation<Dtype>* top5Evaluation = new Top5Evaluation<Dtype>();
	NetworkListener* networkListener = new NetworkMonitor(NetworkMonitor::PLOT_ONLY);

	NetworkConfig<Dtype>* networkConfig =
			(new typename NetworkConfig<Dtype>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->testInterval(testInterval)
			->saveInterval(saveInterval)
			->stepSize(stepSize)
			->clipGradientsLevel(clipGradientsLevel)
			->lrPolicy(lrPolicy)
			->gamma(gamma)
			//->dataSet(dataSet)
			->evaluations({top1Evaluation, top5Evaluation})
			->savePathPrefix(SPARAM(NETWORK_SAVE_DIR))
			->networkListeners({networkListener})
			->build();

	Util::printVramInfo();


    // 네트워크를 등록한다.
    // TODO: 현재는 증가하는 방식으로만 등록을 시키고 있다. 
    //      pool 형태로 돌려쓸 수 있도록 수정이 필요할지 고민해보자.
    // XXX: make network generate its network ID by itself when it is created
	Network<Dtype>* network = new Network<Dtype>(networkConfig);

    return network->getNetworkID();
}

template class LegacyWork<float>;
