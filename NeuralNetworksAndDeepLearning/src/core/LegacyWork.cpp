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
#include "StdOutLog.h"
#include "Debug.h"
#include "Evaluation.h"
#include "Top1Evaluation.h"
#include "Top5Evaluation.h"
#include "NetworkMonitor.h"

using namespace std;


#define FRCNN_TRAIN 0

template <typename Dtype>
void LegacyWork<Dtype>::buildNetwork(Job* job) {
    // XXX: 현재는 CCN Double Layer만 생성하도록 되어 있다. 수정필요!!!
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(job->getIntValue(0));
    

	// (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
#if FRCNN_TRAIN
	//LayersConfig<float>* layersConfig = createFrcnnTrainLayersConfig<float>();
	LayersConfig<float>* layersConfig = createFrcnnTrainOneShotLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createFrcnnStage1RpnTrainLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createFrcnnStage1TrainLayersConfig<float>();
#else
	LayersConfig<float>* layersConfig = createCNNSimpleLayersConfig<float>();
#endif

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

	//for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
	//	layersConfig->_layers[i]->reshape();
	//}
	//layersConfig->_inputLayer->shape(0, in_dim);

	// (4) network에 layersConfig 정보를 등록한다.
	network->setLayersConfig(layersConfig);
	network->loadPretrainedWeights();
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


#if FRCNN_TRAIN
	vector<WeightsArg> weightsArgs(1);
	//weightsArgs[0].weightsPath = 
    //  "/home/jkim/Dev/SOOOA_HOME/network/vgg_cnn_m_1024_full_10000of_30000.param";

	// frcnn 전체 네트워크 파라미터
	weightsArgs[0].weightsPath = "/home/jkim/Dev/SOOOA_HOME/network/network80000.param.bak";


	/*
	weightsArgs[0].weightsPath = "/home/jkim/Dev/SOOOA_HOME/network/frcnn_rpn_stage1.param";
	weightsArgs[0].weightsMap["conv1_filter"] = "conv1:rpn_filter";
	weightsArgs[0].weightsMap["conv1_bias"] = "conv1:rpn_bias";
	weightsArgs[0].weightsMap["conv2_filter"] = "conv2:rpn_filter";
	weightsArgs[0].weightsMap["conv2_bias"] = "conv2:rpn_bias";
	weightsArgs[0].weightsMap["conv3_filter"] = "conv3:rpn_filter";
	weightsArgs[0].weightsMap["conv3_bias"] = "conv3:rpn_bias";
	weightsArgs[0].weightsMap["conv4_filter"] = "conv4:rpn_filter";
	weightsArgs[0].weightsMap["conv4_bias"] = "conv4:rpn_bias";
	weightsArgs[0].weightsMap["conv5_filter"] = "conv5:rpn_filter";
	weightsArgs[0].weightsMap["conv5_bias"] = "conv5:rpn_bias";

	weightsArgs[1].weightsPath = 
        "/home/jkim/Dev/SOOOA_HOME/network/frcnn_detect_stage1_980000.param";
	weightsArgs[1].weightsMap["conv1_filter"] = "conv1:detect_filter";
	weightsArgs[1].weightsMap["conv1_bias"] = "conv1:detect_bias";
	weightsArgs[1].weightsMap["conv2_filter"] = "conv2:detect_filter";
	weightsArgs[1].weightsMap["conv2_bias"] = "conv2:detect_bias";
	weightsArgs[1].weightsMap["conv3_filter"] = "conv3:detect_filter";
	weightsArgs[1].weightsMap["conv3_bias"] = "conv3:detect_bias";
	weightsArgs[1].weightsMap["conv4_filter"] = "conv4:detect_filter";
	weightsArgs[1].weightsMap["conv4_bias"] = "conv4:detect_bias";
	weightsArgs[1].weightsMap["conv5_filter"] = "conv5:detect_filter";
	weightsArgs[1].weightsMap["conv5_bias"] = "conv5:detect_bias";
	*/

	const vector<string> lossLayers = { "softmaxWithLoss" };
	//const vector<string> lossLayers = {"rpn_loss_cls", "rpn_loss_bbox"};
	//const vector<string> lossLayers = {"loss_cls", "loss_bbox"};
	//const NetworkPhase phase = NetworkPhase::TestPhase;
	const NetworkPhase phase = NetworkPhase::TrainPhase;
#else
	vector<WeightsArg> weightsArgs(1);
	weightsArgs[0].weightsPath = "/home/jkim/Dev/SOOOA_HOME/network/network540000.param.bak";
	const vector<string> lossLayers = {};
	const NetworkPhase phase = NetworkPhase::TestPhase;
#endif
	const uint32_t batchSize = 100;
	const uint32_t testInterval = 500;		// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 10000000000;		// 1000000 / batchSize
	const float baseLearningRate = 0.1f;

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0001f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.1;
	//const LRPolicy lrPolicy = LRPolicy::Step;
	const LRPolicy lrPolicy = LRPolicy::Fixed;

	STDOUT_BLOCK(cout << "batchSize: " << batchSize << endl;);
	STDOUT_BLOCK(cout << "testInterval: " << testInterval << endl;);
	STDOUT_BLOCK(cout << "saveInterval: " << saveInterval << endl;);
	STDOUT_BLOCK(cout << "baseLearningRate: " << baseLearningRate << endl;);
	STDOUT_BLOCK(cout << "weightDecay: " << weightDecay << endl;);
	STDOUT_BLOCK(cout << "momentum: " << momentum << endl;);
	STDOUT_BLOCK(cout << "clipGradientsLevel: " << clipGradientsLevel << endl;);

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
			->networkPhase(phase)
			->gamma(gamma)
			->savePathPrefix(SPARAM(NETWORK_SAVE_DIR))
			->weightsArgs(weightsArgs)
#if FRCNN_TRAIN
			->networkListeners({
				new NetworkMonitor("rpn_loss_cls", NetworkMonitor::PLOT_ONLY),
				new NetworkMonitor("rpn_loss_bbox", NetworkMonitor::PLOT_ONLY),
				new NetworkMonitor("loss_cls", NetworkMonitor::PLOT_ONLY),
				new NetworkMonitor("loss_bbox", NetworkMonitor::PLOT_ONLY)
				/*
				new NetworkMonitor("rpn_loss_cls", NetworkMonitor::PLOT_AND_WRITE),
				new NetworkMonitor("rpn_loss_bbox", NetworkMonitor::PLOT_AND_WRITE),
				new NetworkMonitor("loss_cls", NetworkMonitor::PLOT_AND_WRITE),
				new NetworkMonitor("loss_bbox", NetworkMonitor::PLOT_AND_WRITE)
				*/
				})

#else
			->networkListeners({})
#endif
			->lossLayers(lossLayers)
			->build();

	Util::printVramInfo();

    // 네트워크를 등록한다.
	Network<Dtype>* network = new Network<Dtype>(networkConfig);

    return network->getNetworkID();
}

template class LegacyWork<float>;
