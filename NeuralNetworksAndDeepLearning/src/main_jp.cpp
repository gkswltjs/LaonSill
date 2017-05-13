#if 1

#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>

#include "cuda/Cuda.h"

#include "gnuplot-iostream.h"

#include "jsoncpp/json/json.h"

#include "common.h"
#include "DataSet.h"
#include "MockDataSet.h"
#include "Debug.h"
#include "Evaluation.h"
#include "Top1Evaluation.h"
#include "Top5Evaluation.h"
#include "NetworkMonitor.h"
#include "Network.h"
#include "NetworkConfig.h"
#include "Util.h"
#include "Worker.h"
#include "Job.h"
#include "Communicator.h"
#include "Client.h"
#include "InitParam.h"
#include "Param.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "HotLog.h"
#include "StdOutLog.h"
#include "Perf.h"
#include "Atari.h"
#include "Broker.h"
#include "test.h"
#include "DQNImageLearner.h"
#include "ImageUtil.h"
#include "ArtisticStyle.h"

using namespace std;


void artisticStyle();
void vgg16();
void vgg19();
void fasterRcnnTrain();
void fasterRcnnTest();
void ssd();


void developerMain() {
	STDOUT_LOG("enter developerMain()");

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

	//artisticStyle();
	//vgg16();
	//vgg19();
	//fasterRcnnTrain();
	//fasterRcnnTest();
	ssd();

	STDOUT_LOG("exit developerMain()");
}



void artisticStyle() {
	ArtisticStyle<float>* artisticStyle = new ArtisticStyle<float>();
	artisticStyle->transfer_style();
	delete artisticStyle;
}

#define LOAD_WEIGHT 1

void vgg16() {
	const int maxEpochs = 1000;
	const vector<string> lossLayers = {"loss"};
	//const vector<string> accuracyLayers = {"accuracy/top1", "accuracy/top5"};
	const vector<string> accuracyLayers = {};
	const NetworkPhase phase = NetworkPhase::TrainPhase;

#if LOAD_WEIGHT
	vector<WeightsArg> weightsArgs(1);
	weightsArgs[0].weightsPath =
			"/home/jkim/Dev/SOOOA_HOME/network/VGG19.param";
#endif
	const uint32_t batchSize = 16;
	const uint32_t testInterval = 1;			// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 1000000;		// 1000000 / batchSize
	const float baseLearningRate = 0.001f;

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0005f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.0001;
	//const LRPolicy lrPolicy = LRPolicy::Step;
	const LRPolicy lrPolicy = LRPolicy::Fixed;

	STDOUT_BLOCK(cout << "batchSize: " << batchSize << endl;);
	STDOUT_BLOCK(cout << "testInterval: " << testInterval << endl;);
	STDOUT_BLOCK(cout << "saveInterval: " << saveInterval << endl;);
	STDOUT_BLOCK(cout << "baseLearningRate: " << baseLearningRate << endl;);
	STDOUT_BLOCK(cout << "weightDecay: " << weightDecay << endl;);
	STDOUT_BLOCK(cout << "momentum: " << momentum << endl;);
	STDOUT_BLOCK(cout << "clipGradientsLevel: " << clipGradientsLevel << endl;);

	NetworkConfig<float>* networkConfig =
			(new typename NetworkConfig<float>::Builder())
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
#if LOAD_WEIGHT
			->weightsArgs(weightsArgs)
#endif
			->networkListeners({
				new NetworkMonitor("loss", NetworkMonitor::PLOT_ONLY)
				})
			->lossLayers(lossLayers)
			->accuracyLayers(accuracyLayers)
			->build();

	Util::printVramInfo();

	Network<float>* network = new Network<float>(networkConfig);
	LayersConfig<float>* layersConfig = createVGG16NetLayersConfig<float>();

	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
#if LOAD_WEIGHT
	network->loadPretrainedWeights();
#endif
	network->sgd_with_timer(maxEpochs);
}

void vgg19() {
	const int maxEpochs = 1000;
	const vector<string> lossLayers = {"loss"};
	const vector<string> accuracyLayers = {"accuracy/top1", "accuracy/top5"};
	const NetworkPhase phase = NetworkPhase::TrainPhase;

#if LOAD_WEIGHT
	vector<WeightsArg> weightsArgs(1);
	weightsArgs[0].weightsPath =
			"/home/jkim/Dev/SOOOA_HOME/network/VGG19_LMDB_0.01.param";
#endif
	const uint32_t batchSize = 20;
	const uint32_t testInterval = 50;			// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 100000;		// 1000000 / batchSize
	const float baseLearningRate = 0.00001f;

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0005f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.0001;
	//const LRPolicy lrPolicy = LRPolicy::Step;
	const LRPolicy lrPolicy = LRPolicy::Fixed;

	STDOUT_BLOCK(cout << "batchSize: " << batchSize << endl;);
	STDOUT_BLOCK(cout << "testInterval: " << testInterval << endl;);
	STDOUT_BLOCK(cout << "saveInterval: " << saveInterval << endl;);
	STDOUT_BLOCK(cout << "baseLearningRate: " << baseLearningRate << endl;);
	STDOUT_BLOCK(cout << "weightDecay: " << weightDecay << endl;);
	STDOUT_BLOCK(cout << "momentum: " << momentum << endl;);
	STDOUT_BLOCK(cout << "clipGradientsLevel: " << clipGradientsLevel << endl;);

	NetworkConfig<float>* networkConfig =
			(new typename NetworkConfig<float>::Builder())
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
#if LOAD_WEIGHT
			->weightsArgs(weightsArgs)
#endif
			->networkListeners({
				new NetworkMonitor("loss", NetworkMonitor::PLOT_ONLY)
				})
			->lossLayers(lossLayers)
			->accuracyLayers(accuracyLayers)
			->build();

	Util::printVramInfo();

	Network<float>* network = new Network<float>(networkConfig);
	LayersConfig<float>* layersConfig = createVGG19NetLayersConfig<float>();

	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
	network->loadPretrainedWeights();

	network->sgd_with_timer(maxEpochs);
}


void fasterRcnnTrain() {
	srand((uint32_t)time(NULL));

	const int maxEpochs = 1000;
	const vector<string> lossLayers = {"rpn_loss_cls", "rpn_loss_bbox", "loss_cls", "loss_bbox"};
	const NetworkPhase phase = NetworkPhase::TrainPhase;
	const string networkSaveDir = SPARAM(NETWORK_SAVE_DIR);

	vector<WeightsArg> weightsArgs(1);
	//weightsArgs[0].weightsPath = networkSaveDir + "/VGG_CNN_M_1024.param";
	weightsArgs[0].weightsPath = networkSaveDir + "/network600000.param";

	const uint32_t batchSize = 1;
	const float baseLearningRate = 0.001f;
	const LRPolicy lrPolicy = LRPolicy::Step;
	const float gamma = 0.1;
	const uint32_t stepSize = 50000;
	const uint32_t testInterval = 1;			// 10000(목표 샘플수) / batchSize
	const float momentum = 0.9f;
	const float weightDecay = 0.0005f;

	const uint32_t saveInterval = 10000;		// 1000000 / batchSize
	const float clipGradientsLevel = 0.0f;

	STDOUT_BLOCK(cout << "batchSize: " << batchSize << endl;);
	STDOUT_BLOCK(cout << "testInterval: " << testInterval << endl;);
	STDOUT_BLOCK(cout << "saveInterval: " << saveInterval << endl;);
	STDOUT_BLOCK(cout << "baseLearningRate: " << baseLearningRate << endl;);
	STDOUT_BLOCK(cout << "weightDecay: " << weightDecay << endl;);
	STDOUT_BLOCK(cout << "momentum: " << momentum << endl;);
	STDOUT_BLOCK(cout << "clipGradientsLevel: " << clipGradientsLevel << endl;);

	NetworkConfig<float>* networkConfig =
			(new typename NetworkConfig<float>::Builder())
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
			->networkListeners({
				new NetworkMonitor("rpn_loss_cls", NetworkMonitor::PLOT_ONLY),
				new NetworkMonitor("rpn_loss_bbox", NetworkMonitor::PLOT_ONLY),
				new NetworkMonitor("loss_cls", NetworkMonitor::PLOT_ONLY),
				new NetworkMonitor("loss_bbox", NetworkMonitor::PLOT_ONLY)
				})
			->lossLayers(lossLayers)
			->build();

	Network<float>* network = new Network<float>(networkConfig);
	LayersConfig<float>* layersConfig = createFrcnnTrainOneShotLayersConfig<float>();

	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
	network->loadPretrainedWeights();

	/*
	const string targetLayer = "conv2";
	for (int i = 0; i < layersConfig->_learnableLayers.size(); i++) {
		LearnableLayer<float>* layer = layersConfig->_learnableLayers[i];
		if (layer->name == targetLayer) {

			Data<float>::printConfig = true;
			SyncMem<float>::printConfig = true;

			layer->_params[0]->print_data({}, false);

			Data<float>::printConfig = false;
			SyncMem<float>::printConfig = false;
		}
	}
	*/
	network->sgd_with_timer(maxEpochs);
}

void fasterRcnnTest() {
	srand((uint32_t)time(NULL));

	const int maxEpochs = 1000;
	const NetworkPhase phase = NetworkPhase::TestPhase;
	const string networkSaveDir = SPARAM(NETWORK_SAVE_DIR);

	vector<WeightsArg> weightsArgs(1);
	//weightsArgs[0].weightsPath = networkSaveDir + "/VGG_CNN_M_1024_FRCNN_CAFFE.param";
	weightsArgs[0].weightsPath = networkSaveDir + "/SOOOA_FRCNN_600000.param";
	cout << "weight path: " << weightsArgs[0].weightsPath << endl;

	NetworkConfig<float>* networkConfig =
			(new typename NetworkConfig<float>::Builder())
			->networkPhase(phase)
			->weightsArgs(weightsArgs)
			->build();

	Network<float>* network = new Network<float>(networkConfig);
	LayersConfig<float>* layersConfig = createFrcnnTestOneShotLayersConfig<float>();

	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
	network->loadPretrainedWeights();

	RoITestInputLayer<float>* inputLayer = dynamic_cast<RoITestInputLayer<float>*>(layersConfig->_layers[0]);



	//struct timespec startTime;
	//SPERF_START(SERVER_RUNNING_TIME, &startTime);

	const int imageSize = inputLayer->imdb->imageIndex.size();
	while (inputLayer->cur < imageSize) {
		network->_feedforward(0);
	}

	//SPERF_END(SERVER_RUNNING_TIME, startTime);
	//float time = SPERF_TIME(SERVER_RUNNING_TIME);
	//STDOUT_LOG("server running time : %lf for %d images (%lf fps)\n",
	//		time, imageSize, imageSize / time);
}








void ssd() {
	const int maxEpochs = 1000;
	const vector<string> lossLayers = {"mbox_loss"};
	const vector<string> accuracyLayers = {};
	const NetworkPhase phase = NetworkPhase::TrainPhase;

#if LOAD_WEIGHT
	vector<WeightsArg> weightsArgs(1);
	weightsArgs[0].weightsPath =
			"/home/jkim/Dev/SOOOA_HOME/network/SSD_PRETRAINED.param";
#endif

	const uint32_t batchSize = 16;				// 32
	const uint32_t testInterval = 100;			// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 1000000;		// 1000000 / batchSize
	const float baseLearningRate = 0.001f;

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0005f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.1;
	const LRPolicy lrPolicy = LRPolicy::Fixed;

	STDOUT_BLOCK(cout << "batchSize: " << batchSize << endl;);
	STDOUT_BLOCK(cout << "testInterval: " << testInterval << endl;);
	STDOUT_BLOCK(cout << "saveInterval: " << saveInterval << endl;);
	STDOUT_BLOCK(cout << "baseLearningRate: " << baseLearningRate << endl;);
	STDOUT_BLOCK(cout << "weightDecay: " << weightDecay << endl;);
	STDOUT_BLOCK(cout << "momentum: " << momentum << endl;);
	STDOUT_BLOCK(cout << "clipGradientsLevel: " << clipGradientsLevel << endl;);

	NetworkConfig<float>* networkConfig =
			(new typename NetworkConfig<float>::Builder())
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
#if LOAD_WEIGHT
			->weightsArgs(weightsArgs)
#endif
			->networkListeners({
				new NetworkMonitor("mbox_loss", NetworkMonitor::PLOT_ONLY)
				})
			->lossLayers(lossLayers)
			->accuracyLayers(accuracyLayers)
			->build();

	Util::printVramInfo();

	Network<float>* network = new Network<float>(networkConfig);
	LayersConfig<float>* layersConfig = createSSDNetLayersConfig<float>();

	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
#if LOAD_WEIGHT
	network->loadPretrainedWeights();
#endif


	/*
	Data<float>::printConfig = true;
	SyncMem<float>::printConfig = true;
	for (int i = 0; i < layersConfig->_learnableLayers.size(); i++) {
		for (int j = 0; j < layersConfig->_learnableLayers[i]->_params.size(); j++) {
			layersConfig->_learnableLayers[i]->_params[j]->print_data({}, false);
		}
	}
	Data<float>::printConfig = false;
	SyncMem<float>::printConfig = false;
	*/
	network->sgd_with_timer(maxEpochs);
}





#define TEST_MODE 0

#if TEST_MODE
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lmdb++.h"
#include "MathFunctions.h"
#include "LMDBDataSet.h"
#endif

int main(int argc, char** argv) {
#if !TEST_MODE
    int     opt;

    // 처음 생각했던 것보다 실행모드의 개수가 늘었다.
    // 모드가 하나만 더 추가되면 그냥 enum type으로 모드를 정의하도록 하자.
    bool    useDeveloperMode = true;
    char*   singleJobFilePath;
    char*   romFilePath;
    char*   testItemName;

    // (2) 서버 시작 시간 측정을 시작한다.
    //struct timespec startTime;
    //SPERF_START(SERVER_RUNNING_TIME, &startTime);
	STDOUT_BLOCK(cout << "SOOOA engine starts" << endl;);

    // (3) 파라미터, 로깅, job 모듈을 초기화 한다.
    InitParam::init();
    Perf::init();
    SysLog::init();
    ColdLog::init();
    Job::init();
    Broker::init();
    Network<float>::init();
    DQNImageLearner<float>::init();

    SYS_LOG("Logging system is initialized...");

    // (4) 뉴럴 네트워크 관련 기본 설정을 한다.
    //     TODO: SPARAM의 인자로 대체하자.
	cout.precision(7);
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	Util::setPrint(false);

	// (5-A-1) Cuda를 생성한다.
	Cuda::create(SPARAM(GPU_COUNT));
	COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

	// (5-A-2) DeveloperMain()함수를 호출한다.
	developerMain();

	// (5-A-3) 자원을 해제 한다.
	Cuda::destroy();

    ColdLog::destroy();
    SysLog::destroy();
    Broker::destroy();

    // (7) 서버 종료 시간을 측정하고, 계산하여 서버 실행 시간을 출력한다.
    //SPERF_END(SERVER_RUNNING_TIME, startTime);
    //STDOUT_LOG("server running time : %lf\n", SPERF_TIME(SERVER_RUNNING_TIME));
	STDOUT_BLOCK(cout << "SOOOA engine ends" << endl;);

    InitParam::destroy();
#else
    vector<int> vec(1, 1);
    vec.push_back(4);

    for (int i = 0; i < vec.size(); i++) {
    	cout << vec[i] << endl;
    }

#endif

	exit(EXIT_SUCCESS);
}


#endif














