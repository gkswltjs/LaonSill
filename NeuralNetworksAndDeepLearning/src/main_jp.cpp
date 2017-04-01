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
void vgg19();
void fasterRcnn();


void developerMain() {
	STDOUT_LOG("enter developerMain()");

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

	//artisticStyle();
	//vgg19();
	fasterRcnn();

	STDOUT_LOG("exit developerMain()");
}



void artisticStyle() {
	ArtisticStyle<float>* artisticStyle = new ArtisticStyle<float>();
	artisticStyle->transfer_style();
	delete artisticStyle;
}

void vgg19() {
	const int maxEpochs = 1000;
	const vector<string> lossLayers = {"loss"};
	const NetworkPhase phase = NetworkPhase::TrainPhase;

#if LOAD_WEIGHT
	vector<WeightsArg> weightsArgs(1);
	weightsArgs[0].weightsPath =
			"/home/jkim/Dev/SOOOA_HOME/network/VGG19.param";
#endif
	const uint32_t batchSize = 10;
	const uint32_t testInterval = 1;			// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 1;		// 1000000 / batchSize
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


void fasterRcnn() {
	const int maxEpochs = 1000;
	const vector<string> lossLayers = {"rpn_loss_cls", "rpn_loss_bbox", "loss_cls", "loss_bbox"};
	const NetworkPhase phase = NetworkPhase::TrainPhase;

	vector<WeightsArg> weightsArgs(1);
	weightsArgs[0].weightsPath = "/home/jkim/Dev/SOOOA_HOME/network/VGG_CNN_M_1024.param";

	const uint32_t batchSize = 1;
	const uint32_t testInterval = 20;			// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 200000;		// 1000000 / batchSize
	const float baseLearningRate = 0.001f;
	const uint32_t stepSize = 50000;
	const float weightDecay = 0.0005f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.1;
	const LRPolicy lrPolicy = LRPolicy::Step;

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
    struct timespec startTime;
    SPERF_START(SERVER_RUNNING_TIME, &startTime);
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
    SPERF_END(SERVER_RUNNING_TIME, startTime);
    STDOUT_LOG("server running time : %lf\n", SPERF_TIME(SERVER_RUNNING_TIME));
	STDOUT_BLOCK(cout << "SOOOA engine ends" << endl;);

    InitParam::destroy();
#else
    /*
    //const string source = "/home/jkim/Dev/git/caffe/examples/imagenet/ilsvrc12_train_lmdb";
    const string source = "/home/jkim/Dev/git/caffe/examples/test/train_lmdb";
    LMDBDataSet<float>* ds = new LMDBDataSet<float>(source);
    ds->load();

    int i = 0;
    while(true) {
    	cout << i << endl;
    	const float* trainData = ds->getTrainDataAt(i);
    	const float* trainLabel = ds->getTrainLabelAt(i);
    	i++;
    }
    delete ds;
    */

    auto env = lmdb::env::create();
    //env.set_mapsize(1UL * 1024UL * 1024UL * 1024UL);		// 1GB
    //env.open("/home/jkim/Dev/git/caffe/examples/test/train_lmdb_256_jpg");
    env.open("/home/jkim/Dev/git/caffe/examples/imagenet/ilsvrc12_train_lmdb");

    auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
    auto dbi = lmdb::dbi::open(rtxn, nullptr);
    auto cursor = lmdb::cursor::open(rtxn, dbi);

    string key, value;
    //lmdb::val key;
    size_t size = dbi.size(rtxn);
    vector<string> keys(size);
    int i = 0;
    while (cursor.get(key, value, MDB_FIRST)) {
    	//keys[i++] = key;

    	i++;
    	if (i % 10000 == 0) {
    		cout << i << endl;
    	}
    }


    cout << "size: " << size << ", i: " << i << endl;


    //cursor.get()
    /*
    cout << "key: " << key << ", size: " << value.size() << endl;
    key = "00000000_n01843383/n01843383_5825.JPEG";
    if (dbi.get(rtxn, key)) {
    	cout << "key: " << key;
    } else
    	cout << "failed to get ... " << endl;
    	*/







    /*
    unsigned char* ptr = (unsigned char*)(&value.c_str()[12]);

    Data<float>* data = new Data<float>("");
    data->reshape({1, 3, 224, 224});
    float* data_ptr = data->mutable_host_data();
    for (int i = 0; i < 224*224*3; i++) {
    	data_ptr[i] = float(ptr[i]) / 255;
    }
    data->transpose({0, 2, 3, 1});
    cv::Mat result(224, 224, CV_32FC3, data_ptr);
    cv::namedWindow("result");
    cv::imshow("result", result);
    cv::waitKey();
    */

#if 0
    const int numSample = 100;
    //const int numData = 150545;
    const int numData = 100;
    uint8_t data[numSample][numData];	//
    for (int i = 0; i < numSample; i++) {
    	for (int j = 0; j < numData; j++) {
    		data[i][j] = 0;
    	}
    }



    for (int i = 0; i < numSample; i++) {
    	if (!cursor.get(key, value, MDB_NEXT)) {
    		break;
    	}
    	const uint8_t* ptr = (uint8_t*)value.c_str();
    	const int size = value.size();

    	/*
    	for (int j = 0; j < value.size(); j++) {
    		data[i][j] = ptr[j];
    	}
    	*/


    	for (int j = 0; j < 50; j++) {
    		data[i][j] = ptr[j];
    	}
    	for (int j = 150530; j < size; j++) {
    		data[i][50+j-150530] = ptr[j];
    	}

    	/*
    	printf("%s(%d): %d, %d\n", key.c_str(), value.size(), ptr[size-4], value[size-3]);
    	uint8_t upper = ptr[size-4];
    	uint8_t lower = ptr[size-3];
    	uint16_t category = 0;
    	if (upper & 0x80) {
    		category = (0x7f & upper);
    		category = (category | (lower << 7));
    	} else {
    		category = lower;
    	}
    	printf("category: %d\n", category);
    	*/
    }

    for (int j = 0; j < 70; j++) {
    	for (int i = 0; i < numSample; i++) {
    		printf("%d,", data[i][j]);
    	}
    	printf("\n");
    }

    cursor.close();
    rtxn.abort();
#endif

#endif

	exit(EXIT_SUCCESS);
}


#endif














