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

using namespace std;

#ifndef CLIENT_MODE

void printUsageAndExit(char* prog) {
    fprintf(stderr,
        "Usage: %s [-v] [-d | -f jobFilePath | -a romFilePath | -t testItemName]\n", prog);
    exit(EXIT_FAILURE);
}

void drawAvgOfSquaredSumGrad(Gnuplot &plot, vector<pair<int, double>> &plotData,
    LayersConfig<float>* lc, string layerName) {
    // calc squared sum
    Layer<float>* layer = (Layer<float>*)lc->_nameLayerMap[layerName];
    const float* data = layer->_outputData[0]->host_grad(); 
    int nout = layer->_outputData[0]->getCount();
    float sum = 0.0;
    for (int i = 0; i < nout; i++) {
        sum += data[i] * data[i];
    }

    if (plotData.size() > 100) {
        plotData.clear();
    }

    sum /= (float)nout;
    plotData.push_back(make_pair(plotData.size(), sum));

    char cmd[1024];
    sprintf(cmd, "plot '-' using 1:2 with lines title '%s'\n", layerName.c_str());

    plot << cmd;
    plot.send1d(plotData);
}

void drawAvgOfSquaredSumData(Gnuplot &plot, vector<pair<int, double>> &plotData,
    LayersConfig<float>* lc, string layerName) {
    // calc squared sum
    Layer<float>* layer = (Layer<float>*)lc->_nameLayerMap[layerName];
    const float* data = layer->_outputData[0]->host_data(); 
    int nout = layer->_outputData[0]->getCount();
    float sum = 0.0;
    for (int i = 0; i < nout; i++) {
        sum += data[i] * data[i];
    }

    if (plotData.size() > 100) {
        plotData.clear();
    }

    sum /= (float)nout;
    plotData.push_back(make_pair(plotData.size(), sum));

    char cmd[1024];
    sprintf(cmd, "plot '-' using 1:2 with lines title '%s'\n", layerName.c_str());

    plot << cmd;
    plot.send1d(plotData);
}

void printDataForDebug(LayersConfig<float>* lc, const char* title) {
    int layerCount = lc->_layers.size();

    for (int i = 0; i < layerCount; i++) {
        printf("[Layer : %s #%d]\n", title, i);
        const float* data = lc->_layers[i]->_outputData[0]->host_data();
        const float* grad = lc->_layers[i]->_outputData[0]->host_grad();

        for (int j = 0; j < 8; j++) {
            printf(" %f", data[j]);
        }
        printf("\n");

        for (int j = 0; j < 8; j++) {
            printf(" %f", grad[j]);
        }
        printf("\n");
    }
}

// FIXME: 디버깅용 함수.... 나중에 싹다 지우자.
void printWeightAndBias(LayersConfig<float>* lc, const char* title) {
    int layerCount = lc->_layers.size();

    for (int i = 0; i < layerCount; i++) {
        printf("[Layer : %s #%d]\n", title, i);

        FullyConnectedLayer<float>* fcLayer = 
            dynamic_cast<FullyConnectedLayer<float>*>(lc->_layers[i]);
        if (fcLayer) {
            const float* weightParams = fcLayer->_params[0]->host_data();
            const float* biasParams = fcLayer->_params[1]->host_data();
            const float* weightGradParams = fcLayer->_params[0]->host_grad();
            const float* biasGradParams = fcLayer->_params[1]->host_grad();

            int weightCnt = fcLayer->_params[0]->getCount();
            int biasCnt = fcLayer->_params[1]->getCount();

            printf(" - Weight Data : ");
            for (int j = 0; j < min(3, weightCnt) ; j++) {
                printf(" %f", weightParams[j]);
            }
            printf(" ~ ");
            for (int j = max(0, weightCnt - 3); j < weightCnt ; j++) {
                printf(" %f", weightParams[j]);
            }
            printf("\n");
            printf(" - Weight Grad : ");
            for (int j = 0; j < min(3, weightCnt) ; j++) {
                printf(" %f", weightGradParams[j]);
            }
            printf(" ~ ");
            for (int j = max(0, weightCnt - 3); j < weightCnt ; j++) {
                printf(" %f", weightGradParams[j]);
            }
            printf("\n");
            printf(" - Bias Data : ");
            for (int j = 0; j < min(3, biasCnt); j++) {
                printf(" %f", biasParams[j]);
            }
            printf(" ~ ");
            for (int j = max(0, biasCnt - 3); j < biasCnt; j++) {
                printf(" %f", biasParams[j]);
            }
            printf("\n");
            printf(" - Bias Grad : ");
            for (int j = 0; j < min(3, biasCnt); j++) {
                printf(" %f", biasGradParams[j]);
            }
            printf(" ~ ");
            for (int j = max(0, biasCnt - 3); j < biasCnt; j++) {
                printf(" %f", biasGradParams[j]);
            }
            printf("\n");
        }

        ConvLayer<float>* convLayer = dynamic_cast<ConvLayer<float>*>(lc->_layers[i]);
        if (convLayer) {
            const float* filterParams = convLayer->_params[0]->host_data();
            const float* biasParams = convLayer->_params[1]->host_data();
            const float* filterGradParams = convLayer->_params[0]->host_grad();
            const float* biasGradParams = convLayer->_params[1]->host_grad();
            
            int filterCnt = convLayer->_params[0]->getCount();
            int biasCnt = convLayer->_params[1]->getCount();

            printf(" - Filter Data : ");
            for (int j = 0; j < min(3, filterCnt) ; j++) {
                printf(" %f", filterParams[j]);
            }
            printf(" ~ ");
            for (int j = max(0, filterCnt - 3); j < filterCnt ; j++) {
                printf(" %f", filterParams[j]);
            }
            printf("\n");
            printf(" - Filter Grad : ");
            for (int j = 0; j < min(3, filterCnt) ; j++) {
                printf(" %f", filterGradParams[j]);
            }
            printf(" ~ ");
            for (int j = max(0, filterCnt - 3); j < filterCnt ; j++) {
                printf(" %f", filterGradParams[j]);
            }
            printf("\n");
            printf(" - Bias Data : ");
            for (int j = 0; j < min(3, biasCnt); j++) {
                printf(" %f", biasParams[j]);
            }
            printf(" ~ ");
            for (int j = max(0, biasCnt - 3); j < biasCnt; j++) {
                printf(" %f", biasParams[j]);
            }
            printf("\n");
            printf(" - Bias Grad : ");
            for (int j = 0; j < min(3, biasCnt); j++) {
                printf(" %f", biasGradParams[j]);
            }
            printf(" ~ ");
            for (int j = max(0, biasCnt - 3); j < biasCnt; j++) {
                printf(" %f", biasGradParams[j]);
            }
            printf("\n");
        }

        {
            Layer<float>* layer = lc->_layers[i];

            const float* inputData = lc->_layers[i]->_inputData[0]->host_data();
            const float* inputGrad = lc->_layers[i]->_inputData[0]->host_grad();
            int inputDataCnt = lc->_layers[i]->_inputData[0]->getCount();
            const float* outputData = lc->_layers[i]->_outputData[0]->host_data();
            const float* outputGrad = lc->_layers[i]->_outputData[0]->host_grad();
            int outputDataCnt = lc->_layers[i]->_outputData[0]->getCount();

            printf(" - Input Data : ");
            for (int j = 0; j < min(3, inputDataCnt); j++) {
                printf(" %f", inputData[j]);
            }
            printf(" ~ ");
            for (int j = max(0, inputDataCnt - 3); j < inputDataCnt; j++) {
                printf(" %f", inputData[j]);
            }
            printf("\n");
            printf(" - Input Grad : ");
            for (int j = 0; j < min(3, inputDataCnt); j++) {
                printf(" %f", inputGrad[j]);
            }
            printf(" ~ ");
            for (int j = max(0, inputDataCnt - 3); j < inputDataCnt; j++) {
                printf(" %f", inputGrad[j]);
            }
            printf("\n");
            printf(" - Output Data : ");
            for (int j = 0; j < min(3, outputDataCnt); j++) {
                printf(" %f", outputData[j]);
            }
            printf(" ~ ");
            for (int j = max(0, outputDataCnt - 3); j < outputDataCnt; j++) {
                printf(" %f", outputData[j]);
            }
            printf("\n");
            printf(" - Output Grad : ");
            for (int j = 0; j < min(3, outputDataCnt); j++) {
                printf(" %f", outputGrad[j]);
            }
            printf(" ~ ");
            for (int j = max(0, outputDataCnt - 3); j < outputDataCnt; j++) {
                printf(" %f", outputGrad[j]);
            }
            printf("\n");
        }
    }
}

void developerMain() {
    STDOUT_LOG("enter developerMain()");

    checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    // loss layer of Discriminator GAN 
	const vector<string> llDGAN = { "celossDGAN" };
    // loss layer of Generatoer-Discriminator 0 GAN
	const vector<string> llGD0GAN = { "celossGD0GAN" };

	const NetworkPhase phase = NetworkPhase::TrainPhase;
	const uint32_t batchSize = 64;
	const uint32_t testInterval = 1;		// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 100000;		// 1000000 / batchSize
	const float baseLearningRate = 0.0002f;

	const uint32_t stepSize = 100000;
	const float weightDecay = 0.0001f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.1;
	const LRPolicy lrPolicy = LRPolicy::Fixed;

    const Optimizer opt = Optimizer::Adam;

	STDOUT_BLOCK(cout << "batchSize: " << batchSize << endl;);
	STDOUT_BLOCK(cout << "testInterval: " << testInterval << endl;);
	STDOUT_BLOCK(cout << "saveInterval: " << saveInterval << endl;);
	STDOUT_BLOCK(cout << "baseLearningRate: " << baseLearningRate << endl;);
	STDOUT_BLOCK(cout << "weightDecay: " << weightDecay << endl;);
	STDOUT_BLOCK(cout << "momentum: " << momentum << endl;);
	STDOUT_BLOCK(cout << "clipGradientsLevel: " << clipGradientsLevel << endl;);

	NetworkConfig<float>* ncDGAN =
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
			->networkListeners({
				new NetworkMonitor("celossDGAN", NetworkMonitor::PLOT_ONLY),
				})
			->lossLayers(llDGAN)
            ->optimizer(opt)
            ->beta(0.5, 0.999)
			->build();

	NetworkConfig<float>* ncGD0GAN =
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
			->networkListeners({
				new NetworkMonitor("celossGD0GAN", NetworkMonitor::PLOT_ONLY),
				})
			->lossLayers(llGD0GAN)
            ->optimizer(opt)
            ->beta(0.5, 0.999)
			->build();

	Util::printVramInfo();

 	Network<float>* networkDGAN = new Network<float>(ncDGAN);
 	Network<float>* networkGD0GAN = new Network<float>(ncGD0GAN);

    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<float>* lcDGAN = createDOfGANLayersConfig<float>();
	LayersConfig<float>* lcGD0GAN = createGD0OfGANLayersConfig<float>();
 	networkDGAN->setLayersConfig(lcDGAN);
 	networkGD0GAN->setLayersConfig(lcGD0GAN);

	// (2) network config 정보를 layer들에게 전달한다.
	for (uint32_t i = 0; i < lcDGAN->_layers.size(); i++)
		lcDGAN->_layers[i]->setNetworkConfig(ncDGAN);

	for (uint32_t i = 0; i < lcDGAN->_layers.size(); i++)
		lcDGAN->_layers[i]->reshape();

	for (uint32_t i = 0; i < lcGD0GAN->_layers.size(); i++)
		lcGD0GAN->_layers[i]->setNetworkConfig(ncGD0GAN);

	for (uint32_t i = 0; i < lcGD0GAN->_layers.size(); i++)
		lcGD0GAN->_layers[i]->reshape();

    Gnuplot gpGDGanDeconv1;
    vector<pair<int, double>> dataGDGanDeconv1;
    Gnuplot gpGDGanDeconv2;
    Gnuplot gpGDGanDeconv3;
    Gnuplot gpGDGanDeconv4;

    Gnuplot gpGDGanConv1;
    vector<pair<int, double>> dataGDGanConv1;

    Gnuplot gpGDGanConv2;
    Gnuplot gpGDGanConv3;
    Gnuplot gpGDGanConv4;

    Gnuplot gpDGanConv1;
    vector<pair<int, double>> dataDGanConv1;
    Gnuplot gpDGanConv2;
    Gnuplot gpDGanConv3;
    Gnuplot gpDGanConv4;

    int debugPeriod = 100;

    for (int i = 0; i < 25; i++) {  // epoch
        InputLayer<float>* inputLayer = lcDGAN->_inputLayer;
        const uint32_t trainDataSize = inputLayer->getNumTrainData();
        const uint32_t numBatches = trainDataSize / ncDGAN->_batchSize;

        for (int j = 0; j < numBatches; j++) {
            float lossD = 0.0;
            float lossG = 0.0;

            if (j % debugPeriod == 0)
                printWeightAndBias(lcDGAN, "D init");
            lossD = networkDGAN->sgdMiniBatch(j);
            if (j % debugPeriod == 0)
                printWeightAndBias(lcDGAN, "real D");

            if (j % debugPeriod == 0)
                printWeightAndBias(lcGD0GAN, "fake D init");
            lossD += networkGD0GAN->sgd(1);
            if (j % debugPeriod == 0)
                printWeightAndBias(lcGD0GAN, "fake D");
#if 0
            drawAvgOfSquaredSumData(gpGDGanDeconv1, dataGDGanDeconv1, lcGD0GAN, 
                "DeconvLayer1");
            drawAvgOfSquaredSumData(gpGDGanConv1, dataGDGanConv1, lcGD0GAN, "ConvLayer1");
            drawAvgOfSquaredSumData(gpDGanConv1, dataDGanConv1, lcDGAN, "ConvLayer1");
#endif
            //printDataForDebug(lcDGAN, "D-GAN"); 
            //printDataForDebug(lcGD0GAN, "GD0-GAN");

            CrossEntropyWithLossLayer<float>* lossLayer =
                dynamic_cast<CrossEntropyWithLossLayer<float>*>(lcGD0GAN->_lastLayers[0]);
            SASSERT0(lossLayer != NULL);
            NoiseInputLayer<float>* noiseInputLayer =
                dynamic_cast<NoiseInputLayer<float>*>(lcGD0GAN->_firstLayers[0]);
            SASSERT0(noiseInputLayer != NULL);

            lossLayer->setTargetValue(1.0);
            noiseInputLayer->setRegenerateNoise(false);

            if (j % debugPeriod == 0)
                printWeightAndBias(lcGD0GAN, "G init");
            networkGD0GAN->sgd(1);
            if (j % debugPeriod == 0)
                printWeightAndBias(lcGD0GAN, "G 1");
            lossG = networkGD0GAN->sgd(1);
            if (j % debugPeriod == 0)
                printWeightAndBias(lcGD0GAN, "G 2");

            lossLayer->setTargetValue(0.0);
            noiseInputLayer->setRegenerateNoise(true);

            cout << "LOSS[epoch=" << i << "/batch=" << j << "] D: " << lossD << ",G: " <<
                lossG << endl;
        }

#if 1
        if (true) {
            Layer<float>* convLayer = lcGD0GAN->_nameLayerMap["ConvLayer1"];
            const float* host_data = convLayer->_inputData[0]->host_data();
            ImageUtil<float>::saveImage(host_data, 10, 3, 64, 64);

            printf("Generated convlayer1 Data :");
            for (int i = 0; i < 30; i++) {
                printf(" %f", host_data[i]);
            }
            printf("\n");

            sleep(2);
        }

        if (true) {
            Layer<float>* htLayer = lcGD0GAN->_nameLayerMap["hypertangent"];
            const float* host_data = htLayer->_inputData[0]->host_data();
            ImageUtil<float>::saveImage(host_data, 15, 3, 64, 64);

            printf("Generated hyper tangent Data :");
            for (int i = 0; i < 30; i++) {
                printf(" %f", host_data[i]);
            }
            printf("\n");
        }
#endif
    }

    STDOUT_LOG("exit developerMain()");
}

void loadJobFile(const char* fileName, Json::Value& rootValue) {
    filebuf fb;
    if (fb.open(fileName, ios::in) == NULL) {
        fprintf(stderr, "ERROR: cannot open %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    istream is(&fb);
    Json::Reader reader;
    bool parse = reader.parse(is, rootValue);

    if (!parse) {
        fprintf(stderr, "ERROR: invalid json-format file.\n");
        fprintf(stderr, "%s\n", reader.getFormattedErrorMessages().c_str());
        fb.close();
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int     opt;


    // 처음 생각했던 것보다 실행모드의 개수가 늘었다.
    // 모드가 하나만 더 추가되면 그냥 enum type으로 모드를 정의하도록 하자.

    bool    useDeveloperMode = false; 
    bool    useSingleJobMode = false;
    bool    useRLMode = false;
    bool    useTestMode = false;

    char*   singleJobFilePath;
    char*   romFilePath;
    char*   testItemName;

    // (1) 옵션을 읽는다.
    while ((opt = getopt(argc, argv, "vdf:a:t:")) != -1) {
        switch (opt) {
        case 'v':
            printf("%s version %d.%d.%d\n", argv[0], SPARAM(VERSION_MAJOR),
                SPARAM(VERSION_MINOR), SPARAM(VERSION_PATCH));
            exit(EXIT_SUCCESS);

        case 'd':
            if (useSingleJobMode | useRLMode | useTestMode)
                printUsageAndExit(argv[0]);
            useDeveloperMode = true;
            break;

        case 'f':
            if (useDeveloperMode | useRLMode | useTestMode)
                printUsageAndExit(argv[0]);
            useSingleJobMode = true;
            singleJobFilePath = optarg;
            break;

        case 'a':
            if (useDeveloperMode | useSingleJobMode | useTestMode)
                printUsageAndExit(argv[0]);
            useRLMode = true;
            romFilePath = optarg;
            break;

        case 't':
            if (useSingleJobMode | useDeveloperMode | useRLMode)
                printUsageAndExit(argv[0]);
            useTestMode = true;
            testItemName = optarg;
            checkTestItem(testItemName);
            break;

        default:    /* ? */
            printUsageAndExit(argv[0]);
            break; 
        }
    }

    // COMMENT: 만약 이후에 인자를 받고 싶다면 optind를 기준으로 인자를 받으면 된다.
    //  ex. Usage: %s [-d | -f jobFilePath] hostPort 와 같은 식이라면
    //  hostPort = atoi(argv[optind]);로 인자값을 받으면 된다.
    //  개인적으로 host port와 같은 정보는 SPARAM으로 정의하는 것을 더 선호한다.

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

    if (!useDeveloperMode) {
        HotLog::init();
    	HotLog::launchThread(SPARAM(CONSUMER_COUNT) + 1);
    }
    SYS_LOG("Logging system is initialized...");

    // (4) 뉴럴 네트워크 관련 기본 설정을 한다.
    //     TODO: SPARAM의 인자로 대체하자.
	cout.precision(7);
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	Util::setPrint(false);

    // (5) 모드에 따른 동작을 수행한다.
    if (useDeveloperMode) {
        // (5-A-1) Cuda를 생성한다.
        Cuda::create(SPARAM(GPU_COUNT));
        COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

        // (5-A-2) DeveloperMain()함수를 호출한다.
        developerMain();

        // (5-A-3) 자원을 해제 한다.
        Cuda::destroy();
    } else if (useSingleJobMode) {
        // FIXME: we do not support this mode until declaration of job type is done
#if 0
        // TODO: 아직 만들다 말았음
        // (5-B-1) Job File(JSON format)을 로딩한다.
        Json::Value rootValue;
        loadJobFile(singleJobFilePath, rootValue);

        // (5-B-2) Producer&Consumer를 생성.
        Worker<float>::launchThreads(SPARAM(CONSUMER_COUNT));
        
        // (5-B-3) Network를 생성한다.
        // TODO: Network configuration에 대한 정의 필요
        // XXX: 1개의 Network만 있다고 가정하고 있음.
        string networkConf = rootValue.get("Network", "").asString();
        int networkId = Worker<float>::createNetwork();
        Network<float>* network = Worker<float>::getNetwork(networkId); 
        SASSUME0(network);
        
        // (5-B-4) Job을 생성한다.
        // TODO: Job configuration에 대한 정의 필요
        Json::Value jobConfList = rootValue["Job"];
        for (int jobIndex = 0; jobIndex < jobConfList.size(); jobIndex++) {
            Json::Value jobConf = jobConfList[jobIndex];
            SASSUME0(jobConf.size() == 2);
            
            Job* newJob = new Job((Job::JobType)(jobConf[0].asInt()), network,
                                (jobConf[1].asInt()));
            Worker<float>::pushJob(newJob);
        }
#endif
        
        // (5-B-5) Worker Thread (Producer& Consumer)를 종료를 요청한다.
        Job* haltJob = new Job(Job::HaltMachine);
        Worker<float>::pushJob(haltJob);

        // (5-B-6) Producer&Consumer를 종료를 기다린다.
        Worker<float>::joinThreads();
    } else if (useRLMode) {
        // (5-C-1) Producer&Consumer를 생성.
        Worker<float>::launchThreads(SPARAM(CONSUMER_COUNT));

        // (5-C-3) Layer를 생성한다.
        Atari::run(romFilePath);

        Worker<float>::joinThreads();
    } else if (useTestMode) {
        // (5-D-1) Producer&Consumer를 생성.
        Worker<float>::launchThreads(SPARAM(CONSUMER_COUNT));

        // (5-D-2) Listener & Sess threads를 생성.
        Communicator::launchThreads(SPARAM(SESS_COUNT));

        // (5-D-3) 테스트를 실행한다.
        runTest(testItemName);

        // (5-D-4) release resources 
        Job* haltJob = new Job(Job::HaltMachine);
        Worker<float>::pushJob(haltJob);

        Communicator::halt();       // threads will be eventually halt

        // (5-D-5) 각각의 쓰레드들의 종료를 기다린다.
        Worker<float>::joinThreads();
        Communicator::joinThreads();
    } else {
        // (5-E-1) Producer&Consumer를 생성.
        Worker<float>::launchThreads(SPARAM(CONSUMER_COUNT));

        // (5-E-2) Listener & Sess threads를 생성.
        Communicator::launchThreads(SPARAM(SESS_COUNT));

        // (5-E-3) 각각의 쓰레드들의 종료를 기다린다.
        Worker<float>::joinThreads();
        Communicator::joinThreads();
    }

    // (6) 로깅 관련 모듈이 점유했던 자원을 해제한다.
    if (!useDeveloperMode)
        HotLog::destroy();
    ColdLog::destroy();
    SysLog::destroy();
    Broker::destroy();

    // (7) 서버 종료 시간을 측정하고, 계산하여 서버 실행 시간을 출력한다.
    SPERF_END(SERVER_RUNNING_TIME, startTime);
    STDOUT_LOG("server running time : %lf\n", SPERF_TIME(SERVER_RUNNING_TIME));
	STDOUT_BLOCK(cout << "SOOOA engine ends" << endl;);

    InitParam::destroy();
	exit(EXIT_SUCCESS);
}



#else

const char          SERVER_HOSTNAME[] = {"localhost"};
int main(int argc, char** argv) {
    Client::clientMain(SERVER_HOSTNAME, Communicator::LISTENER_PORT);
	exit(EXIT_SUCCESS);
}
#endif
#endif

