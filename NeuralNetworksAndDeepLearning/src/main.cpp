#if 1
#include <fcntl.h>
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
#include "DebugUtil.h"
#include "ResourceManager.h"
#include "PlanOptimizer.h"

#include "GAN.h"
#include "YOLO.h"
#include "LayerFunc.h"
#include "LayerPropList.h"

using namespace std;

#ifndef CLIENT_MODE

void printUsageAndExit(char* prog) {
    fprintf(stderr,
        "Usage: %s [-v] [-d | -f jobFilePath | -a romFilePath | -t testItemName]\n", prog);
    exit(EXIT_FAILURE);
}

void developerMain() {
    STDOUT_LOG("enter developerMain()");

    checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    YOLO<float>::runPretrain();
    //YOLO<float>::run();

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

    ResourceManager::init();
    PlanOptimizer::init();
    LayerFunc::init();
    LayerPropList::init();

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

    LayerFunc::destroy();
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

