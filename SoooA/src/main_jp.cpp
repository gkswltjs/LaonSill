#if 0

#include <unistd.h>
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
#include "NetworkMonitor.h"
#include "Network.h"
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
#include "WorkContext.h"
#include "PlanParser.h"
#include "ThreadMgmt.h"
#include "Sender.h"
#include "Receiver.h"

#include "frcnn/FRCNN.h"
#include "YOLO.h"
#include "LayerFunc.h"
#include "LayerPropList.h"

#include "StdOutLog.h"

using namespace std;



void developerMain() {
    STDOUT_LOG("enter developerMain()");

    checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    FRCNN<float>::run();

    STDOUT_LOG("exit developerMain()");
}


int main(int argc, char** argv) {
    int     opt;


    // 처음 생각했던 것보다 실행모드의 개수가 늘었다.
    // 모드가 하나만 더 추가되면 그냥 enum type으로 모드를 정의하도록 하자.

    bool    useDeveloperMode = false;
    bool    useSingleJobMode = false;
    bool    useTestMode = false;

    char*   singleJobFilePath;
    char*   testItemName;


    // (2) 서버 시작 시간 측정을 시작한다.
    struct timespec startTime;
    SPERF_START(SERVER_RUNNING_TIME, &startTime);
	STDOUT_BLOCK(cout << "LAONSILL engine starts" << endl;);

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

    int threadCount = 0;
    SYS_LOG("Logging system is initialized...");

    // (4) 뉴럴 네트워크 관련 기본 설정을 한다.
	cout.precision(SPARAM(COUT_PRECISION));
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	Util::setPrint(false);

    // (5) 모드에 따른 동작을 수행한다.
    // DeveloperMode와 SingleJobMode는 1쓰레드(메인쓰레드)로만 동작을 한다.
    // TestMode와 DefaultMode(ServerClientMode)는 여러 쓰레드로 동작을 하게 된다.

	// (5-A-1) Cuda를 생성한다.
	Cuda::create(SPARAM(GPU_COUNT));
	COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

	// (5-A-2) DeveloperMain()함수를 호출한다.
	developerMain();

	// (5-A-3) 자원을 해제 한다.
	Cuda::destroy();

    LayerFunc::destroy();
    // (6) 로깅 관련 모듈이 점유했던 자원을 해제한다.
    if (threadCount > 0) {
        ThreadMgmt::destroy();
        HotLog::destroy();
    }
    ColdLog::destroy();
    SysLog::destroy();
    Broker::destroy();

    // (7) 서버 종료 시간을 측정하고, 계산하여 서버 실행 시간을 출력한다.
    SPERF_END(SERVER_RUNNING_TIME, startTime);
    STDOUT_LOG("server running time : %lf\n", SPERF_TIME(SERVER_RUNNING_TIME));
	STDOUT_BLOCK(cout << "LAONSILL engine ends" << endl;);

    InitParam::destroy();
	exit(EXIT_SUCCESS);
}

#endif
