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

void developerMain() {
    STDOUT_LOG("enter developerMain()");

    checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));


	ArtisticStyle<float>* artisticStyle = new ArtisticStyle<float>(/*

#if !SMALL_TEST
			"/data/backup/artistic/tubingen_320.jpg",
			//"/home/jkim/Downloads/sampleR32G64B128.png",
			"/data/backup/artistic/starry_night_320.jpg",
			//"/data/backup/artistic/composition_320.jpg",
			{"conv4_2"},
			{"conv1_1"},
#else
			"/data/backup/artistic/tubingen_16.jpg",
			"/data/backup/artistic/starry_night_16.jpg",
			//{"conv1_1"},
			{},
			{"conv1_1", "conv2_1"},
#endif
			1.0f,					// contentReconstructionFactor
			100.0f,					// styleReconstructionFactor
			100.0,					// learningRate
			"relu4_2",				// end
			true,					// plotContentCost
			true					// plotStyleCost*/
	);

	artisticStyle->transfer_style();
	delete artisticStyle;



    STDOUT_LOG("exit developerMain()");
}


#define TEST_MODE 1

#if TEST_MODE
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "lmdb++.h"
#include "MathFunctions.h"
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

    auto env = lmdb::env::create();
    //env.set_mapsize(1UL * 1024UL * 1024UL * 1024UL);		// 1GB
    env.open("/home/jkim/Dev/git/caffe/examples/imagenet/ilsvrc12_train_lmdb");

    auto rtxn = lmdb::txn::begin(env, nullptr, MDB_RDONLY);
    auto dbi = lmdb::dbi::open(rtxn, nullptr);
    auto cursor = lmdb::cursor::open(rtxn, dbi);
    string key, value;
    //while (cursor.get(key, value, MDB_NEXT)) {
    cursor.get(key, value, MDB_NEXT);
    cout << "key: " << key << ", value: " << value << endl;
    //}
    cursor.close();
    rtxn.abort();

#endif

	exit(EXIT_SUCCESS);
}


#endif














