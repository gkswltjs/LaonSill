#include <cstdint>
#include <vector>

#include "cuda/Cuda.h"

#include "common.h"
#include "dataset/DataSet.h"
#include "dataset/MockDataSet.h"
#include "debug/Debug.h"
#include "evaluation/Evaluation.h"
#include "evaluation/Top1Evaluation.h"
#include "evaluation/Top5Evaluation.h"
#include "monitor/NetworkMonitor.h"
#include "network/Network.h"
#include "network/NetworkConfig.h"
#include "Util.h"
#include "Worker.h"
#include "Job.h"
#include "Communicator.h"
#include "Client.h"
#include "param/InitParam.h"
#include "param/Param.h"
#include "log/ColdLog.h"
#include "log/SysLog.h"
#include "log/HotLog.h"

using namespace std;

#ifndef CLIENT_MODE
void network_load();

int main(int argc, char** argv) {
	cout << "SOOOA engine starts" << endl;
    // (1) load init param
    InitParam::init();

    SysLog::init();
    ColdLog::init();
    HotLog::init();

    // consumer thread + producer thread
    HotLog::launchThread(SPARAM(CONSUMER_COUNT) + 1);

    SYS_LOG("Logging system is initialized...");

    // (2) 기본 설정
	cout.precision(11);
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	Util::setPrint(false);

    // (3) Producer&Consumer를 생성.
    Worker<float>::launchThreads(SPARAM(CONSUMER_COUNT));

    // (4) Listener & Sess threads를 생성.
    Communicator::launchThreads(SPARAM(SESS_COUNT));

    // (5) 종료
    Worker<float>::joinThreads();
    Communicator::joinThreads();

    HotLog::destroy();
    ColdLog::destroy();
    SysLog::destroy();

	cout << "SOOOA engine ends" << endl;
	return 0;
}

void network_load() {
	Cuda::create(0);
	cout << "Cuda creation done ... " << endl;

	//DataSet<float>* dataSet = createMnistDataSet<float>();
	DataSet<float>* dataSet = createImageNet1000DataSet<float>();
	dataSet->load();

	Evaluation<float>* top1Evaluation = new Top1Evaluation<float>();
	Evaluation<float>* top5Evaluation = new Top5Evaluation<float>();

	// save file 경로로 builder 생성,
	NetworkConfig<float>::Builder* networkBuilder = new NetworkConfig<float>::Builder();
	networkBuilder->load(SPARAM(NETWORK_SAVE_DIR));
	networkBuilder->dataSet(dataSet);
	networkBuilder->evaluations({top1Evaluation, top5Evaluation});

	NetworkConfig<float>* networkConfig = networkBuilder->build();
	networkConfig->load();

	Network<float>* network = new Network<float>(networkConfig);
	//network->sgd(1);
	network->test();

	Cuda::destroy();
}
#else

const char          SERVER_HOSTNAME[] = {"localhost"};
int main(int argc, char** argv) {
    Client::clientMain(SERVER_HOSTNAME, Communicator::LISTENER_PORT);
	return 0;
}
#endif
