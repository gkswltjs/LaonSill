#include <cstdint>
#include <iostream>
#include <vector>

#include "cuda/Cuda.h"
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

using namespace std;


void network_test();
void network_load();

// XXX: 임시...
const int CONSUMER_THREAD_COUNT = 2;

int main(int argc, char** argv) {
    // (1) 기본 설정
	cout << "NN engine starts" << endl;
	cout.precision(11);
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	Util::setPrint(false);

    // (2) 테스트를 위한 testThread
    //     추후에는 없어질 예정.
    thread testThread = thread(network_test);

    // (3) Producer&Consumer를 생성.
    Worker<float>* worker = new Worker<float>();
    worker->launchThread(CONSUMER_THREAD_COUNT);

    // (4) 테스트 쓰레드 종료 확인
    testThread.join();

    // (5) 종료
	cout << "NN engine ends" << endl;
	return 0;
}

void network_test() {

    // (1) Worker의 준비가 될때까지 기다린다.
    while (!Worker<float>::isReady()) {
        sleep(1);
    }

    // (2) Network를 생성한다.
    const uint32_t batchSize = 50;
	//const uint32_t batchSize = 1000;
	//const uint32_t testInterval = 20;			// 10000(목표 샘플수) / batchSize
	const uint32_t testInterval = 1000000;			// 10000(목표 샘플수) / batchSize
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

	DataSet<float>* dataSet = createMnistDataSet<float>();
	dataSet->load();

	Evaluation<float>* top1Evaluation = new Top1Evaluation<float>();
	Evaluation<float>* top5Evaluation = new Top5Evaluation<float>();
	NetworkListener* networkListener = new NetworkMonitor(NetworkMonitor::WRITE_ONLY);

	NetworkConfig<float>* networkConfig =
			(new NetworkConfig<float>::Builder())
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
			->dataSet(dataSet)
			->evaluations({top1Evaluation, top5Evaluation})
			->savePathPrefix("/home/jhkim/network")
			->networkListeners({networkListener})
			->build();

	Util::printVramInfo();

	Network<float>* network = new Network<float>(networkConfig);

    // (3) Job을 생성한다.
    Job<float>* job1 = new Job<float>(Job<float>::BuildLayer, network, 0);
    Job<float>* job2 = new Job<float>(Job<float>::TrainNetwork, network, 2);
    Job<float>* job3 = new Job<float>(Job<float>::CleanupLayer, network, 0);
    Job<float>* job4 = new Job<float>(Job<float>::HaltMachine, network, 0);

    // (4) Job을 집어 넣는다.
    Worker<float>::pushJob(job1);
    Worker<float>::pushJob(job2);
    Worker<float>::pushJob(job3);
    Worker<float>::pushJob(job4);
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
	networkBuilder->load("/home/jhkim/network");
	networkBuilder->dataSet(dataSet);
	networkBuilder->evaluations({top1Evaluation, top5Evaluation});

	NetworkConfig<float>* networkConfig = networkBuilder->build();
	networkConfig->load();

	Network<float>* network = new Network<float>(networkConfig);
	//network->sgd(1);
	network->test();

	Cuda::destroy();
}
