/*
 * Worker.cpp
 *
 *  Created on: 2016. 10. 5.
 *      Author: mhlee
 */

#include "cuda/Cuda.h"
#include "Worker.h"

#include "debug/Debug.h"
#include "evaluation/Evaluation.h"
#include "evaluation/Top1Evaluation.h"
#include "evaluation/Top5Evaluation.h"
#include "monitor/NetworkMonitor.h"
#include "network/Network.h"
#include "network/NetworkConfig.h"

template<typename Dtype>
vector<NetworkConfig<Dtype>*> Worker<Dtype>::configs;

template<typename Dtype>
atomic<int> Worker<Dtype>::runningThreadCnt;

template<typename Dtype>
mutex Worker<Dtype>::commMutex;

template<typename Dtype>
condition_variable Worker<Dtype>::commCondV;

template<typename Dtype>
int Worker<Dtype>::gpuCount;

template<typename Dtype>
thread* Worker<Dtype>::producer;

template<typename Dtype>
vector<thread> Worker<Dtype>::consumers;

template <typename Dtype>
thread_local int Worker<Dtype>::gpuIdx;

template <typename Dtype>
thread_local int Worker<Dtype>::consumerIdx;

template <typename Dtype>
Worker<Dtype>::Worker(int gpuCount) {
    if (gpuCount > Cuda::gpuCount) {
        printf("ERROR: Invalid GPU count of Worker. (There are %d available GPU but requested"
            " GPU count of Worker is %d\n", Cuda::gpuCount, gpuCount);
        exit(1);
    }
	Worker<Dtype>::gpuCount = gpuCount;
}


template <typename Dtype>
Worker<Dtype>::~Worker() {

}


template <typename Dtype>
void Worker<Dtype>::producer_thread() {
	cout << "producer_thread starts" << endl;
}

template <typename Dtype>
void Worker<Dtype>::consumer_thread(int consumerIdx, int gpuIdx) {
	Worker<Dtype>::consumerIdx = consumerIdx;
	Worker<Dtype>::gpuIdx = gpuIdx;

	cout << "consumer_thread #" << consumerIdx << "(GPU:#" << gpuIdx << ") starts" << endl;

	// 리소스 초기화
	checkCudaErrors(cudaSetDevice(gpuIdx));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

	//const uint32_t maxEpoch = 10000;
	//const uint32_t maxEpoch = 4;
	const uint32_t maxEpoch = 2;
    const uint32_t batchSize = 50;
	//const uint32_t batchSize = 1000;
	//const uint32_t testInterval = 20;			// 10000(목표 샘플수) / batchSize
	const uint32_t testInterval = 1000000;			// 10000(목표 샘플수) / batchSize
	//const uint32_t saveInterval = 20000;		// 1000000 / batchSize
	const uint32_t saveInterval = 1000000;		// 1000000 / batchSize
	const float baseLearningRate = 0.001f;
	const float weightDecay = 0.0002f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;


	cout << "maxEpoch: " << maxEpoch << endl;
	cout << "batchSize: " << batchSize << endl;
	cout << "testInterval: " << testInterval << endl;
	cout << "saveInterval: " << saveInterval << endl;
	cout << "baseLearningRate: " << baseLearningRate << endl;
	cout << "weightDecay: " << weightDecay << endl;
	cout << "momentum: " << momentum << endl;
	cout << "clipGradientsLevel: " << clipGradientsLevel << endl;

	//SyncMem<float>::setOutstream("./mem");

	//DataSet<float>* dataSet = new MockDataSet<float>(4, 4, 2, 20, 20, 10, MockDataSet<float>::NOTABLE_IMAGE);
	//DataSet<float>* dataSet = new MockDataSet<float>(28, 28, 1, 100, 100, 10);
	//DataSet<float>* dataSet = new MockDataSet<float>(56, 56, 3, 10, 10, 10);
	//DataSet<float>* dataSet = new MnistDataSet<float>(0.8);
	//DataSet<float>* dataSet = new MockDataSet<float>(224, 224, 3, 100, 100, 100);
	//DataSet<float>* dataSet = createImageNet10CatDataSet<float>();
	//DataSet<float>* dataSet = createImageNet100CatDataSet<float>();
	//DataSet<float>* dataSet = createImageNet1000DataSet<float>();
	//DataSet<float>* dataSet = createImageNet10000DataSet<float>();
	//DataSet<float>* dataSet = createImageNet50000DataSet<float>();
	DataSet<float>* dataSet = createMnistDataSet<float>();
	//DataSet<float>* dataSet = createSampleDataSet<float>();
	dataSet->load();
	//dataSet->zeroMean(true);

	Evaluation<float>* top1Evaluation = new Top1Evaluation<float>();
	Evaluation<float>* top5Evaluation = new Top5Evaluation<float>();
	//NetworkListener* networkListener = new NetworkMonitor(NetworkMonitor::PLOT_AND_WRITE);
	NetworkListener* networkListener = new NetworkMonitor(NetworkMonitor::WRITE_ONLY);

	//LayersConfig<float>* layersConfig = createCNNSimpleLayersConfig<float>();
	LayersConfig<float>* layersConfig = createCNNDoubleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception3ALayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception3ALayersConfigTest<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception3ASimpleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception5BLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInceptionAuxLayersConfig<float>();

	NetworkConfig<float>* networkConfig =
			(new NetworkConfig<float>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->testInterval(testInterval)
			->saveInterval(saveInterval)
			->clipGradientsLevel(clipGradientsLevel)
			->dataSet(dataSet)
			->evaluations({top1Evaluation, top5Evaluation})
			->layersConfig(layersConfig)
			->savePathPrefix("/home/jhkim/network")
			->networkListeners({networkListener})
			->build();

	Util::printVramInfo();

	Network<float>* network = new Network<float>(networkConfig);

	//network->sgd(maxEpoch);
	//network->sgd_with_timer(maxEpoch);
	//network->save();
    
    //Worker<Dtype>::configs[consumerIdx] = networkConfig;
    Worker<Dtype>::configs.push_back(networkConfig);    // 순서는 상관 없다.

    cout << "configs count " << Worker<Dtype>::configs.size() << endl;

    network->sgd_with_timer(maxEpoch);

    if (consumerIdx == 0)
        network->save();

	// 리소스 정리
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));
}

template <typename Dtype>
void Worker<Dtype>::launchThread() {
	int i;

    //Worker<Dtype>::configs.reserve(Worker<Dtype>::gpuCount);
	// (1) producer 쓰레드를 생성한다.
	producer = new thread(producer_thread);

    // (1-1) 임시작업: running Thread Cnt수를 GPU 개수만큼으로 설정한다.
    atomic_store(&runningThreadCnt, Worker<Dtype>::gpuCount);

    cout << "launchThread GPUCount " << Worker<Dtype>::gpuCount << endl;

	// (2) consumer 쓰레드들을 생성한다.
	for (i = 0; i < Worker<Dtype>::gpuCount; i++) {
		consumers.push_back(thread(consumer_thread, i, Cuda::availableGPU[i]));
	}

	for (i = 0; i < Worker<Dtype>::gpuCount; i++) {
		consumers[i].join();
	}
	producer->join();
	free(producer);
}

template class Worker<float>;
