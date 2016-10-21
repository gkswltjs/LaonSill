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
#include "network/NetworkConfig.h"

template<typename Dtype>
atomic<int> Worker<Dtype>::runningPeerCount;
template<typename Dtype>
mutex Worker<Dtype>::peerMutex;
template<typename Dtype>
condition_variable Worker<Dtype>::peerCondVar;
template<typename Dtype>
thread_local atomic<long> Worker<Dtype>::peerStep;
template<typename Dtype>
atomic<long> Worker<Dtype>::peerDoneStep;

template<typename Dtype>
int Worker<Dtype>::consumerCount;
template <typename Dtype>
thread_local int Worker<Dtype>::consumerIdx;
template<typename Dtype>
volatile void* Worker<Dtype>::consumerJob;

template<typename Dtype>
mutex Worker<Dtype>::consumerMutex;
template<typename Dtype>
condition_variable Worker<Dtype>::consumerCondVar;
template<typename Dtype>
vector<ConsumerStatus::Type> Worker<Dtype>::consumerStatuses;
template<typename Dtype>
thread_local long Worker<Dtype>::consumerJobStep;
template<typename Dtype>
long Worker<Dtype>::consumerCurJobStep;
template<typename Dtype>
atomic<int> Worker<Dtype>::wakeupConsumerJobCount;

template<typename Dtype>
mutex Worker<Dtype>::producerMutex;
template<typename Dtype>
condition_variable Worker<Dtype>::producerCondVar;

template <typename Dtype>
list<Job<Dtype>*> Worker<Dtype>::jobQueue;
template <typename Dtype>
mutex Worker<Dtype>::jobQueueMutex;

template <typename Dtype>
thread_local int Worker<Dtype>::gpuIdx;

template<typename Dtype>
thread* Worker<Dtype>::producer;
template<typename Dtype>
vector<thread> Worker<Dtype>::consumers;

template<typename Dtype>
atomic<int> Worker<Dtype>::readyCount;

template <typename Dtype>
bool Worker<Dtype>::isReady() {
    int readyCount = atomic_load(&Worker<Dtype>::readyCount);

    if (readyCount < 1)     // 1 producer + N consumer >= 2
        return false;

    if (readyCount != Worker<Dtype>::consumerCount + 1)
        return false;
    
    return true;
}

const int PRODUCER_PERIODIC_CHECK_MSEC_TIME = 3 * 1000;

template <typename Dtype>
void Worker<Dtype>::producerThread() {
	cout << "producer_thread starts" << endl;
    atomic_fetch_add(&Worker<Dtype>::readyCount, 1); 

    // (1) 서버가 준비 될때까지 대기 한다.
    while (!Worker<Dtype>::isReady()) {
        sleep(0);
    }

    // (2) 메인 루프
    while (true) {
        unique_lock<mutex> producerLock(Worker<Dtype>::producerMutex);
        Worker<Dtype>::producerCondVar.wait_for(producerLock,
            chrono::milliseconds(PRODUCER_PERIODIC_CHECK_MSEC_TIME));
        producerLock.unlock();

        // (2-1) 멈춘 Consumer Thread를 재개
        //     더 최적화 할 수 있으나.. 그다지 성능향상이 필요없는 부분이라.. 대충짰음.
        long peerDoneStep = atomic_load(&Worker<Dtype>::peerDoneStep);
        for (int i = 0; i < Worker<Dtype>::consumerCount; i++) {
            long peerStep = atomic_load(&Worker<Dtype>::peerStep);

            if (peerStep < peerDoneStep) {
                unique_lock<mutex> peerLock(Worker<Dtype>::peerMutex);
                Worker<Dtype>::peerCondVar.notify_all();
                peerLock.unlock();
                break; 
            }
        }
        
        // (2-2) Consumer Thread가 종료 되었으면 job이 있는지 확인
        bool canStartNewJob = true;
        for (int i = 0; i < Worker<Dtype>::consumerCount; i++) {
            if (Worker<Dtype>::consumerStatuses[i] == ConsumerStatus::Running) {
                canStartNewJob = false;
                break;
            }
        }

        if (!canStartNewJob)
            continue;

        Job<Dtype>* job = popJob();
        if (job == NULL)
            continue;

        // (2-3) consumer thread들을 깨운다.
        Worker<Dtype>::consumerCurJobStep += 1L;

        consumerJob = job;
        atomic_store(&Worker<Dtype>::wakeupConsumerJobCount, 0);

        atomic_store(&Worker<Dtype>::peerDoneStep, 0L);
        atomic_store(&Worker<Dtype>::runningPeerCount, Worker<Dtype>::consumerCount);

        unique_lock<mutex> consumerLock(Worker<Dtype>::consumerMutex);
        Worker<Dtype>::consumerCondVar.notify_all();
        consumerLock.unlock();

        // (2-4) 혹시 안깨워진 consumer가 있는지 체크하고 깨운다.
        while (atomic_load(&Worker<Dtype>::wakeupConsumerJobCount) <
                Worker<Dtype>::consumerCount) {
            consumerLock.lock();
            Worker<Dtype>::consumerCondVar.notify_all();
            consumerLock.unlock();
            sleep(0);
        }

        // (2-5) 종료를 요청한 작업인 경우 종료 한다.
        if (job->getType() == Job<Dtype>::HaltMachine)
            break;
    }

    cout << "producer_thread ends" << endl;
}

template <typename Dtype>
void Worker<Dtype>::consumerThread(int consumerIdx, int gpuIdx) {
    bool doLoop = true;
	Worker<Dtype>::consumerIdx = consumerIdx;
	Worker<Dtype>::gpuIdx = gpuIdx;
    Worker<Dtype>::consumerJobStep = 0L;
    atomic_fetch_add(&Worker<Dtype>::readyCount, 1); 

	cout << "consumer_thread #" << consumerIdx << "(GPU:#" << gpuIdx << ") starts" << endl;

	// 리소스 초기화
	checkCudaErrors(cudaSetDevice(gpuIdx));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    while (doLoop) {
        unique_lock<mutex> consumerLock(Worker<Dtype>::consumerMutex);
        Worker<Dtype>::consumerCondVar.wait(consumerLock,
            [] { return (Worker<Dtype>::consumerCurJobStep == 
                        Worker<Dtype>::consumerJobStep + 1L); });
        consumerLock.unlock();
        Worker<Dtype>::consumerJobStep += 1L;
        atomic_fetch_add(&Worker<Dtype>::wakeupConsumerJobCount, 1);
        atomic_store(&Worker<Dtype>::peerStep, 0L);

        Worker<Dtype>::consumerStatuses[consumerIdx] = ConsumerStatus::Running;
        Job<Dtype>* job = (Job<Dtype>*)(Worker::consumerJob);

        switch (job->getType()) {
        case Job<Dtype>::BuildLayer:
            buildLayer(job->getNetwork());
            break;
        case Job<Dtype>::TrainNetwork:
            trainNetwork(job->getNetwork(), job->getArg1());
            break;
        case Job<Dtype>::CleanupLayer:
            cleanupLayer(job->getNetwork());
            break;
        case Job<Dtype>::HaltMachine:
            doLoop = false;
            Worker<Dtype>::consumerStatuses[consumerIdx] = ConsumerStatus::Waiting;
            break;
        default:
            assert(!"Invalid job type");
        }

        Worker<Dtype>::consumerStatuses[consumerIdx] = ConsumerStatus::Waiting;
    }

	// 리소스 정리
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));

	cout << "consumer_thread #" << consumerIdx << "(GPU:#" << gpuIdx << ") ends" << endl;
}

template <typename Dtype>
void Worker<Dtype>::launchThread(int consumerCount) {
    // (1) Cuda를 생성한다.
    Cuda::create(consumerCount);
	cout << "Cuda creation done ... " << endl;

    // (2) Worker Count를 설정한다.
    if (consumerCount > Cuda::gpuCount) {
        printf("ERROR: Invalid GPU count of Worker. (There are %d available GPU but requested"
            " GPU count of Worker is %d\n", Cuda::gpuCount, consumerCount);
        exit(1);
    }
	Worker<Dtype>::consumerCount = consumerCount;
    Worker<Dtype>::consumerCurJobStep = 0L;
    Worker<Dtype>::consumerStatuses.assign(consumerCount, ConsumerStatus::Waiting);

	// (3) producer 쓰레드를 생성한다.
	producer = new thread(producerThread);
    cout << "launchThread GPUCount " << Worker<Dtype>::consumerCount << endl;

	// (4) consumer 쓰레드들을 생성한다.
	for (int i = 0; i < Worker<Dtype>::consumerCount; i++) {
		consumers.push_back(thread(consumerThread, i, Cuda::availableGPU[i]));
	}

    // (5) producer, consumer 쓰레드가 종료하기를 기다린다.
	for (int i = 0; i < Worker<Dtype>::consumerCount; i++) {
		consumers[i].join();
	}
	producer->join();
	free(producer);
}

template <typename Dtype>
bool Worker<Dtype>::waitPeer() {
    if (atomic_fetch_sub(&Worker<Dtype>::runningPeerCount, 1) == 1) {
        atomic_store(&Worker<Dtype>::runningPeerCount, Worker<Dtype>::consumerCount);
        return true; 
    } else {
        unique_lock<mutex> peerLock(Worker<Dtype>::peerMutex);
        Worker<Dtype>::peerCondVar.wait(peerLock, []
            { return (atomic_load(&Worker<Dtype>::peerDoneStep) ==
                atomic_load(&Worker<Dtype>::peerStep) + 1L); });
        peerLock.unlock();

        atomic_fetch_add(&Worker<Dtype>::peerStep, 1L);
        return false;
    }
}

template <typename Dtype>
void Worker<Dtype>::wakeupPeer() {
    atomic_fetch_add(&Worker<Dtype>::peerStep, 1L);
    atomic_fetch_add(&Worker<Dtype>::peerDoneStep, 1L);
    unique_lock<mutex> peerLock(Worker<Dtype>::peerMutex);
    Worker<Dtype>::peerCondVar.notify_all();
    peerLock.unlock();
}

template <typename Dtype>
void Worker<Dtype>::pushJob(Job<Dtype>* job) {
    Worker<Dtype>::jobQueueMutex.lock();
    Worker<Dtype>::jobQueue.push_back(job);
    Worker<Dtype>::jobQueueMutex.unlock();

    unique_lock<mutex> producerLock(Worker<Dtype>::producerMutex);
    Worker<Dtype>::producerCondVar.notify_one();
    producerLock.unlock();
}

template <typename Dtype>
Job<Dtype>* Worker<Dtype>::popJob() {
    Job<Dtype>* popedJob;
    Worker<Dtype>::jobQueueMutex.lock();
    
    if (Worker<Dtype>::jobQueue.empty()) {
        Worker<Dtype>::jobQueueMutex.unlock();
        return NULL;
    }

    popedJob = Worker<Dtype>::jobQueue.front();
    Worker<Dtype>::jobQueue.pop_front();
    Worker<Dtype>::jobQueueMutex.unlock();

    return popedJob;
}

template <typename Dtype>
void Worker<Dtype>::buildLayer(Network<Dtype>* network) {
    // XXX: 현재는 CCN Double Layer만 생성하도록 되어 있다. 수정필요!!!
    
    // (1) layer config를 만든다. 이 과정중에 layer들의 초기화가 진행된다.
	LayersConfig<float>* layersConfig = createCNNDoubleLayersConfig<float>();

    // (2) network config 정보를 layer들에게 전달한다.
    for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
        layersConfig->_layers[i]->setNetworkConfig(network->config);
    }

    // (3) shape 과정을 수행한다. 
    io_dim in_dim;
    in_dim.rows = network->config->_dataSet->getRows();
    in_dim.cols = network->config->_dataSet->getCols();
    in_dim.channels = network->config->_dataSet->getChannels();
    in_dim.batches = network->config->_batchSize;
    layersConfig->_inputLayer->shape(0, in_dim);

    // (4) network에 layersConfig 정보를 등록한다.
    network->setLayersConfig(layersConfig);
}

template <typename Dtype>
void Worker<Dtype>::trainNetwork(Network<Dtype>* network, int maxEpochs) {
    if (consumerIdx == 0)
	    cout << "maxEpoch: " << maxEpochs << endl;

    network->sgd_with_timer(maxEpochs);

    // XXX: save() 함수 확인 다시하자.
    //if (consumerIdx == 0)
    //    network->save();
}

template <typename Dtype>
void Worker<Dtype>::cleanupLayer(Network<Dtype>* network) {
    delete network->getLayersConfig()->_inputLayer;
}

template class Worker<float>;
