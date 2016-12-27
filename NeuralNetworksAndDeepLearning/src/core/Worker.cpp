/*
 * Worker.cpp
 *
 *  Created on: 2016. 10. 5.
 *      Author: moonhoen lee
 */

#include "Cuda.h"
#include "Worker.h"

#include "Debug.h"
#include "NetworkConfig.h"
#include "Param.h"
#include "ColdLog.h"
#include "HotLog.h"
#include "SysLog.h"
#include "ALEInputLayer.h"
#include "Broker.h"
#include "DQNWork.h"
#include "LegacyWork.h"

using namespace std;

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
int Worker<Dtype>::consumerCount = 1;
template <typename Dtype>
thread_local int Worker<Dtype>::consumerIdx = 0;
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
list<Job*> Worker<Dtype>::jobQueue;
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

template<typename Dtype>
bool Worker<Dtype>::useWorker = false;

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
    COLD_LOG(ColdLog::INFO, true, "producer thread starts");
    atomic_fetch_add(&Worker<Dtype>::readyCount, 1); 
    
    HotLog::initForThread();

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

        Job* job = popJob();
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
        if (job->getType() == Job::HaltMachine)
            break;
    }

    COLD_LOG(ColdLog::INFO, true, "producer thread ends");
    HotLog::markExit();
}

template <typename Dtype>
void Worker<Dtype>::consumerThread(int consumerIdx, int gpuIdx) {
    bool doLoop = true;
	Worker<Dtype>::consumerIdx = consumerIdx;
	Worker<Dtype>::gpuIdx = gpuIdx;
    Worker<Dtype>::consumerJobStep = 0L;
    atomic_fetch_add(&Worker<Dtype>::readyCount, 1); 

    HotLog::initForThread();

    COLD_LOG(ColdLog::INFO, true, "consumer thread #%d (GPU:#%d) starts", consumerIdx, gpuIdx);

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
        Job* job = (Job*)Worker::consumerJob;

        switch (job->getType()) {
            case Job::BuildNetwork:
                LegacyWork<Dtype>::buildNetwork(job);
                break;

            case Job::TrainNetwork:
                LegacyWork<Dtype>::trainNetwork(job);
                break;

            case Job::CleanupNetwork:
                LegacyWork<Dtype>::cleanupNetwork(job);
                break;

            /* DQN related Jobs */
            case Job::CreateDQNImageLearner:
                DQNWork<Dtype>::createDQNImageLearner(job);
                break;

            case Job::BuildDQNNetworks:
                DQNWork<Dtype>::buildDQNNetworks(job);
                break;

            case Job::StepDQNImageLearner:
                DQNWork<Dtype>::stepDQNImageLearner(job);
                break;

            case Job::FeedForwardDQNNetwork:
                DQNWork<Dtype>::feedForwardDQNNetwork(job);
                break;

            case Job::HaltMachine:
                doLoop = false;
                Worker<Dtype>::consumerStatuses[consumerIdx] = ConsumerStatus::Waiting;
                break;

            default:
                SASSERT(false, "Invalid job type");
        }

        Worker<Dtype>::consumerStatuses[consumerIdx] = ConsumerStatus::Waiting;
        // resource 해제
        if (atomic_fetch_sub(&job->refCnt, 1) == 1) {
            delete job;
            job = NULL;
        }
    }

	// 리소스 정리
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));

    HotLog::markExit();
    COLD_LOG(ColdLog::INFO, true, "consumer thread #%d (GPU:#%d) ends", consumerIdx, gpuIdx);
}

template <typename Dtype>
void Worker<Dtype>::launchThreads(int consumerCount) {
    // (1) Cuda를 생성한다.
    Cuda::create(consumerCount);
    COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

    Worker<Dtype>::useWorker = true;

    // (2) Worker Count를 설정한다.
    if (consumerCount > Cuda::gpuCount) {
        SYS_LOG("ERROR: Invalid GPU count of Worker. ");
        SYS_LOG("There are %d available GPU but requested GPU count of Worker is %d.",
            Cuda::gpuCount, consumerCount);
        exit(1);
    }
	Worker<Dtype>::consumerCount = consumerCount;
    Worker<Dtype>::consumerCurJobStep = 0L;
    Worker<Dtype>::consumerStatuses.assign(consumerCount, ConsumerStatus::Waiting);

	// (3) producer 쓰레드를 생성한다.
    Worker<Dtype>::producer = new thread(producerThread);

	// (4) consumer 쓰레드들을 생성한다.
	for (int i = 0; i < Worker<Dtype>::consumerCount; i++) {
		Worker<Dtype>::consumers.push_back(thread(consumerThread, i, Cuda::availableGPU[i]));
	}
}

template <typename Dtype>
void Worker<Dtype>::joinThreads() {
	for (int i = 0; i < Worker<Dtype>::consumerCount; i++) {
		Worker<Dtype>::consumers[i].join();
	}

	Worker<Dtype>::producer->join();
	delete Worker<Dtype>::producer;
    Worker<Dtype>::producer = NULL;
}

template <typename Dtype>
bool Worker<Dtype>::waitPeer() {
    // XXX: 아래의 코드는 오직 1개의 GPU를 사용하는 developer mode에서만 유효하다.
    if (!useWorker)
        return false;

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
int Worker<Dtype>::pushJob(Job* job) {
    int pubJobID = -1;

    // (1) pubJob이 있는 경우에 pubJob을 생성하고 pubJob ID를 할당받는다.
    if (job->hasPubJob()) {
        Job* pubJob = new Job(job->getPubJobType());
        unique_lock<mutex> reqPubJobMapLock(Job::reqPubJobMapMutex); 
        Job::reqPubJobMap[job->getJobID()] = pubJob; 
        // subscriber will deallocate pubJob
        reqPubJobMapLock.unlock();
        pubJobID = pubJob->getJobID();
    }

    // (2) job queue에 job을 넣는다.
    Worker<Dtype>::jobQueueMutex.lock();
    Worker<Dtype>::jobQueue.push_back(job);
    Worker<Dtype>::jobQueueMutex.unlock();

    // (3) 프로듀서에게 새로운 잡이 추가되었음을 알려준다.
    unique_lock<mutex> producerLock(Worker<Dtype>::producerMutex);
    Worker<Dtype>::producerCondVar.notify_one();
    producerLock.unlock();

    return pubJobID;
}

template <typename Dtype>
Job* Worker<Dtype>::popJob() {
    Job* popedJob;
    Worker<Dtype>::jobQueueMutex.lock();
    
    if (Worker<Dtype>::jobQueue.empty()) {
        Worker<Dtype>::jobQueueMutex.unlock();
        return NULL;
    }

    popedJob = Worker<Dtype>::jobQueue.front();
    Worker<Dtype>::jobQueue.pop_front();
    Worker<Dtype>::jobQueueMutex.unlock();

    atomic_store(&popedJob->refCnt, Worker<Dtype>::consumerCount);

    return popedJob;
}

template class Worker<float>;
