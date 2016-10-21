/**
 * @file Worker.h
 * @date 2016/10/5
 * @author mhlee
 * @brief 병렬작업을 위한 worker 쓰레드를 관리
 * @details
 * @todo
 */

#ifndef WORKER_H_
#define WORKER_H_

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <list>
#include <chrono>

#include "Job.h"
#include "network/Network.h"

using namespace std;

class ConsumerStatus {
public:
    enum Type {
        Waiting = 0,
        Running,
    };

    ConsumerStatus () {};
    virtual ~ConsumerStatus() {};
};


/**
 * @brief 각종 유틸리티 함수들을 정적으로 포함하는 클래스
 * @details
 */
template <typename Dtype>
class Worker {
public:
	Worker() {
        Worker<Dtype>::readyCount = 0;
        Worker<Dtype>::consumerCount = 0;
    }
	virtual ~Worker() {}

    /**
     * consumer에 대한 정보를 담고 있는 변수들
     */
	static int consumerCount;
	static thread_local int consumerIdx;

    /**
     * @return      마지막으로 깨어난 consumer는 true, 그 외에는 false를 반환
     */
    static bool waitPeer();
    static void wakeupPeer();

	/**
	 * @brief producer, consumer 쓰레드를 실행한다.
	 * @param network producer 쓰레드가 담당할 network
	 */
	static void launchThread(int consumerCount);

    static void pushJob(Job<Dtype>* job);

    static bool isReady();
private:
    /**
     * Consumer간의 동기화를 지원하기 위한 변수들
     *
     *  작업 A,B가 있다. 각각의 작업은 A = {A1, A2, A3, ... An}, B = {B1, B2, ... Bn}로 나누어
     * 작업이 수행이 될 수 있다. n개의 consumer들은 A1, A2에 대한 작업을 수행한다. B라는
     * 작업은 모든 A라는 작업이 종료가 되고 수행이 되어야 한다. 이 때 n개의 consumer가 A라는
     * 작업을 끝내고, B라는 작업을 수행하기 위한 동기화가 필요하다. 
     */
    static atomic<int> runningPeerCount;
	static mutex peerMutex;
	static condition_variable peerCondVar;
	static thread_local atomic<long> peerStep;
    static atomic<long> peerDoneStep;

    /**
     * consumer에 대한 job control을 위한 변수들
     */
	static mutex consumerMutex;
	static condition_variable consumerCondVar;
    static vector<ConsumerStatus::Type> consumerStatuses;
    static thread_local long consumerJobStep;
    static long consumerCurJobStep;
    static atomic<int> wakeupConsumerJobCount;
    static volatile void* consumerJob;

    /**
     * producer에 대한 job control을 위한 변수들
     */
	static mutex producerMutex;
	static condition_variable producerCondVar;

    static list<Job<Dtype>*> jobQueue;
    static mutex jobQueueMutex;

    static atomic<int> readyCount;

	static thread_local int gpuIdx;

    static Job<Dtype>* popJob();

    static void buildLayer(Network<Dtype>* network);
    static void trainNetwork(Network<Dtype>* network, int maxEpochs);
    static void cleanupLayer(Network<Dtype>* network); 

	static void producerThread();
	static void consumerThread(int consumerIdx, int gpuIdx);

	static thread* producer;
	static vector<thread> consumers;
};

#endif /* WORKER_H_ */
