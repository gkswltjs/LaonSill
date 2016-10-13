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

#include "network/Network.h"

using namespace std;

/**
 * @brief 각종 유틸리티 함수들을 정적으로 포함하는 클래스
 * @details
 */
template <typename Dtype>
class Worker {
public:
	Worker() {}
	Worker(int gpuCount);
	virtual ~Worker();

    static atomic<int> runningThreadCnt; // XXX: 예쁘게.. 나중에..
    static vector<NetworkConfig<Dtype>*> configs;    // XXX: 예쁘게.. 나중에..
	static mutex commMutex;
	static condition_variable commCondV;

	static int gpuCount;
	static thread_local int consumerIdx;

	/**
	 * @brief producer, consumer 쓰레드를 실행한다.
	 * @param network producer 쓰레드가 담당할 network
	 * @details
	 * 			producer 쓰레드는 인자로 받은 network에서 정의된 작업을 수행한다.
	 * 			정의된 작업은 현재는 network의 sgd()를 수행하는 것으로 한정되어 있다.
	 * @todo
	 * 		  	(1) 동작시킬 network를 전달하는 방식에 대해서 수정이 필요 하다.
	 *        	현재는 1개의 네트워크를 전달하고, 그 네트워크가 수행해야할 모든 동작이 완료가 되면,
	 *       	producer와 consumer가 종료되는 방식이다.
	 *       	producer는 계속 살아 있고, 명령을 받으면 수행하는 형태로 수정이 필요하다.
	 *	      	(2) 1개의 producer만 존재하지만, 여러개의 producer가 필요할 수 있다.
	 *	      	성능을 보고 판단해야 한다.
	 *	      	(3) 어떤 작업을 수행할 지에 대한 유연한 정의가 필요하다. (ex. job description)
	 */
	void launchThread();

private:
	static thread_local int gpuIdx;

	static void producer_thread();
	static void consumer_thread(int consumerIdx, int gpuIdx);

	static thread* producer;
	static vector<thread> consumers;

};

#endif /* WORKER_H_ */
