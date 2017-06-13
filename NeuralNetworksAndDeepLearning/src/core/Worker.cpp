/*
 * Worker.cpp
 *
 *  Created on: 2016. 10. 5.
 *      Author: moonhoen lee
 */

#include "Cuda.h"
#include "Worker.h"

#include "Debug.h"
#include "Network.h"
#include "Param.h"
#include "ColdLog.h"
#include "HotLog.h"
#include "SysLog.h"
#include "ALEInputLayer.h"
#include "Broker.h"
#include "DQNWork.h"
#include "LegacyWork.h"
#include "ThreadMgmt.h"
#include "Updater.h"

using namespace std;

thread_local int        Worker::gpuIdx;

list<Job*>              Worker::jobQueue;
mutex                   Worker::jobQueueMutex;

list<int>               Worker::jcReadyQueue;   /* job consumer ready queue */
mutex                   Worker::jcReadyMutex;

vector<TaskQueue*>      Worker::taskQueues;

thread*                 Worker::producer;
vector<thread>          Worker::consumers;

void Worker::producerThread() {
    ThreadMgmt::setThreadReady();
    COLD_LOG(ColdLog::INFO, true, "producer thread starts");
    
    HotLog::initForThread();

    // (1) 서버가 준비 될때까지 대기 한다.
    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }

    int threadID = ThreadMgmt::getThreadID(ThreadType::Producer, 0);

    // (2) 메인 루프
    while (true) {
        ThreadEvent event =
            ThreadMgmt::wait(threadID, SPARAM(PRODUCER_PERIODIC_CHECK_TIME_MS)); 
        int wakeupCount = Worker::getJobCount();
        vector<int> jcIDs = Worker::getReadyJCs(wakeupCount);

        for (int i = 0; i < jcIDs.size(); i++) {
            int targetThreadID = ThreadMgmt::getThreadID(ThreadType::JobConsumer, jcIDs[i]);
            ThreadMgmt::signal(targetThreadID, ThreadEvent::Wakeup);
        }

        if (event == ThreadEvent::Halt)
            break;
    }

    COLD_LOG(ColdLog::INFO, true, "producer thread ends");
    HotLog::markExit();
}

void Worker::taskConsumerThread(int consumerIdx, int gpuIdx) {
    ThreadMgmt::setThreadReady();
    bool doLoop = true;
	Worker::gpuIdx = gpuIdx;

    HotLog::initForThread();

    COLD_LOG(ColdLog::INFO, true, "task consumer thread #%d (GPU:#%d) starts", consumerIdx,
        gpuIdx);

    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }

    int threadID = ThreadMgmt::getThreadID(ThreadType::TaskConsumer, consumerIdx);

	// 리소스 초기화
	checkCudaErrors(cudaSetDevice(gpuIdx));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

    vector<UpdaterTaskDef*> remainUpdaterTaskDefs;
    while (doLoop) {
        ThreadEvent event =
            ThreadMgmt::wait(threadID, SPARAM(TASK_CONSUMER_PERIODIC_CHECK_TIME_MS)); 

        if (event == ThreadEvent::Wakeup || event == ThreadEvent::Timeout) {
            vector<TaskDef> taskDefs;
            vector<UpdaterTaskDef*> updaterTaskDefs;

            vector<TaskDef> removeTaskDefs;

            TaskQueue *tq = taskQueues[consumerIdx];
            unique_lock<mutex> lock(tq->mutex);

            // 전에 못했던 것들을 먼저 updaterTaskDefs에 추가한다.
            for (int i = 0; i < remainUpdaterTaskDefs.size(); i++) {
                updaterTaskDefs.push_back(remainUpdaterTaskDefs[i]);
            }
            remainUpdaterTaskDefs.clear();

            while (!tq->updaterTaskDefs.empty()) {
                updaterTaskDefs.push_back(tq->updaterTaskDefs.front());
                tq->updaterTaskDefs.pop_front();
            }

            for (int i = 0; i < tq->taskDefs.size(); i++) {
                taskDefs.push_back(tq->taskDefs[i]);
            }
            lock.unlock();

            bool hasRemainTask = false;
            for (int i = 0; i < updaterTaskDefs.size(); i++) {
                bool ret = Updater::updateParams(updaterTaskDefs[i]->networkID, 
                    updaterTaskDefs[i]->layerID, updaterTaskDefs[i]->planID,
                    updaterTaskDefs[i]->dopID, updaterTaskDefs[i]->updateParams, false);

                if (!ret) {
                    remainUpdaterTaskDefs.push_back(updaterTaskDefs[i]);
                    hasRemainTask = true;
                } else {
                    delete updaterTaskDefs[i];
                }
            }

            for (int i = 0; i < taskDefs.size(); i++) {
                // TODO: run run run
            }

            // 남은 task가 있다면 자기 스스로를 깨운다.
            if (hasRemainTask)
                ThreadMgmt::signal(threadID, ThreadEvent::Wakeup);
                
        }

        if (event == ThreadEvent::Halt)
            break;
    }

	// 리소스 정리
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));

    HotLog::markExit();
    COLD_LOG(ColdLog::INFO, true, "task consumer thread #%d (GPU:#%d) ends", consumerIdx,
        gpuIdx);
}

void Worker::jobConsumerThread(int consumerIdx) {
    ThreadMgmt::setThreadReady();
    bool doLoop = true;

    HotLog::initForThread();

    COLD_LOG(ColdLog::INFO, true, "job consumer thread #%d (GPU:#%d) starts", consumerIdx,
        gpuIdx);

    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }

    int threadID = ThreadMgmt::getThreadID(ThreadType::JobConsumer, consumerIdx);

    while (doLoop) {
        ThreadEvent event =
            ThreadMgmt::wait(threadID, SPARAM(JOB_CONSUMER_PERIODIC_CHECK_TIME_MS)); 

        if (event == ThreadEvent::Halt) {
            break;
        }

        Job* job = Worker::popJob();
        if (job == NULL)
            continue;

        switch (job->getType()) {
            case Job::HaltMachine:
                doLoop = false;
                ThreadMgmt::signalAll(ThreadEvent::Halt);
                break;

            default:
                SASSERT(false, "Invalid job type");
        }

    }

    HotLog::markExit();
    COLD_LOG(ColdLog::INFO, true, "job consumer thread #%d (GPU:#%d) ends", consumerIdx,
        gpuIdx);
}

void Worker::launchThreads(int taskConsumerCount, int jobConsumerCount) {
    // (1) Cuda를 생성한다.
    Cuda::create(SPARAM(GPU_COUNT));
    COLD_LOG(ColdLog::INFO, true, "CUDA is initialized");

    // (2) Worker Count를 설정한다.
    if (taskConsumerCount > Cuda::gpuCount) {
        SYS_LOG("ERROR: Invalid GPU count of Worker. ");
        SYS_LOG("There are %d available GPU but requested GPU count of Worker is %d.",
            Cuda::gpuCount, taskConsumerCount);
        exit(1);
    }

	// (3) producer 쓰레드를 생성한다.
    Worker::producer = new thread(producerThread);

	// (4) consumer 쓰레드들을 생성한다.
	for (int i = 0; i < SPARAM(GPU_COUNT); i++) {
		Worker::consumers.push_back(thread(taskConsumerThread, i, Cuda::availableGPU[i]));
        TaskQueue *tq = new TaskQueue();
        Worker::taskQueues.push_back(tq);
    }

    for (int i = 0; i < SPARAM(JOB_CONSUMER_COUNT); i++) {
        Worker::consumers.push_back(thread(jobConsumerThread, i));
        Worker::jcReadyQueue.push_back(i);
    }
}

void Worker::joinThreads() {
	for (int i = 0; i < Worker::consumers.size(); i++) {
		Worker::consumers[i].join();
        // XXX: 그냥 쓰레드는 메모리해제 관련하여 아무 작업도 필요 없나? 확인해봐야 함!!
	}
    Worker::consumers.clear();

	Worker::producer->join();
	delete Worker::producer;
    Worker::producer = NULL;

    for (int i = 0; i < Worker::taskQueues.size(); i++) {
        delete Worker::taskQueues[i];
    }

    Worker::taskQueues.clear();
}

int Worker::pushJob(Job* job) {
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
    Worker::jobQueueMutex.lock();
    Worker::jobQueue.push_back(job);
    Worker::jobQueueMutex.unlock();

    // (3) 프로듀서에게 새로운 잡이 추가되었음을 알려준다.
    ThreadMgmt::signal(ThreadMgmt::getThreadID(ThreadType::Producer, 0), ThreadEvent::Wakeup);

    return pubJobID;
}

Job* Worker::popJob() {
    Job* popedJob;
    Worker::jobQueueMutex.lock();
    
    if (Worker::jobQueue.empty()) {
        Worker::jobQueueMutex.unlock();
        return NULL;
    }

    popedJob = Worker::jobQueue.front();
    Worker::jobQueue.pop_front();
    Worker::jobQueueMutex.unlock();

    return popedJob;
}

int Worker::getJobCount() {
    unique_lock<mutex> lock(Worker::jobQueueMutex);
    return jobQueue.size();
}

void Worker::insertJCReadyQueue(int consumerIdx) {
    unique_lock<mutex> lock(Worker::jcReadyMutex);
    jcReadyQueue.push_back(consumerIdx);
}

vector<int> Worker::getReadyJCs(int count) {
    vector<int> result;
    unique_lock<mutex> lock(Worker::jcReadyMutex);
    for (int i = 0; i < count; i++) {
        if (Worker::jcReadyQueue.empty())
            break;

        int popedJCIdx = Worker::jcReadyQueue.front();
        Worker::jcReadyQueue.pop_front();
        result.push_back(popedJCIdx);
    }
    lock.unlock();
    return result;
}

void Worker::addTaskQueue(int consumerIdx, int networkID, int dopID) {
    TaskDef taskDef;
    taskDef.networkID = networkID;
    taskDef.dopID = dopID;

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->taskDefs.push_back(taskDef);
}

void Worker::addUpdaterTask(int consumerIdx, int networkID, int dopID, int layerID,
    int planID, vector<UpdateParam> updateParams) {

    UpdaterTaskDef *def = new UpdaterTaskDef(); // 이거는 Task Consumer가 메모리 해제한다.
    def->networkID = networkID;
    def->dopID = dopID;
    def->layerID = layerID;
    def->planID = planID;
    for (int i = 0 ; i < updateParams.size(); i++)
        def->updateParams.push_back(updateParams[i]);

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->updaterTaskDefs.push_back(def);
}

int Worker::getConsumerIdx(int devIdx) {
    for (int i = 0; i < Cuda::availableGPU.size(); i++) {
        if (Cuda::availableGPU[i] == devIdx)
            return i;
    }

    return -1;
}
