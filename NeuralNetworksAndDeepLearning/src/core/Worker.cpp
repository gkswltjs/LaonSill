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
#include "WorkContext.h"

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

bool Worker::handleAllocTensorTask(TaskAllocTensor* task) {

}

bool Worker::handleUpdateTensorTask(TaskUpdateTensor* task) {
#if 0
                bool ret = Updater::updateParams(tasks[i]->networkID, 
                    tasks[i]->layerID, tasks[i]->planID,
                    tasks[i]->dopID, tasks[i]->updateParams, false);
#endif
}

bool Worker::handleRunPlanTask(TaskRunPlan* task) {
#if 0
            for (int i = 0; i < taskDefs.size(); i++) {
                WorkContext::updateNetwork(taskDefs[i].networkID);
                WorkContext::updatePlan(taskDefs[i].dopID);
            }

#endif
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

    vector<TaskBase*> remainTasks;
    while (doLoop) {
        ThreadEvent event =
            ThreadMgmt::wait(threadID, SPARAM(TASK_CONSUMER_PERIODIC_CHECK_TIME_MS)); 

        if (event == ThreadEvent::Wakeup || event == ThreadEvent::Timeout) {
            vector<TaskBase*> tasks;

            TaskQueue *tq = taskQueues[consumerIdx];
            unique_lock<mutex> lock(tq->mutex);

            while (!tq->tasks.empty()) {
                tasks.push_back(tq->tasks.front());
                tq->tasks.pop_front();
            }
            lock.unlock();

            for (int i = 0; i < remainTasks.size(); i++) {
                tasks.push_back(remainTasks[i]);
            }
            remainTasks.clear();

            bool hasRemainTask = false;
            for (int i = 0; i < tasks.size(); i++) {
                bool ret;
                switch (tasks[i]->taskType) {
                    case TaskType::AllocTensor:
                        ret = handleAllocTensorTask((TaskAllocTensor*)tasks[i]);
                        break;

                    case TaskType::UpdateTensor:
                        ret = handleUpdateTensorTask((TaskUpdateTensor*)tasks[i]);
                        break;

                    case TaskType::RunPlan:
                        ret = handleRunPlanTask((TaskRunPlan*)tasks[i]);
                        break;

                    default:
                        SASSUME0(false);
                }

                if (!ret) {
                    remainTasks.push_back(tasks[i]);
                    hasRemainTask = true;
                } else {
                    Task::releaseElem(tasks[i]->taskType, (void*)tasks[i]);
                }
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

void Worker::addAllocTensorTask(int consumerIdx, int nodeID, int devID, int requestThreadID,
    string tensorName) {
    TaskAllocTensor* task = (TaskAllocTensor*)Task::getElem(TaskType::AllocTensor);
    SASSUME0(task != NULL);     // pool이 넉넉하지 않을때에 대한 전략이 반드시 필요하다

    task->nodeID = nodeID;
    task->devID = devID;
    task->requestThreadID = requestThreadID;
    task->tensorName = tensorName;

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->tasks.push_back((TaskBase*)task);
}

void Worker::addRunPlanTask(int consumerIdx, int networkID, int dopID) {
    TaskRunPlan* task = (TaskRunPlan*)Task::getElem(TaskType::RunPlan);
    SASSUME0(task != NULL);     // pool이 넉넉하지 않을때에 대한 전략이 반드시 필요하다

    task->networkID = networkID;
    task->dopID = dopID;

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->tasks.push_back((TaskBase*)task);
}

void Worker::addUpdateTensorTask(int consumerIdx, int networkID, int dopID, int layerID,
    int planID, vector<UpdateParam> updateParams) {

    TaskUpdateTensor* task = (TaskUpdateTensor*)Task::getElem(TaskType::UpdateTensor);
    SASSUME0(task != NULL);

    task->networkID = networkID;
    task->dopID = dopID;
    task->layerID = layerID;
    task->planID = planID;

    for (int i = 0 ; i < updateParams.size(); i++)
        task->updateParams.push_back(updateParams[i]);

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->tasks.push_back((TaskBase*)task);
}

int Worker::getConsumerIdx(int devIdx) {
    for (int i = 0; i < Cuda::availableGPU.size(); i++) {
        if (Cuda::availableGPU[i] == devIdx)
            return i;
    }

    return -1;
}
