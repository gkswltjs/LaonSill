/**
 * @file Worker.h
 * @date 2016/10/5
 * @author moonhoen lee
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

#include "common.h"
#include "Job.h"
#include "Update.h"

typedef struct UpdaterTaskDef_s {
    int                         networkID;
    int                         dopID;
    int                         layerID;
    int                         planID;
    std::vector<UpdateParam>    updateParams;
} UpdaterTaskDef;

typedef struct TaskDef_s {
    int networkID;
    int dopID;
} TaskDef;

typedef struct TaskQueue_s {
    std::mutex                      mutex;
    std::vector<TaskDef>            taskDefs;
    std::list<UpdaterTaskDef*>      updaterTaskDefs;
} TaskQueue;

class Worker {
public:
	                                    Worker() {}
	virtual                            ~Worker() {}

	static void                         launchThreads(int taskConsumerCount, 
                                                      int jobConsumerCount);
    static void                         joinThreads();
    static int                          pushJob(Job* job);
                                        /* called by Sess Thread, Receiver Thread */
	static thread_local int             gpuIdx;

    static void                         addUpdaterTask(int consumerIdx, int networkID, 
                                                       int dopID, int layerID, int planID,
                                                       std::vector<UpdateParam> updateParams);
    static int                          getConsumerIdx(int devIdx);

private:
    /**
     * producer에 대한 job control을 위한 변수들
     */
    static std::list<Job*>              jobQueue;
    static std::mutex                   jobQueueMutex;
    static Job*                         popJob();
    static int                          getJobCount();
	static void                         producerThread();

    /**
     * variables and functions for job consumer
     */
	static void                         jobConsumerThread(int consumerIdx);
    static std::list<int>               jcReadyQueue;   /* job consumer ready queue */
    static std::mutex                   jcReadyMutex;
    static void                         insertJCReadyQueue(int consumerIdx);
    static std::vector<int>             getReadyJCs(int count);


    /**
     * variables and functions for task consumer
     */
	static void                         taskConsumerThread(int consumerIdx,
                                                           int gpuIdx);
    static std::vector<TaskQueue*>      taskQueues;
    static void                         addTaskQueue(int consumerIdx, int networkID,
                                                             int dopID);


    /*
     * variables for joining thread
     */
	static std::thread*                 producer;
	static std::vector<std::thread>     consumers;  // for join threads
};

#endif /* WORKER_H_ */
