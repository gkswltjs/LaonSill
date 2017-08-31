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
#include "Broker.h"
#include "ThreadMgmt.h"
#include "Updater.h"
#include "WorkContext.h"
#include "PlanParser.h"
#include "LayerFunc.h"
#include "PropMgmt.h"
#include "StdOutLog.h"

#include "InputDataProvider.h"
#include "AnnotationDataLayer.h"
#include "DetectionOutputLayer.h"

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
    int threadID = ThreadMgmt::getThreadID(ThreadType::Producer, 0);
    ThreadMgmt::setThreadReady(threadID);
    COLD_LOG(ColdLog::INFO, true, "producer thread starts");
    
    HotLog::initForThread();

    // (1) 서버가 준비 될때까지 대기 한다.
    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }


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
    if (task->nodeID != SPARAM(NODE_ID)) {
        // alloc tensor to other nodes.
        SASSERT0(false);        // not implemented yet
    }

    // XXX: float형 코딩으로 박지 말고, 설정에 따라서 template date type을 설정하도록 수정해야
    //     한다. 
    if (task->step == TaskAllocTensorStep::Alloc) {
        Data<float>* tensor = new Data<float>(task->tensorName);
        SASSERT0(tensor != NULL);

        task->tensorPtr = tensor;
        task->step = TaskAllocTensorStep::Done;
        
        ThreadMgmt::signal(task->requestThreadID, ThreadEvent::Wakeup);
    }

    return true;
}

bool Worker::handleUpdateTensorTask(TaskUpdateTensor* task) {
    bool ret = Updater::updateParams(task->networkID, task->layerID, task->planID,
        task->dopID, task->updateParams, false);
    return ret;
}

bool Worker::handleRunPlanTask(TaskRunPlan* task) {
    WorkContext::updateNetwork(task->networkID);
    WorkContext::updatePlan(task->dopID, true);

    PhysicalPlan* pp = PhysicalPlan::getCurPhysicalPlan();
    bool canRunPlan = true;
    while (canRunPlan) {
        canRunPlan = pp->runPlan(task->inference);
    }

    bool jobRemain = pp->generatePlan(true);

    if (jobRemain) {
        return false;
    } else {
        bool runFinished = false;
        unique_lock<mutex> lock(WorkContext::curPlanInfo->planMutex);
        WorkContext::curPlanInfo->doneCount += 1;
        if (WorkContext::curPlanInfo->doneCount == WorkContext::curPlanInfo->dopCount)
            runFinished = true;
        lock.unlock();

        if (runFinished) {
            ThreadMgmt::signal(task->requestThreadID, ThreadEvent::Wakeup); 
        }
        
        return true;
    }
}

bool Worker::handleAllocLayerTask(TaskAllocLayer* task) {
    if (task->nodeID != SPARAM(NODE_ID)) {
        // alloc tensor to other nodes.
        SASSERT0(false);        // not implemented yet
    }

    WorkContext::updateLayer(task->networkID, task->layerID);
    WorkContext::updatePlan(task->dopID, true);

    SASSERT0(LayerFunc::allocLayerTensors(task->layerType, task->instancePtr) == true);
    ThreadMgmt::signal(task->requestThreadID, ThreadEvent::Wakeup);

    return true;
}

void Worker::taskConsumerThread(int consumerIdx, int gpuIdx) {
    int threadID = ThreadMgmt::getThreadID(ThreadType::TaskConsumer, consumerIdx);
    ThreadMgmt::setThreadReady(threadID);
    bool doLoop = true;
	Worker::gpuIdx = gpuIdx;

    HotLog::initForThread();

    COLD_LOG(ColdLog::INFO, true, "task consumer thread #%d (GPU:#%d) starts", consumerIdx,
        gpuIdx);

    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }


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
                bool taskDone;
                switch (tasks[i]->taskType) {
                    case TaskType::AllocTensor:
                        taskDone = handleAllocTensorTask((TaskAllocTensor*)tasks[i]);
                        break;

                    case TaskType::UpdateTensor:
                        taskDone = handleUpdateTensorTask((TaskUpdateTensor*)tasks[i]);
                        break;

                    case TaskType::RunPlan:
                        taskDone = handleRunPlanTask((TaskRunPlan*)tasks[i]);
                        break;

                    case TaskType::AllocLayer:
                        taskDone = handleAllocLayerTask((TaskAllocLayer*)tasks[i]);
                        break;

                    default:
                        SASSUME0(false);
                }

                if (!taskDone) {
                    remainTasks.push_back(tasks[i]);
                    hasRemainTask = true;
                } else if (tasks[i]->taskType != TaskType::AllocTensor) {
                    // Alloc Tensor의 경우에는 caller가 release한다.
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
    int threadID = ThreadMgmt::getThreadID(ThreadType::JobConsumer, consumerIdx);
    ThreadMgmt::setThreadReady(threadID);
    bool doLoop = true;

    HotLog::initForThread();

    COLD_LOG(ColdLog::INFO, true, "job consumer thread #%d (GPU:#%d) starts", consumerIdx,
        gpuIdx);

    while (!ThreadMgmt::isReady()) {
        sleep(0);
    }

    while (doLoop) {
        //ThreadEvent event = ThreadMgmt::wait(threadID, SPARAM(JOB_CONSUMER_PERIODIC_CHECK_TIME_MS)); 
        ThreadEvent event = ThreadMgmt::wait(threadID, 0UL);

        if (event == ThreadEvent::Halt) {
            break;
        }

        Job* job = Worker::popJob();
        if (job == NULL)
            continue;

        doLoop = handleJob(job);

        delete job;

        Worker::insertJCReadyQueue(consumerIdx);
    }

    HotLog::markExit();
    COLD_LOG(ColdLog::INFO, true, "job consumer thread #%d (GPU:#%d) ends", consumerIdx,
        gpuIdx);
}

Job* Worker::getPubJob(Job* job) {
    SASSUME0(job->hasPubJob());
    unique_lock<mutex> reqPubJobMapLock(Job::reqPubJobMapMutex); 
    Job *pubJob = Job::reqPubJobMap[job->getJobID()];
    SASSUME0(pubJob != NULL);
    Job::reqPubJobMap.erase(job->getJobID());
    reqPubJobMapLock.unlock();
    SASSUME0(pubJob->getType() == job->getPubJobType());
    return pubJob;
}

void Worker::handleCreateNetworkFromFileJob(Job* job) {
    int networkID = PlanParser::loadNetwork(job->getStringValue(0));
    
    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::IntType, 1, (void*)&networkID);

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleCreateNetwork(Job* job) {
    int networkID = PlanParser::loadNetworkByJSONString(job->getStringValue(0));

    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::IntType, 1, (void*)&networkID);

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleDestroyNetwork(Job* job) {
    int networkID = job->getIntValue(0);

    // XXX: 네트워크가 제거될 수 있는 상황인지에 대한 파악을 해야하고, 그것에 따른 에러처리가
    //      필요하다.
    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    SASSERT0(network->getLoaded());

    if (SPARAM(USE_INPUT_DATA_PROVIDER)) {
        InputDataProvider::removePool(networkID);
    }

    LogicalPlan::cleanup(networkID);

    if (network->getBuilt())
        PhysicalPlan::removePlan(networkID);

    PropMgmt::removeNetworkProp(networkID);
    PropMgmt::removeLayerProp(networkID);
    delete network;

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleBuildNetwork(Job* job) {
    int networkID = job->getIntValue(0);
    int epochs = job->getIntValue(1);

    cout << " build network" << endl;

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->build(epochs);

    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::IntType, 1, (void*)&networkID);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleResetNetwork(Job* job) {
    int networkID = job->getIntValue(0);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->reset();

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunNetwork(Job* job) {
    int networkID = job->getIntValue(0);
    int inference = job->getIntValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);

    if (SPARAM(USE_INPUT_DATA_PROVIDER)) {
        WorkContext::updateNetwork(networkID);

        Job* startIDPJob = new Job(JobType::StartInputDataProvider);   // InputDataProvider
        startIDPJob->addJobElem(Job::IntType, 1, (void*)&networkID);
        Worker::pushJob(startIDPJob);
    }

    network->run((bool)inference);

    ThreadMgmt::wait(WorkContext::curThreadID, 0);

    int clientError = 1;
    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::IntType, 1, (void*)&clientError);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunNetworkMiniBatch(Job* job) {
    int networkID = job->getIntValue(0);
    int inference = job->getIntValue(1);
    int miniBatchIdx = job->getIntValue(2);

    int clientError = 1;

    if (SPARAM(USE_INPUT_DATA_PROVIDER)) {
        COLD_LOG(ColdLog::ERROR, true, "run network minibatch is not supported in IDP mode");
        clientError = 0;
    } else {
        Network<float>* network = Network<float>::getNetworkFromID(networkID);
        network->runMiniBatch((bool)inference, miniBatchIdx);

        ThreadMgmt::wait(WorkContext::curThreadID, 0);
    }

    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::IntType, 1, (void*)&clientError);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleSaveNetwork(Job* job) {
    int networkID = job->getIntValue(0);
    string filePath = job->getStringValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->save(filePath);

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleLoadNetwork(Job* job) {
    int networkID = job->getIntValue(0);
    string filePath = job->getStringValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->load(filePath);

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunNetworkWithInputData(Job* job) {
    int networkID = job->getIntValue(0);
    int channel = job->getIntValue(1);
    int height = job->getIntValue(2);
    int width = job->getIntValue(3);
    float* imageData = job->getFloatArray(4);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);

    std::vector<Layer<float>*> inputLayers =
        network->findLayersByType(Layer<float>::AnnotationData);
    SASSUME0(inputLayers.size() == 1);
    AnnotationDataLayer<float>* inputLayer = (AnnotationDataLayer<float>*)inputLayers[0];

    std::vector<Layer<float>*> outputLayers =
        network->findLayersByType(Layer<float>::DetectionOutput);
    SASSUME0(outputLayers.size() == 1);
    DetectionOutputLayer<float>* outputLayer =
        (DetectionOutputLayer<float>*)outputLayers[0];

    inputLayer->feedImage(channel, height, width, imageData);
    network->runMiniBatch(true, 0);

    ThreadMgmt::wait(WorkContext::curThreadID, 0);

    Job* pubJob = getPubJob(job);

    int count = outputLayer->_outputData[0]->getCount();
    const float* result = outputLayer->_outputData[0]->host_data();
    int resultCount = 0;

    for (int i = 0; i < count; i += 7) {
    	if (result[i + 1] == 15) {
    		resultCount++;
    	}
    }
    pubJob->addJobElem(Job::IntType, 1, (void*)&resultCount);

    for (int i = 0; i < count; i++) {
    	if (result[i * 7 + 1] != 15) {
    		continue;
    	}

        int left    = int(result[i * 7 + 3] * width) ;
        int top     = int(result[i * 7 + 4] * height);
        int right   = int(result[i * 7 + 5] * width);
        int bottom  = int(result[i * 7 + 6] * height);
        float score = result[i * 7 + 2];

        pubJob->addJobElem(Job::IntType, 1, (void*)&top);
        pubJob->addJobElem(Job::IntType, 1, (void*)&left);
        pubJob->addJobElem(Job::IntType, 1, (void*)&bottom);
        pubJob->addJobElem(Job::IntType, 1, (void*)&right);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&score);
    }

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleStartIDP(Job* job) {
    int networkID = job->getIntValue(0);
    InputDataProvider::handleIDP(networkID);
}

bool Worker::handleJob(Job* job) {
    bool doLoop = true;

    switch (job->getType()) {
        case JobType::HaltMachine:
            doLoop = false;
            ThreadMgmt::signalAll(ThreadEvent::Halt);
            break;

        case JobType::CreateNetworkFromFile:
            handleCreateNetworkFromFileJob(job);
            break;

        case JobType::CreateNetwork:
            handleCreateNetwork(job);
            break;

        case JobType::DestroyNetwork:
            handleDestroyNetwork(job);
            break;

        case JobType::BuildNetwork:
            handleBuildNetwork(job);
            break;

        case JobType::ResetNetwork:
            handleResetNetwork(job);
            break;

        case JobType::RunNetwork:
            handleRunNetwork(job);
            break;

        case JobType::RunNetworkMiniBatch:
            handleRunNetworkMiniBatch(job);
            break;

        case JobType::SaveNetwork:
            handleSaveNetwork(job);
            break;

        case JobType::LoadNetwork:
            handleLoadNetwork(job);
            break;

        case JobType::RunNetworkWithInputData:
            handleRunNetworkWithInputData(job);
            break;

        case JobType::StartInputDataProvider:
            handleStartIDP(job);
            break;

        default:
            SASSERT(false, "Invalid job type");
    }

    return doLoop;
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

TaskAllocTensor* Worker::addAllocTensorTask(int consumerIdx, int nodeID, int devID,
    int requestThreadID, string tensorName) {
    TaskAllocTensor* task = (TaskAllocTensor*)Task::getElem(TaskType::AllocTensor);
    SASSUME0(task != NULL);     // pool이 넉넉하지 않을때에 대한 전략이 반드시 필요하다

    task->nodeID = nodeID;
    task->devID = devID;
    task->requestThreadID = requestThreadID;
    task->tensorName = tensorName;
    task->step = TaskAllocTensorStep::Alloc;

    SASSUME0(consumerIdx < Worker::taskQueues.size());
    TaskQueue* tq = Worker::taskQueues[consumerIdx];

    unique_lock<mutex> lock(tq->mutex);
    tq->tasks.push_back((TaskBase*)task);

    return task;
}

void Worker::addRunPlanTask(int consumerIdx, int networkID, int dopID, bool inference,
    int requestThreadID) {
    TaskRunPlan* task = (TaskRunPlan*)Task::getElem(TaskType::RunPlan);
    SASSUME0(task != NULL);     // pool이 넉넉하지 않을때에 대한 전략이 반드시 필요하다

    task->networkID = networkID;
    task->dopID = dopID;
    task->inference = inference;
    task->requestThreadID = requestThreadID;

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

void Worker::addAllocLayerTask(int consumerIdx, int networkID, int dopID, int layerID,
    int nodeID, int devID, int requestThreadID, int layerType, void* instancePtr) {
    
    TaskAllocLayer* task = (TaskAllocLayer*)Task::getElem(TaskType::AllocLayer);
    SASSUME0(task != NULL);

    task->networkID = networkID;
    task->dopID = dopID;
    task->layerID = layerID;
    task->nodeID = nodeID;
    task->devID = devID;
    task->requestThreadID = requestThreadID;
    task->layerType = layerType;
    task->instancePtr = instancePtr;

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
