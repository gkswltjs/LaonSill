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
#include "RoITestLiveInputLayer.h"
#include "FrcnnTestLiveOutputLayer.h"
#include "LiveDataInputLayer.h"

#include "MeasureEntry.h"
#include "MeasureManager.h"
#include "MemoryMgmt.h"
#include "DebugUtil.h"
#include "ImageUtil.h"

#include "frcnn_common.h"   // for use nsm() func

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
        Data<float>* tensor = NULL;
        SNEW(tensor, Data<float>, task->tensorName);
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
	checkCUBLAS(cublasCreate(&Cuda::cublasHandle));
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
	checkCUBLAS(cublasDestroy(Cuda::cublasHandle));
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
        ThreadEvent event = ThreadMgmt::wait(threadID, 0UL);

        if (event == ThreadEvent::Halt) {
            break;
        }

        Job* job = Worker::popJob();
        if (job == NULL)
            continue;

        doLoop = handleJob(job);

        SDELETE(job);

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
    string networkID = PlanParser::loadNetwork(job->getStringValue(0));
    
    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleCreateNetwork(Job* job) {
    string networkID = PlanParser::loadNetworkByJSONString(job->getStringValue(0));

    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleDestroyNetwork(Job* job) {
    string networkID = job->getStringValue(0);

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
    SDELETE(network);

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleBuildNetwork(Job* job) {
    string networkID = job->getStringValue(0);
    int epochs = job->getIntValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->build(epochs);

    Job* pubJob = getPubJob(job);
    pubJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleResetNetwork(Job* job) {
    string networkID = job->getStringValue(0);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->reset();

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunNetwork(Job* job) {
    string networkID = job->getStringValue(0);
    int inference = job->getIntValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);

    if (SPARAM(USE_INPUT_DATA_PROVIDER)) {
        WorkContext::updateNetwork(networkID);

        Job* startIDPJob = NULL;
        SNEW(startIDPJob, Job, JobType::StartInputDataProvider);   // InputDataProvider
        SASSUME0(startIDPJob != NULL);
        startIDPJob->addJobElem(Job::StringType, strlen(networkID.c_str()),
            (void*)networkID.c_str());
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
    string networkID = job->getStringValue(0);
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
    string networkID = job->getStringValue(0);
    string filePath = job->getStringValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->save(filePath);

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleLoadNetwork(Job* job) {
    string networkID = job->getStringValue(0);
    string filePath = job->getStringValue(1);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    network->load(filePath);

    Job* pubJob = getPubJob(job);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleGetMeasureItemName(Job* job) {
    string networkID = job->getStringValue(0);

    MeasureEntry* entry = MeasureManager::getMeasureEntry(networkID);
    int itemCount;
    Job* pubJob = getPubJob(job);

    if (entry == NULL) {
        itemCount = -1;
        pubJob->addJobElem(Job::IntType, 1, (void*)&itemCount);
    } else {
        vector<string> itemNames = entry->getItemNames();

        itemCount = itemNames.size();
        pubJob->addJobElem(Job::IntType, 1, (void*)&itemCount);

        for (int i = 0; i < itemCount; i++) {
            pubJob->addJobElem(Job::StringType, strlen(itemNames[i].c_str()),
                (void*)itemNames[i].c_str());
        }
    }

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleGetMeasures(Job* job) {
    string networkID = job->getStringValue(0);
    int isForward = job->getIntValue(1);
    int start = job->getIntValue(2);
    int count = job->getIntValue(3);

    int startIterNum;
    int measureCount;

    MeasureEntry* entry = MeasureManager::getMeasureEntry(networkID);
    Job* pubJob = getPubJob(job);

    if (entry == NULL) {
        int N = -1;
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
    } else {
        int itemCount = entry->getItemNames().size();
        float* data = NULL;
        int allocSize = sizeof(float) * count * itemCount;
        SMALLOC(data, float, allocSize);
        SASSUME0(data != NULL);

        entry->getData(start, count, (bool)isForward, &startIterNum, &measureCount, data); 

        int N = measureCount * itemCount;
        pubJob->addJobElem(Job::IntType, 1, (void*)&N);
        pubJob->addJobElem(Job::IntType, 1, (void*)&startIterNum);

        int totalIterCount;
        int curIterCount;

        WorkContext::getNetworkProgress(networkID, curIterCount, totalIterCount);
        pubJob->addJobElem(Job::IntType, 1, (void*)&curIterCount);
        pubJob->addJobElem(Job::IntType, 1, (void*)&totalIterCount);

        if (N > 0)
            pubJob->addJobElem(Job::FloatArrayType, N, data);

        SFREE(data);
    }

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunNetworkWithInputData(Job* job) {
    string networkID = job->getStringValue(0);
    int channel = job->getIntValue(1);
    int height = job->getIntValue(2);
    int width = job->getIntValue(3);
    int coordRelative = job->getIntValue(4);
    float* imageData = job->getFloatArray(5);

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

    float left, top, right, bottom;
    for (int i = 0; i < count; i += 7) {
    	if (result[i + 1] != 15) {
    		continue;
    	}

    	left	= std::min(std::max(result[i + 3], 0.f), 1.f);
    	top		= std::min(std::max(result[i + 4], 0.f), 1.f);
    	right	= std::min(std::max(result[i + 5], 0.f), 1.f);
    	bottom	= std::min(std::max(result[i + 6], 0.f), 1.f);

    	if (coordRelative == 0) {
    		left    = int(left * width);
			top     = int(top * height);
			right   = int(right * width);
			bottom  = int(bottom * height);
    	}
        float score = result[i + 2];
        int labelIndex = (int)(result[i + 1] + 0.000001);

        pubJob->addJobElem(Job::FloatType, 1, (void*)&top);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&left);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&bottom);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&right);
        pubJob->addJobElem(Job::FloatType, 1, (void*)&score);
        pubJob->addJobElem(Job::IntType, 1, (void*)&labelIndex);
    }
    Broker::publish(job->getJobID(), pubJob);
}


void Worker::handleRunObjectDetectionNetworkWithInput(Job* job) {
    string networkID = job->getStringValue(0);
    int channel = job->getIntValue(1);
    int height = job->getIntValue(2);
    int width = job->getIntValue(3);
    int baseNetworkType = job->getIntValue(4);
    float* imageData = job->getFloatArray(5);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    WorkContext::updateNetwork(networkID);
    
    InputLayer<float>* commonInputLayer;
    Layer<float>* commonOutputLayer;
    commonInputLayer = (InputLayer<float>*)network->findLayer(SNPROP(inputLayer));
    SASSUME0(commonInputLayer != NULL);
    commonOutputLayer = network->findLayer(SNPROP(outputLayer));
    SASSUME0(commonOutputLayer != NULL);

    WorkContext::updateLayer(networkID, commonInputLayer->layerID);
    commonInputLayer->feedImage(channel, height, width, imageData);
    SASSUME0(baseNetworkType < (int)WORKER_OD_eMAX);

    network->runMiniBatch(true, 0);
    ThreadMgmt::wait(WorkContext::curThreadID, 0);

    //DebugUtil<float>::printNetworkEdges(stdout, "YOLO gogo", networkID, 0);

    Job* pubJob = getPubJob(job);

    if ((baseNetworkType == (int)WORKER_OD_eSSD) || 
        (baseNetworkType == (int)WORKER_OD_eFRCNN)) {
        // for SSD, frcnn

        int count = commonOutputLayer->_outputData[0]->getCount();
        const float* result = commonOutputLayer->_outputData[0]->host_data();
        int resultCount = 0;

        for (int i = 0; i < count; i += 7) {
            resultCount++;
        }
        pubJob->addJobElem(Job::IntType, 1, (void*)&resultCount);

        float left, top, right, bottom;
        for (int i = 0; i < resultCount; i++) {
            if (baseNetworkType == 0) {     // SSD, 여기서는 무조건 절대좌표로 변환한다.
                left	= std::min(std::max(result[i * 7 + 3], 0.f), 1.f);
                top		= std::min(std::max(result[i * 7 + 4], 0.f), 1.f);
                right	= std::min(std::max(result[i * 7 + 5], 0.f), 1.f);
                bottom	= std::min(std::max(result[i * 7 + 6], 0.f), 1.f);

                left    = int(left * width);
                top     = int(top * height);
                right   = int(right * width);
                bottom  = int(bottom * height);
            } else {        // FRCNN case
                left	= int(result[i * 7 + 3]);
                top		= int(result[i * 7 + 4]);
                right	= int(result[i * 7 + 5]);
                bottom	= int(result[i * 7 + 6]);
            }

            float score = result[i * 7 + 2];
            int labelIndex = (int)(result[i * 7 + 1] + 0.000001);

            pubJob->addJobElem(Job::FloatType, 1, (void*)&top);
            pubJob->addJobElem(Job::FloatType, 1, (void*)&left);
            pubJob->addJobElem(Job::FloatType, 1, (void*)&bottom);
            pubJob->addJobElem(Job::FloatType, 1, (void*)&right);
            pubJob->addJobElem(Job::FloatType, 1, (void*)&score);
            pubJob->addJobElem(Job::IntType, 1, (void*)&labelIndex);
        }
    } else {
        SASSERT0(baseNetworkType == (int)WORKER_OD_eYOLO);     // YOLO case
        const float* result = commonOutputLayer->_outputData[0]->host_data();

        int gridCount = YOLO_GRID_COUNT;
        int gridAxisCount = YOLO_GRID_ONE_AXIS_COUNT;
        int elemCountPerGrid = YOLO_GRID_ELEM_COUNT;
        int anchorBoxCount = YOLO_ANCHOR_BOX_COUNT;
        int classCount = YOLO_CLASS_COUNT;
        int imageWidth = YOLO_IMAGE_DEFAULT_WIDTH;
        int imageHeight = YOLO_IMAGE_DEFAULT_HEIGHT;
        float confThres = YOLO_DEFAULT_CONFIDENCE_THRES;

        int resultCount = 0;
        float left, top, right, bottom;

        vector<yoloJobPack> yoloPacks;

        for (int i = 0; i < gridCount; i++) {
            int gridX = i % gridAxisCount;
            int gridY = i / gridAxisCount;

            for (int j = 0; j < anchorBoxCount; j++) {
                int resultBaseIndex = i * elemCountPerGrid + j * (classCount + 5);
                float x = result[resultBaseIndex + 0];
                float y = result[resultBaseIndex + 1];
                float w = result[resultBaseIndex + 2];
                float h = result[resultBaseIndex + 3];
                float c = result[resultBaseIndex + 4];

                float maxClassConfidence = result[resultBaseIndex + 5];
                int maxClassIdx = 0;

                for (int classIdx = 1; classIdx < classCount; classIdx++) {
                    if (maxClassConfidence < result[resultBaseIndex + 5 + classIdx]) {
                        maxClassIdx = classIdx;
                        maxClassConfidence = result[resultBaseIndex + 5 + classIdx];
                    }
                }

                float score = c * maxClassConfidence;

                if (score <= confThres) {
                    continue; 
                }
                resultCount++;

                top = (float)((((float)gridY + y) / (float)gridAxisCount - 0.5 * h) * 
                    (float)imageHeight);
                bottom = (float)((((float)gridY + y) / (float)gridAxisCount + 0.5 * h) * 
                    (float)imageHeight);
                left = (float)((((float)gridX + x) / (float)gridAxisCount - 0.5 * w) * 
                    (float)imageWidth);
                right = (float)((((float)gridX + x) / (float)gridAxisCount + 0.5 * w) * 
                    (float)imageWidth);

                yoloJobPack yoloPack;
                yoloPack.top = top;
                yoloPack.left = left;
                yoloPack.bottom = bottom;
                yoloPack.right = right;
                yoloPack.score = score;
                yoloPack.labelIndex = maxClassIdx;

                yoloPacks.push_back(yoloPack);
            }
        }

        // NMS 적용
        int nmsThres = YOLO_DEFAULT_NMS_THRES;
        vector<yoloJobPack> nmsYoloPacks;

        for (int labelIdx = 0; labelIdx < classCount; labelIdx++) {
            vector<uint32_t> keep;
            vector<vector<float>> proposals;
            vector<float> scores;

            for (int i = 0; i < resultCount; i++) {
                if (labelIdx != yoloPacks[i].labelIndex)
                    continue;

                vector<float> newProposal = {yoloPacks[i].left, yoloPacks[i].top, 
                    yoloPacks[i].right, yoloPacks[i].bottom};
                proposals.push_back(newProposal);
                scores.push_back(yoloPacks[i].score);
            }

            if (proposals.size() == 0)
                continue;

            ImageUtil<float>::nms(proposals, scores, nmsThres, keep);

            for (int i = 0; i < keep.size(); i++) {
                yoloJobPack nmsYoloPack;
                nmsYoloPack.top = proposals[keep[i]][1];
                nmsYoloPack.left = proposals[keep[i]][0];
                nmsYoloPack.bottom = proposals[keep[i]][3];
                nmsYoloPack.right = proposals[keep[i]][2];
                nmsYoloPack.score = scores[keep[i]];
                nmsYoloPack.labelIndex = labelIdx;

                nmsYoloPacks.push_back(nmsYoloPack);
            }
        }

        resultCount = nmsYoloPacks.size();
        pubJob->addJobElem(Job::IntType, 1, (void*)&resultCount);

        for (int i = 0; i < resultCount; i++) {
            pubJob->addJobElem(Job::FloatType, 1, (void*)&nmsYoloPacks[i].top);
            pubJob->addJobElem(Job::FloatType, 1, (void*)&nmsYoloPacks[i].left);
            pubJob->addJobElem(Job::FloatType, 1, (void*)&nmsYoloPacks[i].bottom);
            pubJob->addJobElem(Job::FloatType, 1, (void*)&nmsYoloPacks[i].right);
            pubJob->addJobElem(Job::FloatType, 1, (void*)&nmsYoloPacks[i].score);
            pubJob->addJobElem(Job::IntType, 1, (void*)&nmsYoloPacks[i].labelIndex);
        }
    }

    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleRunClassificationNetworkWithInput(Job* job) {
    string networkID = job->getStringValue(0);
    int channel = job->getIntValue(1);
    int height = job->getIntValue(2);
    int width = job->getIntValue(3);
    int baseNetworkType = job->getIntValue(4);
    float* imageData = job->getFloatArray(5);

    Network<float>* network = Network<float>::getNetworkFromID(networkID);
    WorkContext::updateNetwork(networkID);
   
    InputLayer<float>* commonInputLayer;
    Layer<float>* commonOutputLayer;

    SASSUME0(baseNetworkType < (int)WORKER_IC_eMAX);

    commonInputLayer = (InputLayer<float>*)network->findLayer(SNPROP(inputLayer));
    SASSUME0(commonInputLayer != NULL); 
    commonOutputLayer = network->findLayer(SNPROP(outputLayer));
    SASSUME0(commonOutputLayer != NULL); 
    WorkContext::updateLayer(networkID, commonInputLayer->layerID);
    commonInputLayer->feedImage(channel, height, width, imageData);

    network->runMiniBatch(true, 0);
    ThreadMgmt::wait(WorkContext::curThreadID, 0);

    Job* pubJob = getPubJob(job);

    const float* result = commonOutputLayer->_outputData[0]->host_data();
    int count = commonOutputLayer->_outputData[0]->getCount();

    // find argument index that has maximum value and return it
    int maxArgIndex = 0;
    int maxValue = result[0];

    for (int i = 1; i < count; i++) {
        if (result[i] > maxValue) {
            maxValue = result[i];
            maxArgIndex = i;
        }
    }

    pubJob->addJobElem(Job::IntType, 1, (void*)&maxArgIndex);
    Broker::publish(job->getJobID(), pubJob);
}

void Worker::handleStartIDP(Job* job) {
    string networkID = job->getStringValue(0);
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

        case JobType::RunObjectDetectionNetworkWithInput:
            handleRunObjectDetectionNetworkWithInput(job);
            break;

        case JobType::RunClassificationNetworkWithInput:
            handleRunClassificationNetworkWithInput(job);
            break;

        case JobType::StartInputDataProvider:
            handleStartIDP(job);
            break;

        case JobType::GetMeasureItemName:
            handleGetMeasureItemName(job);
            break;

        case JobType::GetMeasures:
            handleGetMeasures(job);
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
    Worker::producer = NULL;
    SNEW_ONCE(Worker::producer, thread, producerThread);
    SASSUME0(producerThread != NULL);

	// (4) consumer 쓰레드들을 생성한다.
	for (int i = 0; i < SPARAM(GPU_COUNT); i++) {
		Worker::consumers.push_back(thread(taskConsumerThread, i, Cuda::availableGPU[i]));
        TaskQueue *tq = NULL;
        SNEW_ONCE(tq, TaskQueue);
        SASSUME0(tq != NULL);
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
	SDELETE(Worker::producer);
    Worker::producer = NULL;

    for (int i = 0; i < Worker::taskQueues.size(); i++) {
        SDELETE(Worker::taskQueues[i]);
    }

    Worker::taskQueues.clear();
}

int Worker::pushJob(Job* job) {
    int pubJobID = -1;

    // (1) pubJob이 있는 경우에 pubJob을 생성하고 pubJob ID를 할당받는다.
    if (job->hasPubJob()) {
        Job* pubJob = NULL;
        SNEW(pubJob, Job, job->getPubJobType());
        SASSUME0(pubJob != NULL);

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

void Worker::addRunPlanTask(int consumerIdx, string networkID, int dopID, bool inference,
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

void Worker::addUpdateTensorTask(int consumerIdx, string networkID, int dopID, int layerID,
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

void Worker::addAllocLayerTask(int consumerIdx, string networkID, int dopID, int layerID,
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
