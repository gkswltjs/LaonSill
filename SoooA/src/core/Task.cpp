/**
 * @file Task.cpp
 * @date 2017-06-14
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "Task.h"
#include "Param.h"
#include "SysLog.h"
#include "ColdLog.h"

using namespace std;

vector<TaskPool*> Task::taskPools;

void Task::allocTaskPool(TaskType taskType) {
    TaskPool *tp = new TaskPool();
    SASSERT0(tp != NULL);
    tp->taskType = taskType;

    int elemCount;
    switch(taskType) {
        case TaskType::AllocTensor:
            elemCount = SPARAM(TASKPOOL_ALLOCTENSOR_ELEM_COUNT);
            break;

        case TaskType::UpdateTensor:
            elemCount = SPARAM(TASKPOOL_UPDATETENSOR_ELEM_COUNT);
            break;

        case TaskType::RunPlan:
            elemCount = SPARAM(TASKPOOL_RUNPLAN_ELEM_COUNT);
            break;

        case TaskType::AllocLayer:
            elemCount = SPARAM(TASKPOOL_ALLOCLAYER_ELEM_COUNT);
            break;

        default:
            SASSERT0(false);
    }

    for (int i = 0; i < elemCount; i++) {
        void *elem;
        switch (taskType) {
            case TaskType::AllocTensor:
                elem = (void*)(new TaskAllocTensor());
                break;
            
            case TaskType::UpdateTensor:
                elem = (void*)(new TaskUpdateTensor());
                break;

            case TaskType::RunPlan:
                elem = (void*)(new TaskRunPlan());
                break;

            case TaskType::AllocLayer:
                elem = (void*)(new TaskAllocLayer());
                break;

            default:
                SASSERT0(false);
        }
        SASSERT0(elem != NULL);
       
        TaskBase* tb = (TaskBase*)elem;
        tb->elemID = i;
        tb->taskType = taskType;

        tp->alloc.push_back(elem);
        tp->freeElemIDList.push_back(i);
    }
    taskPools.push_back(tp); 
}

void Task::init() {
    for (int i = 0; i < TaskType::TaskTypeMax; i++) {
        allocTaskPool((TaskType)i);
    }
}

void Task::releaseTaskPool(TaskType taskType) {
    int elemCount; 
    for (int i = 0; i < taskPools[taskType]->alloc.size(); i++) {
        switch(taskType) {
            case TaskType::AllocTensor:
                delete (TaskAllocTensor*)taskPools[taskType]->alloc[i];
                break;

            case TaskType::UpdateTensor:
                delete (TaskUpdateTensor*)taskPools[taskType]->alloc[i];
                break;

            case TaskType::RunPlan:
                delete (TaskRunPlan*)taskPools[taskType]->alloc[i];
                break;

            case TaskType::AllocLayer:
                delete (TaskAllocLayer*)taskPools[taskType]->alloc[i];
                break;

            default:
                SASSERT0(false);
        }
    }
    taskPools[taskType]->alloc.clear();
    delete taskPools[taskType];
}

void Task::destroy() {
    for (int i = 0; i < TaskType::TaskTypeMax; i++) {
        releaseTaskPool((TaskType)i);
    }
    taskPools.clear();
}

void* Task::getElem(TaskType taskType) {
    SASSUME0(taskType < TaskType::TaskTypeMax);
    TaskPool *tp = taskPools[taskType];
    unique_lock<mutex> lock(tp->mutex);

    if (tp->freeElemIDList.empty()) {
        lock.unlock();
        COLD_LOG(ColdLog::WARNING, true,
            "there is no free elem for task pool. task pool type=%d", (int)taskType);
        return NULL;
    }

    int elemID = tp->freeElemIDList.front();
    tp->freeElemIDList.pop_front();

    lock.unlock();

    return tp->alloc[elemID];
}

void Task::releaseElem(TaskType taskType, void* elemPtr) {
    SASSUME0(taskType < TaskType::TaskTypeMax);
    TaskPool *tp = taskPools[taskType];

    TaskBase* tb = (TaskBase*)elemPtr;
    int elemID = tb->elemID;

    unique_lock<mutex> lock(tp->mutex);
    tp->freeElemIDList.push_back(elemID);
}
