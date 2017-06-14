/**
 * @file Task.h
 * @date 2017-06-14
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef TASK_H
#define TASK_H 

#include <list>
#include <mutex>
#include <vector>
#include <string>

#include "Update.h"

typedef enum TaskType_e {
    AllocTensor = 0,
    UpdateTensor,
    RunPlan,
    TaskTypeMax
} TaskType;

typedef struct TaskPool_s {
    TaskType            taskType;
    std::vector<void*>  alloc;
    std::list<int>      freeElemIDList;
    std::mutex          mutex;
} TaskPool;

typedef struct TaskBase_s {
    int         elemID;
} TaskBase;

typedef struct TaskAllocTensor_s {
    int         elemID;
    int         nodeID;
    int         devID;
    int         requestThreadID; 
    std::string tensorName;
} TaskAllocTensor;

typedef struct TaskUpdateTensor_s {
    int                         elemID;
    int                         networkID;
    int                         dopID;
    int                         layerID;
    int                         planID;
    std::vector<UpdateParam>    updateParams;
} TaskUpdateTensor;

typedef struct TaskRunPlan_s {
    int     elemID;
    int     networkID;
    int     dopID;
} TaskRunPlan;

class Task {
public: 
    Task() {}
    virtual ~Task() {}

    static void init();
    static void destroy();

    static void* getElem(TaskType taskType);
    static void releaseElem(TaskType taskType, void* elemPtr);

private:
    static void allocTaskPool(TaskType taskType);
    static void releaseTaskPool(TaskType taskType);
    static std::vector<TaskPool*> taskPools;
};
#endif /* TASK_H */
