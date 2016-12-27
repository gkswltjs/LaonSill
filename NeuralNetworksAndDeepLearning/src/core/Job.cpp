/**
 * @file Job.cpp
 * @date 2016-12-15
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "common.h"
#include "Job.h"
#include "SysLog.h"
#include "ColdLog.h"

using namespace std;

atomic<int> Job::jobIDGen;

map<int, Job*> Job::reqPubJobMap;
mutex Job::reqPubJobMapMutex;

void Job::init() {
    atomic_store(&Job::jobIDGen, 0);
}

Job::Job(Job::JobType jobType, int jobElemCnt, Job::JobElemType *jobElemTypes,
    char *jobElemValues) {
    SASSERT0(jobType < Job::JobTypeMax);
    this->jobType = jobType;
    this->jobElemCnt = jobElemCnt;
 
    if (this->jobElemCnt > 0) {
        int allocSize = sizeof(JobElemDef) * this->jobElemCnt;
        this->jobElemDefs = (Job::JobElemDef*)malloc(allocSize);
        SASSERT0(this->jobElemDefs != NULL);
        SASSERT0(jobElemValues != NULL);
    } else {
        this->jobElemDefs = NULL;
    }

    int offset = 0;
    for (int i = 0; i < jobElemCnt; i++) {
        this->jobElemDefs[i].elemType = jobElemTypes[i];

        switch (jobElemTypes[i]) {
            case Job::IntType: 
                this->jobElemDefs[i].elemOffset = offset;
                this->jobElemDefs[i].arrayCount = 0;
                offset += sizeof(int);
                break;
            case Job::FloatType:
                this->jobElemDefs[i].elemOffset = offset;
                this->jobElemDefs[i].arrayCount = 0;
                offset += sizeof(float);
                break;
            case Job::FloatArrayType:
                offset += sizeof(int);
                this->jobElemDefs[i].elemOffset = offset;
                memcpy((void*)&this->jobElemDefs[i].arrayCount,
                    (void*)&jobElemTypes[offset], sizeof(int));
                offset += sizeof(float) * this->jobElemDefs[i].arrayCount;
                break;
            default:
                COLD_LOG(ColdLog::ERROR, true,
                    "invalid job elem type. request elem type=%d", jobElemTypes[i]);
                break;
        }
    }

    int jobElemValueAllocSize = offset;

    if (jobElemValueAllocSize > 0) {
        this->jobElemValues = (char*)malloc(jobElemValueAllocSize);
        SASSERT0(this->jobElemValues != NULL);
        memcpy((void*)this->jobElemValues, (void*)jobElemValues, jobElemValueAllocSize);
    } else {
        this->jobElemValues = NULL;
    }

    this->jobID = Job::genJobID();
    atomic_store(&this->taskIDGen, 1);
}

Job::Job(Job::JobType jobType) {
    // for incremental building usage
    this->jobType = jobType;
    this->jobElemCnt = 0;
    this->jobElemDefs = NULL;
    this->jobElemValues = NULL;
    this->jobID = Job::genJobID();
    atomic_store(&this->taskIDGen, 1);
}

Job::~Job() {
    if (this->jobElemDefs != NULL) {
        free(this->jobElemDefs);
    }
}

void Job::addJobElem(Job::JobElemType jobElemType, int arrayCount, void* dataPtr) {
    // realloc보다는 free & malloc 형태로 진행하였음.
    // 추후 메모리관리 할때에 free & malloc 형태로 구현할 계획을 가지고 있기 때문
    if (this->jobElemCnt == 0) {
        SASSERT0(this->jobElemDefs == NULL);
        SASSERT0(this->jobElemValues == NULL);
    } else {
        SASSERT0(this->jobElemDefs != NULL);
        SASSERT0(this->jobElemValues != NULL);
    }

    // for backup
    JobElemDef *tempJobElemDefs = this->jobElemDefs;
    char *tempJobElemValues = this->jobElemValues;
    int tempJobElemValueSize = this->getJobElemValueSize();

    this->jobElemCnt++;

    this->jobElemDefs = (Job::JobElemDef*)malloc(sizeof(Job::JobElemDef) * this->jobElemCnt);
    SASSERT0(this->jobElemDefs != NULL);

    int elemCnt = 1;
    if (jobElemType == Job::FloatArrayType) {
        SASSERT0(arrayCount > 0);
        elemCnt = arrayCount;
    }

    int elemSize;
    switch (jobElemType) {
        case Job::IntType:
            elemSize = sizeof(int);
            break;
        case Job::FloatType:
        case Job::FloatArrayType:
            elemSize = sizeof(float);
            break;
        default:
            SASSERT(false, "invalid job elem type. job elem type=%d", (int)jobElemType);
    }

    int elemValueSize = elemCnt * elemSize;

    if (jobElemType == Job::FloatArrayType) {
        // array size만큼을 고려해 줘야 한다.
        this->jobElemValues = (char*)malloc(tempJobElemValueSize + elemValueSize + sizeof(int));
    } else {
        this->jobElemValues = (char*)malloc(tempJobElemValueSize + elemValueSize);
    }
    SASSERT0(this->jobElemValues != NULL);

    // restore previous backup to new jobElemDefs & jobElemValues
    if (this->jobElemCnt > 1) {
        int copySize = sizeof(Job::JobElemDef) * (this->jobElemCnt - 1);
        memcpy((void*)this->jobElemDefs, (void*)tempJobElemDefs, copySize);
        free(tempJobElemDefs);

        memcpy((void*)this->jobElemValues, (void*)tempJobElemValues, tempJobElemValueSize);
        free(tempJobElemValues);
    }

    // fill job elem def
    int curJobElemIdx = this->jobElemCnt - 1;
    this->jobElemDefs[curJobElemIdx].elemType = jobElemType;
    this->jobElemDefs[curJobElemIdx].elemOffset = tempJobElemValueSize;

    if (jobElemType == Job::FloatArrayType)
        this->jobElemDefs[curJobElemIdx].elemOffset += sizeof(int);

    this->jobElemDefs[curJobElemIdx].arrayCount = arrayCount;

    // fill job elem value
    if (jobElemType == Job::FloatArrayType) {
        memcpy((void*)&this->jobElemValues[tempJobElemValueSize], (void*)&arrayCount,
            sizeof(int));
    }

    memcpy((void*)&this->jobElemValues[this->jobElemDefs[curJobElemIdx].elemOffset],
        (void*)dataPtr, elemValueSize);
}

// TODO: 에러 처리에 대해서 고민이 필요하다. 
//       에러 상황에 대해서는 ColdLog에 남을 것이다.
//       하지만 그 상황에 대해서 프로그램적으로 처리하기 위해서는 에러값을 반환 하거나
//       Exception을 던져야 한다.
//       이 케이스는 exception을 throw하는 것이 더 좋아보이는데.. 
//       이 부분에 대해서 논의를 하고, 그에 따른 구현을 수행하자.
//       지금은 assert로 걸어 두었다.
int Job::getIntValue(int elemIdx) {
    int     intVal;
    bool    isValid = isValidElemValue(Job::IntType, elemIdx);
    SASSUME0(isValid == true);

    int elemOffset = this->jobElemDefs[elemIdx].elemOffset;
    memcpy((void*)&intVal, (void*)&this->jobElemValues[elemOffset], sizeof(int));

    return intVal;
}

float Job::getFloatValue(int elemIdx) {
    float   floatVal;
    bool    isValid = isValidElemValue(Job::FloatType, elemIdx);
    SASSUME0(isValid == true);

    int elemOffset = this->jobElemDefs[elemIdx].elemOffset;
    memcpy((void*)&floatVal, (void*)&this->jobElemValues[elemOffset], sizeof(float));

    return floatVal;
}

float *Job::getFloatArray(int elemIdx) {
    bool isValid = isValidElemValue(Job::FloatArrayType, elemIdx);
    SASSUME0(isValid == true);

    int elemOffset = this->jobElemDefs[elemIdx].elemOffset;
    return (float*)(&this->jobElemValues[elemOffset]);
}

float Job::getFloatArrayValue(int elemIdx, int arrayIdx) {
    bool isValid = isValidElemArrayValue(Job::FloatArrayType, elemIdx, arrayIdx);
    SASSUME0(isValid == true);

    int elemOffset = this->jobElemDefs[elemIdx].elemOffset;
    float *floatArray = (float*)(&this->jobElemValues[elemOffset]);
    return floatArray[arrayIdx];
}

Job::JobElemDef Job::getJobElemDef(int elemIdx) {
    SASSUME0(this->isVaildElemIdx(elemIdx));

    return this->jobElemDefs[elemIdx];
}

int Job::getJobElemValueSize() {
    int totalSize = 0;

    for (int i = 0; i < this->jobElemCnt; i++) {
        switch (this->jobElemDefs[i].elemType) {
            case Job::IntType:
                totalSize += sizeof(int);
                break;

            case Job::FloatType:
                totalSize += sizeof(float);
                break;

            case Job::FloatArrayType:
                totalSize += sizeof(int);
                totalSize += sizeof(float) * this->jobElemDefs[i].arrayCount;
                break;

            default:
                COLD_LOG(ColdLog::ERROR, true,
                    "invalid job elem type. job type=%d, elem index=%d, elem type=%d",
                    this->jobType, i, this->jobElemDefs[i].elemType);
        }
    }

    return totalSize;
}

int Job::getJobSize() {
    // metasize = size of (jobID + jobtype + jobElemCnt + jobElemtTypes)
    int metaSize = sizeof(int) * (3 + this->jobElemCnt);

    // total size = metasize + size of jobelems
    int totalSize = metaSize + this->getJobElemValueSize();

    return totalSize;
}

bool Job::isVaildElemIdx(int elemIdx) {
    if (elemIdx >= this->jobElemCnt) {
        COLD_LOG(ColdLog::ERROR, true,
            "invalid request elem index. job type=%d, elem count=%d, elem index=%d",
            this->jobType, this->jobElemCnt, elemIdx);
        return false;
    }

    SASSUME0(this->jobElemDefs != NULL);

    return true;
}

bool Job::isValidElemValue(Job::JobElemType elemType, int elemIdx) {
    bool isVaildElemIdx = this->isVaildElemIdx(elemIdx);

    if (!isVaildElemIdx)
        return false;

    if (elemType != this->jobElemDefs[elemIdx].elemType) {
        COLD_LOG(ColdLog::ERROR, true, "invalid request elem type. "
            "job type=%d, elem type=%d, request elem type=%d, request elem index=%d",
            this->jobType, this->jobElemDefs[elemIdx].elemType, elemType, elemIdx);
        return false;
    }

    return true;
}

bool Job::isValidElemArrayValue(Job::JobElemType elemType, int elemIdx, int arrayIdx) {
    bool isValidElemValue = this->isValidElemValue(elemType, elemIdx);

    if (!isValidElemValue)
        return false;
    
    if (arrayIdx >= this->jobElemDefs[elemIdx].arrayCount) {
        COLD_LOG(ColdLog::ERROR, true, "invalid request array index. "
            "job type=%d, elem type=%d, request elem index=%d,"
            " array size=%d, request array index=%d",
            this->jobType, elemType, elemIdx, this->jobElemDefs[elemIdx].arrayCount,
            arrayIdx);
        return false;
    }

    return true;
}

int Job::genJobID() {
   return atomic_fetch_add(&Job::jobIDGen, 1); 
}

int Job::genTaskID() {
    return atomic_fetch_add(&this->taskIDGen, 1);
}

int Job::getJobID() {
    return this->jobID;
}

bool Job::hasPubJob() {
    if ((this->jobType == Job::CreateDQNImageLearner) ||
        (this->jobType == Job::StepDQNImageLearner))
        return true;

    return false;
}

Job::JobType Job::getPubJobType() {
    if (this->jobType == Job::CreateDQNImageLearner)
        return Job::CreateDQNImageLearnerResult;

    if (this->jobType == Job::StepDQNImageLearner)
        return Job::StepDQNImageLearnerResult;

    return Job::JobTypeMax;     // meaningless
}
