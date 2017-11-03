/**
 * @file MeasureEntry.cpp
 * @date 2017-11-02
 * @author moonhoen lee
 * @brief 
 *  정확도, 에러율과 같은 관측가능한 수치들을 관리하는 모듈이다.
 * @details
 *  옵션으로 file 관리와 memory 관리를 지원하는데, 현재는 memory 관리만 지원하고 있다.
 *  추후에 수정할 예정이다.
 */

#include "MeasureEntry.h"
#include "SysLog.h"
#include "StdOutLog.h"

using namespace std;

#define MEASURE_ENTRY_ADDDATA_MAX_RETRY_COUNT   10
#define MEASURE_ENTRY_RETRY_MSEC                10UL

MeasureEntry::MeasureEntry(int networkID, int queueSize, MeasureOption option,
    vector<string> itemNames) {
    this->networkID = networkID;
    this->queueSize = queueSize;
    this->itemNames = itemNames;
    this->itemCount = this->itemNames.size();
    SASSERT0(this->itemCount > 0);

    SASSERT0(option == MEASURE_OPTION_MEMORY);  // XXX: 임시적인 제한

    this->head = 0;
    this->tail = 0;
    this->option = option;

    this->freeCount = this->queueSize;
    this->baseIterNum = 0;

    this->data = (float*)malloc(sizeof(float) * queueSize * this->itemCount); 
    SASSERT0(this->data != NULL);

    this->status =
        (MeasureEntryDataStatus*)malloc(sizeof(MeasureEntryDataStatus) * this->queueSize);
    SASSERT0(this->status != NULL);
    for (int i = 0; i < this->queueSize; i++)
        this->status[i] = MEASURE_ENTRY_STATUS_NONE;

    this->readRefCount = (int*)malloc(sizeof(int) * this->queueSize);
    SASSERT0(this->readRefCount != NULL);
    for (int i = 0; i < this->queueSize; i++)
        this->readRefCount[i] = 0;

    this->addBuffer = (float*)malloc(sizeof(float) * this->itemCount);
    SASSERT0(this->addBuffer != NULL);
}

MeasureEntry::~MeasureEntry() {
    SASSERT0(this->data != NULL);
    free((void*)this->data);

    SASSERT0(this->status != NULL);
    free((void*)this->status);

    SASSERT0(this->readRefCount != NULL);
    free((void*)this->readRefCount);
}

void MeasureEntry::addData(float* data) {
    int dataOffset;

    // write access는 언제나 한명이 하나의 원소에 대해서만 수행하게 될 것이다.
    // 아니면 아키텍쳐를 수정해야 한다.
    dataOffset = this->head * this->itemCount;

    // FIXME: Mutex로 자원보호를 하는 것보다는 AOP로 하는 것이 성능상 더 좋아보니다.
    bool doLoop = true;
    int retryCount = 0;
    unique_lock<mutex> entryLock(this->entryMutex);
    while (doLoop) {
        MeasureEntryDataStatus curStatus = this->status[this->head];

        if (curStatus == MEASURE_ENTRY_STATUS_WRITE) {
            entryLock.unlock();
            SASSERT0(false);
        } else if (curStatus == MEASURE_ENTRY_STATUS_READ) {
            entryLock.unlock();
            retryCount++;

            SASSERT(retryCount < MEASURE_ENTRY_ADDDATA_MAX_RETRY_COUNT,
                "can not add data in the measure entry(network ID=%d)", this->networkID);

            usleep(MEASURE_ENTRY_RETRY_MSEC);
            entryLock.lock();
        } else {        // MEASURE_ENTRY_STATUS_NONE case
            this->status[this->head] = MEASURE_ENTRY_STATUS_WRITE;
            entryLock.unlock();
            doLoop = false;
        }
    }

    memcpy((void*)&this->data[dataOffset], data, sizeof(float) * this->itemCount);

    entryLock.lock();
    this->status[this->head] = MEASURE_ENTRY_STATUS_NONE;
    this->head = this->head + 1;
    if (this->head == this->queueSize) {
        this->head = 0;
        this->baseIterNum = this->baseIterNum + this->queueSize;
    }
    entryLock.unlock();
}

void MeasureEntry::getDataInternal(int start, int count, float* data) {
    int dataOffset;
    dataOffset = start * this->itemCount;

    SASSUME0(count > 0);
    SASSUME0(start < this->queueSize);
    SASSUME0(start + count <= this->queueSize);
    SASSUME0(data != NULL);

    // FIXME: Mutex로 자원보호를 하는 것보다는 AOP로 하는 것이 성능상 더 좋아보니다.
    unique_lock<mutex> entryLock(this->entryMutex);

    for (int i = 0; i < count; i++) {
        int index = start + i;
        int retryCount = 0;

    // 복잡한 조건의 loop에서 goto를 통한 예외처리는 좋은 방법이 될 수 있다.
    // (혹시 이 소스를 보고 비판할까봐.....)
retry:
        MeasureEntryDataStatus curStatus = this->status[index];
        
        if (curStatus == MEASURE_ENTRY_STATUS_READ) {
            SASSUME0(this->readRefCount[index] > 0);
            this->readRefCount[index] = this->readRefCount[index] + 1;
        } 
        
        else if (curStatus == MEASURE_ENTRY_STATUS_WRITE) {
            entryLock.unlock();
            retryCount++;

            SASSERT(retryCount < MEASURE_ENTRY_ADDDATA_MAX_RETRY_COUNT,
                "can not get data in the measure entry(network ID=%d)", 
                this->networkID);

            usleep(MEASURE_ENTRY_RETRY_MSEC);
            entryLock.lock();

            goto retry;

        } else {        // MEASURE_ENTRY_STATUS_NONE case
            this->status[index] = MEASURE_ENTRY_STATUS_READ;
            SASSUME0(this->readRefCount[index] == 0);
            this->readRefCount[index] = 1;
        }
    }
    entryLock.unlock();

    // FIXME: 하다보니 실제 필요한 동작은 메모리 카피 하나밖에 없는데 너무 동기화 관련 코드가 
    // 많아졌다. 좀 더 나은 코드로 대체하자.
    memcpy((void*)data, (void*)&this->data[dataOffset],
        sizeof(float) * this->itemCount * count);

    entryLock.lock();

    for (int i = 0; i < count; i++) {
        int index = start + i;
        int retryCount = 0;

        this->readRefCount[index] = this->readRefCount[index] - 1;
        if (this->readRefCount[index] == 0) {
            this->status[index] = MEASURE_ENTRY_STATUS_NONE;
        }
    }
    entryLock.unlock();
}

void MeasureEntry::getData(int start, int count, bool forward, float* data) {
    int start1, start2;
    int count1, count2;

    if (forward) {
        start1 = start;

        if (this->queueSize - start < count) {
            count1 = this->queueSize - start;
            count2 = count - count1;
            start2 = 0;
        } else {
            count1 = count;
            count2 = 0;
            start2 = -1;    // meaning less
        }
    } else {
        if (this->head < count) {
            start2 = 0;
            count2 = this->head;
            count1 = count2 - count1;
            start1 = this->queueSize - count1;
        } else {
            start1 = this->head - count;
            count1 = count;
            start2 = -1;    //meaningless
            count2 = 0;
        }
    }

    getDataInternal(start1, count1, data); 

    if (count2 > 0) {
        int offset = count1 * this->itemCount;
        getDataInternal(start2, count2, (float*)&data[offset]);
    }
}

void MeasureEntry::printStatus() {
    STDOUT_LOG("[Measure Entry Info]");
    STDOUT_LOG("  - networkID : %d", this->networkID);
    STDOUT_LOG("  - option : %d", int(this->option));
    STDOUT_LOG("  - queue size : %d", this->queueSize);

    STDOUT_LOG("  - item count : %d", this->itemCount);
    STDOUT_LOG("  - item names :");
    for (int i = 0 ; i < this->itemNames.size(); i++) {
        STDOUT_LOG("      > %s", this->itemNames[i].c_str());
    }
    STDOUT_LOG("  - head : %d", this->head);
    STDOUT_LOG("  - tail : %d", this->tail);
    STDOUT_LOG("  - base iter num : %d", this->baseIterNum);
}
