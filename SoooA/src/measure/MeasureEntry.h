/**
 * @file MeasureEntry.h
 * @date 2017-11-02
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef MEASUREENTRY_H
#define MEASUREENTRY_H 

#include <vector>
#include <string>
#include <mutex>

typedef enum MeasureOption_e {
    MEASURE_OPTION_MEMORY = 1,
    MEASURE_OPTION_FILE = 2
} MeasureOption;

typedef enum MeasureEntryDataStutus_e : int{
    MEASURE_ENTRY_STATUS_NONE = 0,
    MEASURE_ENTRY_STATUS_WRITE,
    MEASURE_ENTRY_STATUS_READ
} MeasureEntryDataStatus;

// XXX: 현재는 default option만 사용하고 있음.
const MeasureOption MEASURE_OPTION_DEFAULT = MEASURE_OPTION_MEMORY;
//const int MEASURE_DEFAULT_QUEUE_SIZE = 10000;
const int MEASURE_DEFAULT_QUEUE_SIZE = 500;

class MeasureEntry {

public: 
    MeasureEntry(int networkID, int queueSize, MeasureOption option,
        std::vector<std::string> itemNames);
    virtual ~MeasureEntry();

    void addData(float* data);

    void getDataInternal(int start, int count, float* data); 
    void getData(int start, int count, bool forward, int* startIterNum, int* measureCount,
            float* data);

    float* getAddBuffer() { return this->addBuffer; }
    std::vector<std::string> getItemNames() { return this->itemNames; }
    void printStatus();

private:
    std::vector<std::string>    itemNames;
    int                         itemCount;
    int                         networkID;
    MeasureOption               option;
    int                         queueSize;
    volatile int                head;           // entry insert position
    int                         tail;           // file flush position
    int                         freeCount;      // free entry count
                                                // MEMORY 옵션만 존재한 경우에는 의미 없음.
    int                         baseIterNum;    // 큐의 첫번째 자리에 해당하는 iteration
                                                // number. 큐에 원소를 다 채우게 된뎌
                                                // 이 값은 queueSize만큼 늘어나게 된다.
    std::mutex                          entryMutex;     // head, tail, lastIterNum을 보호
    volatile float*                     data;
    volatile MeasureEntryDataStatus*    status;         // RW data access status
    volatile int*                       readRefCount;   // read reference count

    float*                      addBuffer;      // for convenience
    void setAreaLock(int start, int count);
};
#endif /* MEASUREENTRY_H */
