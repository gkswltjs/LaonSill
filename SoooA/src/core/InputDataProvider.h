/**
 * @file InputDataProvider.h
 * @date 2017-07-10
 * @author moonhoen lee
 * @brief 
 * @details
 *   현재는 1개의 input layer가 존재한다는 가정아래 설계하였습니다.
 */

#ifndef INPUTDATAPROVIDER_H
#define INPUTDATAPROVIDER_H 

#include <mutex>
#include <vector>
#include <map>

#include "Datum.h"

// Data Reader Type
typedef enum DRType_e {
    Datum = 0,
    DRTypeMax
} DRType;

typedef void(*CBAllocDRElem)(void** elemPtr);
typedef void(*CBDeallocDRElem)(void* elemPtr);
typedef void(*CBFillDRElem)(void* reader, void* elemPtr);

typedef struct DRCBFuncs_s {
    CBAllocDRElem       allocFunc;
    CBDeallocDRElem     deallocFunc;
    CBFillDRElem        fillFunc;
} DRCBFuncs;

typedef struct InputPool_s {
    int                     head;
    int                     tail;
    int                     elemCnt;
    int                     remainElemCnt;
    int                     activeElemCnt;
    std::mutex              mutex;
    std::vector<int>        waitingList;
    std::vector<void*>      elemArray;
} InputPool;


typedef struct InputPoolKey_s {
    int         networkID;
    int         dopID;

    bool operator < (const struct InputPoolKey_s &x) const {
        if (networkID == x.networkID) {
            return dopID < x.dopID;
        } else {
            return networkID < x.networkID;
        }
    }
} InputPoolKey;

typedef struct PoolInfo_s {
    int             networkID;
    int             dopCount;
    volatile int    threadID;   // input data provider's thread ID (job consumer thread id)
    volatile int    cleanupThreadID;
    DRType          drType;
    void*           reader;
} PoolInfo;

class InputDataProvider {
public: 
    InputDataProvider() {}
    virtual ~InputDataProvider() {}

    static void init();

    static void addPool(DRType drType, void* reader);
    static void removePool(int networkID);

    // for input layer
    static void* getData(int networkID, int dopID);

    // for caller
    static void handleIDP(int networkID);

private:
    static std::mutex                           poolMutex;
    static std::map<InputPoolKey, InputPool*>   pools;
    static std::map<int, PoolInfo>              poolInfoMap;

    static std::map<DRType, DRCBFuncs>          drFuncMap;

    static void handler(int networkID);
};

#endif /* INPUTDATAPROVIDER_H */
