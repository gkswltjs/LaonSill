/**
 * @file InputDataProvider.cpp
 * @date 2017-07-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "InputDataProvider.h"
#include "SysLog.h"
#include "Param.h"
#include "WorkContext.h"
#include "ColdLog.h"
#include "ThreadMgmt.h"
#include "PhysicalPlan.h"

#include "Datum.h"
#include "DataReader.h"
#include "Param.h"

using namespace std;

mutex                           InputDataProvider::poolMutex;
map<InputPoolKey, InputPool*>   InputDataProvider::pools;
map<int, PoolInfo>              InputDataProvider::poolInfoMap;
map<DRType, DRCBFuncs>          InputDataProvider::drFuncMap;

// FIXME: turn in into an auto-generated function
void InputDataProvider::init() {
    // (1) add Datum CB funcs
    DRCBFuncs funcDatum;
    funcDatum.allocFunc     = DataReader<class Datum>::allocElem;
    funcDatum.deallocFunc   = DataReader<class Datum>::deallocElem;
    funcDatum.fillFunc      = DataReader<class Datum>::fillElem;
    drFuncMap[DRType::Datum] = funcDatum;
}

// input layer에서 이 함수를 호출해서 pool을 등록해야 한다.
void InputDataProvider::addPool(DRType drType, void* reader) {
    int networkID = WorkContext::curNetworkID;
    int dopCount = PhysicalPlan::getDOPCount(networkID);

    PoolInfo poolInfo;
    poolInfo.networkID = networkID;
    poolInfo.dopCount = dopCount;
    poolInfo.reader = reader;
    poolInfo.drType = drType;
    poolInfo.threadID = -1;
    poolInfo.cleanupThreadID = -1;

    SASSERT0(drFuncMap.find(drType) != drFuncMap.end());
    DRCBFuncs funcs = drFuncMap[drType];

    unique_lock<mutex> poolLock(poolMutex);
    if (poolInfoMap.find(networkID) != poolInfoMap.end()) {
        // 이미 등록이 되어 있다.
        return;
    }

    poolInfoMap[networkID] = poolInfo;
    poolLock.unlock();

    for (int i = 0; i < dopCount; i++) {
        InputPoolKey poolKey;
        poolKey.networkID = networkID;
        poolKey.dopID = i;

        InputPool* newPool = new InputPool();
        newPool->head = 0;
        newPool->tail = 0;
        newPool->remainElemCnt = SPARAM(INPUT_DATA_PROVIDER_ELEM_COUNT);
        newPool->elemCnt = SPARAM(INPUT_DATA_PROVIDER_ELEM_COUNT);
        newPool->activeElemCnt = 0;
        for (int j = 0; j < newPool->remainElemCnt; j++) {
            void* elemPtr;
            funcs.allocFunc(&elemPtr);
            newPool->elemArray.push_back(elemPtr);
        }

        poolLock.lock();
        SASSERT0(pools.find(poolKey) == pools.end());
        pools[poolKey] = newPool;
        poolLock.unlock();
    }
}

// 이 함수는 네트워크 destory시에 호출을 해야 한다.
void InputDataProvider::removePool(int networkID) {
    int dopCount;
    unique_lock<mutex> poolLock(poolMutex);
    SASSERT0(poolInfoMap.find(networkID) != poolInfoMap.end());
    dopCount = poolInfoMap[networkID].dopCount;
    DRType drType = poolInfoMap[networkID].drType;
    poolInfoMap[networkID].cleanupThreadID = WorkContext::curThreadID;
    poolLock.unlock();

    ThreadMgmt::signal(poolInfoMap[networkID].threadID, ThreadEvent::FinishJob);
    ThreadMgmt::wait(WorkContext::curThreadID, 0UL);

    SASSERT0(drFuncMap.find(drType) != drFuncMap.end());
    DRCBFuncs funcs = drFuncMap[drType];

    for (int i = 0; i < dopCount; i++) {
        InputPoolKey poolKey;
        poolKey.networkID = networkID;
        poolKey.dopID = i;

        vector<int> waitingList;

        poolLock.lock();
        SASSERT0(pools.find(poolKey) != pools.end());
        InputPool* removingPool = pools[poolKey];
        SASSERT0(removingPool->waitingList.size() == 0);
        pools.erase(poolKey);
        poolLock.unlock();

        vector<void*>::iterator iter = removingPool->elemArray.begin();
        while (iter != removingPool->elemArray.end()) {
            void* removingElem = (*iter);
            funcs.deallocFunc(removingElem);
            iter = removingPool->elemArray.erase(iter);
        }

        delete removingPool;
    }

    poolLock.lock();
    SASSERT0(poolInfoMap.find(networkID) != poolInfoMap.end());
    poolInfoMap.erase(networkID);
    poolLock.unlock();
}

void* InputDataProvider::getData(int networkID, int dopID) {
    InputPoolKey poolKey;
    poolKey.networkID = networkID;
    poolKey.dopID = dopID;

    SASSUME0(SPARAM(USE_INPUT_DATA_PROVIDER));

    // FIXME: taking pool lock every time you get data is too costly. 
    unique_lock<mutex> poolLock(poolMutex);
    SASSUME0(pools.find(poolKey) != pools.end());
    InputPool* pool = pools[poolKey];
    poolLock.unlock();

    void *data = NULL;
    while (true) {
        unique_lock<mutex> datumLock(pool->mutex);
        if (pool->activeElemCnt == 0) {
            if (SPARAM(INPUT_DATA_PROVIDER_BLOCKING)) {
                pool->waitingList.push_back(WorkContext::curThreadID);
            }
            datumLock.unlock();

            COLD_LOG(ColdLog::WARNING, true, "The available data pool is empty.");

            if (!SPARAM(INPUT_DATA_PROVIDER_BLOCKING))
                return NULL;

            ThreadMgmt::wait(WorkContext::curThreadID,
                SPARAM(INPUT_DATA_PROVIDER_CALLER_WAIT_TIME_MS));
        } else {
            data = pool->elemArray[pool->head];
            pool->head = (pool->head + 1) % pool->elemCnt;
            pool->activeElemCnt -= 1;
            pool->remainElemCnt += 1;
            datumLock.unlock();

            break; 
        }
    }

    SASSUME0(data != NULL);
    return data;
}

void InputDataProvider::handleIDP(int networkID) {
    int timeout = SPARAM(INPUT_DATA_PROVIDER_WAIT_TIME_MS);
    handler(networkID);

    while (true) {
        ThreadEvent event = ThreadMgmt::wait(WorkContext::curThreadID, timeout); 
        if (event & Halt) {
            // 이러한 상황에서.. 메모리는 누가 해제해야 하나? 
            // 어차피 프로세스 종료니까 heap에 올라간 메모리 해제에 대해서 
            // 신경쓰지 않아도 되긴 하는데..
            // 그래도 깔끔하게 처리할지 말지 고민중.
            break;
        } else if (event & FinishJob) {
            int threadID;
            unique_lock<mutex> poolLock(poolMutex);
            SASSERT0(poolInfoMap.find(networkID) != poolInfoMap.end());
            threadID = poolInfoMap[networkID].cleanupThreadID;
            poolLock.unlock();

            SASSERT0(threadID != -1);
            ThreadMgmt::signal(threadID, ThreadEvent::Wakeup);
            break;
        }

        handler(networkID);
    }
}

// handler()의 종료조건이 명확하지 않다. 현재는 계속 일을하다가 network destroy 시에 종료가
// 되는 형태이다.
void InputDataProvider::handler(int networkID) {
    unique_lock<mutex> poolLock(poolMutex);
    if (poolInfoMap.find(networkID) == poolInfoMap.end()) {
        // input data provider job이 시작이 되기 전에 사용자가 네트워크를 destroy 하였을
        // 경우에 발생할 수 있다. 이걸 예외처리해야 할지 아니면 허용해야 할지 고민..
        COLD_LOG(ColdLog::WARNING, true,
            "The specified network has been destroyed or has not yet been registered."
            " networkID=%d", networkID);
    }
    PoolInfo poolInfo = poolInfoMap[networkID];
    DRType drType = poolInfo.drType;
    SASSERT0(poolInfo.threadID == -1);
    poolInfo.threadID = WorkContext::curThreadID;
    poolLock.unlock();

    SASSERT0(drFuncMap.find(drType) != drFuncMap.end());
    DRCBFuncs funcs = drFuncMap[drType];

    while (true) {
        bool hasProgress = false;

        for (int i = 0; i < poolInfo.dopCount; i++) {
            InputPoolKey poolKey;
            poolKey.networkID = networkID;
            poolKey.dopID = i;

            int remainElemCnt;

            poolLock.lock();
            SASSUME0(pools.find(poolKey) != pools.end());
            InputPool* pool = pools[poolKey];
            poolLock.unlock();
           
            unique_lock<mutex> datumPoolLock(pool->mutex);
            remainElemCnt = pool->remainElemCnt;
            datumPoolLock.unlock();

            if (remainElemCnt == 0)
                continue;

            hasProgress = true;

            for (int j = 0; j < remainElemCnt; j++) {
                int elemIndex = (pool->tail + j) % pool->elemCnt;
                void* elemPtr = pool->elemArray[elemIndex];
                funcs.fillFunc(poolInfo.reader, elemPtr);
            }

            vector<int> waitingList;

            datumPoolLock.lock();
            pool->remainElemCnt -= remainElemCnt;
            pool->activeElemCnt += remainElemCnt;
            pool->tail = (pool->tail + remainElemCnt) % pool->elemCnt;
            for (int j = 0; j < pool->waitingList.size(); j++) {
                waitingList.push_back(pool->waitingList[j]);
            }
            pool->waitingList.clear();
            datumPoolLock.unlock();

            for (int j = 0; j < waitingList.size(); j++) {
                ThreadMgmt::signal(waitingList[j], ThreadEvent::Wakeup);
            }
        }

        if (!hasProgress)
            break;
    }
}
