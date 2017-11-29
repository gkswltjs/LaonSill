/**
 * @file MeasureManager.cpp
 * @date 2017-11-02
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "MeasureManager.h"
#include "SysLog.h"
#include "Param.h"

using namespace std;

map<string, MeasureEntry*>     MeasureManager::entryMap;
mutex                       MeasureManager::entryMapMutex;

void MeasureManager::insertEntryEx(string networkID, vector<string> itemNames,
        MeasureOption option, int queueSize) {
    MeasureEntry* newEntry = new MeasureEntry(networkID, queueSize, option, itemNames);
    SASSERT0(newEntry != NULL);

    unique_lock<mutex> entryMapLock(MeasureManager::entryMapMutex);
    SASSERT0(MeasureManager::entryMap.find(networkID) == MeasureManager::entryMap.end());
    MeasureManager::entryMap[networkID] = newEntry;
}

void MeasureManager::insertEntry(string networkID, vector<string> itemNames) {
    insertEntryEx(networkID, itemNames, MEASURE_OPTION_DEFAULT, 
        SPARAM(MEASURE_QUEUE_DEFAULT_SIZE));
}

void MeasureManager::removeEntry(string networkID) {
    unique_lock<mutex> entryMapLock(MeasureManager::entryMapMutex);
    SASSERT0(MeasureManager::entryMap.find(networkID) != MeasureManager::entryMap.end());
    MeasureEntry* removeEntry = MeasureManager::entryMap[networkID];
    MeasureManager::entryMap.erase(MeasureManager::entryMap.find(networkID));
    delete removeEntry;
}

MeasureEntry* MeasureManager::getMeasureEntry(string networkID) {
    unique_lock<mutex> entryMapLock(MeasureManager::entryMapMutex);
    SASSERT0(MeasureManager::entryMap.find(networkID) != MeasureManager::entryMap.end());
    MeasureEntry* entry = MeasureManager::entryMap[networkID];
    return entry;
}
