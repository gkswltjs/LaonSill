/**
 * @file MeasureManager.h
 * @date 2017-11-02
 * @author moonhoen lee
 * @brief Measure Entry를 관리한다.
 * @details
 */

#ifndef MEASUREMANAGER_H
#define MEASUREMANAGER_H 

#include <vector>
#include <mutex>
#include <string>
#include <map>

#include "MeasureEntry.h"


class MeasureManager {
public: 
    MeasureManager() {}
    virtual ~MeasureManager() {}
    
    static void insertEntryEx(int networkID, std::vector<std::string> itemNames,
        MeasureOption option, int queueSize);
    static void insertEntry(int networkID, std::vector<std::string> itemNames);
    static void removeEntry(int networkID);
    static MeasureEntry* getMeasureEntry(int networkID);

private:
    static std::map<int, MeasureEntry*>     entryMap;
    static std::mutex                       entryMapMutex;

};

#endif /* MEASUREMANAGER_H */
