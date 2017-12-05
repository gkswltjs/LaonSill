/**
 * @file MemoryMgmt.cpp
 * @date 2017-12-05
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <sys/time.h>
#include <time.h>

#include "MemoryMgmt.h"
#include "SysLog.h"
#include "Param.h"
#include "FileMgmt.h"

using namespace std;

extern const char*  LAONSILL_HOME_ENVNAME;

map<void*, MemoryEntry>     MemoryMgmt::entryMap;
mutex                       MemoryMgmt::entryMutex;
uint64_t                    MemoryMgmt::usedMemTotalSize;

void MemoryMgmt::init() {
    char dumpDir[PATH_MAX];
    SASSERT0(sprintf(dumpDir, "%s/dump", getenv(LAONSILL_HOME_ENVNAME)) != -1);
    FileMgmt::checkDir(dumpDir);

    MemoryMgmt::usedMemTotalSize = 0ULL;
}

void MemoryMgmt::insertEntry(const char* filename, const char* funcname, int line,
        unsigned long size, void* ptr) {
    MemoryEntry entry;
    strcpy(entry.filename, filename);
    strcpy(entry.funcname, funcname);
    entry.line = line;
    entry.size = size;

    unique_lock<mutex> entryMutexLock(MemoryMgmt::entryMutex);
    SASSUME0(MemoryMgmt::entryMap.find(ptr) == MemoryMgmt::entryMap.end());
    MemoryMgmt::entryMap[ptr] = entry;
    MemoryMgmt::usedMemTotalSize += size;
}

void MemoryMgmt::removeEntry(void* ptr) {
    map<void*, MemoryEntry>::iterator it;
    unique_lock<mutex> entryMutexLock(MemoryMgmt::entryMutex);    
    it = MemoryMgmt::entryMap.find(ptr);
    SASSUME0(it != MemoryMgmt::entryMap.end());
    MemoryMgmt::usedMemTotalSize -= MemoryMgmt::entryMap[ptr].size;
    MemoryMgmt::entryMap.erase(it);
}

// WARNING: 본 함수는 매우 무거운 함수이다. 함부로 자주 호출하면 안된다.
void MemoryMgmt::dump() {
    FILE*           fp;
    char            dumpFilePath[PATH_MAX];
    struct timeval  val;
    struct tm*      tmPtr;

    gettimeofday(&val, NULL);
    tmPtr = localtime(&val.tv_sec);

    SASSERT0(sprintf(dumpFilePath, "%s/dump/%04d%02d%02d_%02d%02d%02d_%06ld.dump",
        getenv(LAONSILL_HOME_ENVNAME), tmPtr->tm_year + 1900, tmPtr->tm_mon + 1,
        tmPtr->tm_mday, tmPtr->tm_hour, tmPtr->tm_min, tmPtr->tm_sec, val.tv_usec) != -1);

    fp = fopen(dumpFilePath, "w+");
    SASSERT0(fp != NULL);

    map<void*, MemoryEntry>::iterator it;
    unique_lock<mutex> entryMutexLock(MemoryMgmt::entryMutex);    
    map<void*, MemoryEntry> cp = MemoryMgmt::entryMap;
    entryMutexLock.unlock();

    for (it = cp.begin(); it != cp.end(); it++) {
        void* ptr = it->first;
        MemoryEntry entry = it->second;

        SASSUME0(fprintf(fp, "[%p] : %llu (%s()@%s:%d\n", ptr, entry.size, entry.funcname, 
            entry.filename, entry.line) > 0);
    }
    fflush(fp);
    fclose(fp);
}
