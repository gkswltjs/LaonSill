/**
 * @file MemoryMgmt.cpp
 * @date 2017-12-05
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <sys/time.h>
#include <time.h>

#include <vector>
#include <algorithm>

#include "MemoryMgmt.h"
#include "SysLog.h"
#include "Param.h"
#include "FileMgmt.h"

using namespace std;

extern const char*  LAONSILL_HOME_ENVNAME;

map<void*, MemoryEntry>     MemoryMgmt::entryMap;
mutex                       MemoryMgmt::entryMutex;
uint64_t                    MemoryMgmt::usedMemTotalSize;
uint64_t                    MemoryMgmt::currIndex;

void MemoryMgmt::init() {
    char dumpDir[PATH_MAX];
    SASSERT0(sprintf(dumpDir, "%s/dump", getenv(LAONSILL_HOME_ENVNAME)) != -1);
    FileMgmt::checkDir(dumpDir);

    MemoryMgmt::usedMemTotalSize = 0ULL;
    MemoryMgmt::currIndex = 0ULL;
}

void MemoryMgmt::insertEntry(const char* filename, const char* funcname, int line,
        unsigned long size, bool once, void* ptr) {
    MemoryEntry entry;
    strcpy(entry.filename, filename);
    strcpy(entry.funcname, funcname);
    entry.line = line;
    entry.size = size;
    entry.once = once;

    unique_lock<mutex> entryMutexLock(MemoryMgmt::entryMutex);
    entry.index = MemoryMgmt::currIndex;
    MemoryMgmt::currIndex = MemoryMgmt::currIndex + 1;

    // XXX: LMH debug
    if (MemoryMgmt::entryMap.find(ptr) != MemoryMgmt::entryMap.end()) {
        entryMutexLock.unlock();
        dump(MemoryMgmtSortOptionNone, true);

        printf("filename : %s, funcname : %s, line : %d, size : %lu, pointer : %p\n", 
                filename, funcname, line, size, ptr);
        entryMutexLock.lock();
    }

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
void MemoryMgmt::dump(MemoryMgmtSortOption option, bool skipOnce) {
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

    if (option == MemoryMgmtSortOptionNone) {
        fprintf(fp, "no option, ");
    } else if (option == MemoryMgmtSortOptionIndex) {
        fprintf(fp, "index sort, ");
    } else {
        SASSERT0(option == MemoryMgmtSortOptionSize);
        fprintf(fp, "size sort, ");
    }

    if (skipOnce) {
        fprintf(fp, "skip once.\n");
    } else {
        fprintf(fp, "print all.\n");
    }
    fflush(fp);

    map<void*, MemoryEntry>::iterator it;
    unique_lock<mutex> entryMutexLock(MemoryMgmt::entryMutex);    
    map<void*, MemoryEntry> cp = MemoryMgmt::entryMap;
    entryMutexLock.unlock();

    if (option == MemoryMgmtSortOptionNone) {
        for (it = cp.begin(); it != cp.end(); it++) {
            void* ptr = it->first;
            MemoryEntry entry = it->second;

            if (skipOnce && entry.once)
                continue;

            SASSUME0(fprintf(fp, "[%p|%llu] : %llu (%s()@%s:%d\n", ptr, entry.index, 
                entry.size, entry.funcname, entry.filename, entry.line) > 0);
        }
    } else if (option == MemoryMgmtSortOptionIndex) {
        vector<pair<void*, uint64_t>> vec;

        // XXX: 더 좋은 방법이 있겠지만.. 지금은 vector에 필요한 것들을 복사해서 사용하자.
        //      dump() 조금 빨리 한다고 큰 이점은 없을 것 같다. 
        //      메모리도 deep learing framework에서 더 쓴다고 큰 문제는 없을 꺼 같다.
        for (it = cp.begin(); it != cp.end(); it++)
            vec.push_back(make_pair(it->first, it->second.index));

        struct memoryMgmtSortIndexPred {
            bool operator()(
                const pair<void*, uint64_t> &left, const pair<void*, uint64_t> &right ) {
                return left.second < right.second;
            }
        };
        sort(vec.begin(), vec.end(), memoryMgmtSortIndexPred());

        for (int i = 0; i < vec.size(); i++) {
            void* ptr = vec[i].first;
            MemoryEntry entry = cp[ptr];

            if (skipOnce && entry.once)
                continue;

            SASSUME0(fprintf(fp, "[%p|%llu] : %llu (%s()@%s:%d\n", ptr, entry.index, 
                entry.size, entry.funcname, entry.filename, entry.line) > 0);
        }

    } else { 
        SASSERT0(option == MemoryMgmtSortOptionSize);
        vector<pair<void*, unsigned long>> vec;

        for (it = cp.begin(); it != cp.end(); it++)
            vec.push_back(make_pair(it->first, it->second.size));

        struct memoryMgmtSortIndexSize {
            bool operator()(
                const pair<void*, unsigned long> &left, 
                const pair<void*, unsigned long> &right ) {
                return left.second > right.second;
            }
        };
        sort(vec.begin(), vec.end(), memoryMgmtSortIndexSize());

        for (int i = 0; i < vec.size(); i++) {
            void* ptr = vec[i].first;
            MemoryEntry entry = cp[ptr];

            if (skipOnce && entry.once)
                continue;

            SASSUME0(fprintf(fp, "[%p|%llu] : %llu (%s()@%s:%d\n", ptr, entry.index, 
                entry.size, entry.funcname, entry.filename, entry.line) > 0);
        }
    }


    fflush(fp);
    fclose(fp);
}
