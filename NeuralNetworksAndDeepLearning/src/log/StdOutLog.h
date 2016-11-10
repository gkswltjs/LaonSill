/**
 * @file StdOutLog.h
 * @date 2016-11-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef STDOUTLOG_H
#define STDOUTLOG_H 

#include "../common.h"

#include <mutex>

#define STDOUT_BLOCK(stmt)                                                  \
    do {                                                                    \
        std::unique_lock<std::mutex> logLock(StdOutLog::logMutex);          \
        StdOutLog::writeLogHeader();                                        \
        do { stmt; } while(0);                                              \
        logLock.unlock();                                                   \
        fflush(stdout);                                                     \
    } while (0)                                                             

#define STDOUT_LOG(fmt, args...)                                            \
    do {                                                                    \
        std::unique_lock<std::mutex> logLock(StdOutLog::logMutex);          \
        StdOutLog::writeLogHeader();                                        \
        fprintf(stdout, fmt, ##args);                                       \
        fprintf(stdout, "\n");                                              \
        logLock.unlock();                                                   \
        fflush(stdout);                                                     \
    } while (0)                                                             

class StdOutLog {
public: 
    StdOutLog() {}
    virtual ~StdOutLog() {}

    static std::mutex   logMutex;
    static void         writeLogHeader();
};

#endif /* STDOUTLOG_H */
