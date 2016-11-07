/**
 * @file SysLog.h
 * @date 2016-10-30
 * @author mhlee
 * @brief 
 * @details
 */

#ifndef SYSLOG_H
#define SYSLOG_H 

#include <mutex>

#include "../common.h"

#define SYS_LOG(fmt, args...)                                                       \
    do {                                                                            \
        if (SysLog::fp) {                                                           \
            std::unique_lock<std::mutex>  logLock(SysLog::logMutex);                \
            SysLog::writeLogHeader(__FILE__,__LINE__);                              \
            fprintf(SysLog::fp, fmt, ##args);                                       \
            fprintf(SysLog::fp, "\n");                                              \
            logLock.unlock();                                                       \
            fflush(SysLog::fp);                                                     \
        } else {                                                                    \
            std::unique_lock<std::mutex>  logLock(SysLog::logMutex);                \
            fprintf(stderr, fmt, ##args);                                           \
            fprintf(stderr, "\n");                                                  \
            logLock.unlock();                                                       \
            fflush(stderr);                                                         \
        }                                                                           \
    } while (0)

#define SASSERT(cond, fmt, args...)                                                 \
    do {                                                                            \
        if (!cond) {                                                                \
            if (SysLog::fp) {                                                       \
                std::unique_lock<std::mutex>  logLock(SysLog::logMutex);            \
                SysLog::writeLogHeader(__FILE__,__LINE__);                          \
                fprintf(SysLog::fp, fmt, ##args);                                   \
                fprintf(SysLog::fp, "\n");                                          \
                SysLog::printStackTrace(SysLog::fp);                                \
                logLock.unlock();                                                   \
                fflush(SysLog::fp);                                                 \
            } else {                                                                \
                std::unique_lock<std::mutex>  logLock(SysLog::logMutex);            \
                fprintf(stderr, fmt, ##args);                                       \
                fprintf(stderr, "\n");                                              \
                SysLog::printStackTrace(stderr);                                    \
                logLock.unlock();                                                   \
                fflush(stderr);                                                     \
            }                                                                       \
            if (SPARAM(SLEEP_WHEN_ASSERTED))                                        \
                sleep(INT_MAX);                                                     \
            else                                                                    \
                assert(0);                                                          \
        }                                                                           \
    } while (0)

#ifdef DEBUG_MODE
#define SASSUME(cond, fmt, args...)			SASSERT(cond, fmt, ##args)
#else
#define SASSUME(cond, fmt, args...)			Nop()
#endif

class SysLog {
public:
                        SysLog() {}
    virtual            ~SysLog() {}
    static void         init();
    static void         destroy();
    static FILE*        fp;
    static std::mutex   logMutex;
    static void         writeLogHeader(const char* fileName, int lineNum);
    static void         printStackTrace(FILE* fp);
private:
    static const char*  sysLogFileName;
};

#endif /* SYSLOG_H */
