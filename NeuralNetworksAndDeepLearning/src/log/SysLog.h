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
            fprintf(stderr, fmt, ##args);                                           \
            fprintf(stderr, "\n");                                                  \
            fflush(stderr);                                                         \
        }                                                                           \
    } while (0)

class SysLog {
public:
                        SysLog() {}
    virtual            ~SysLog() {}
    static void         init();
    static void         destroy();
    static FILE*        fp;
    static std::mutex   logMutex;
    static void         writeLogHeader(const char* fileName, int lineNum);
private:
    static const char*  sysLogFileName;
};
#endif /* SYSLOG_H */
