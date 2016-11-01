/**
 * @file SysLog.cpp
 * @date 2016-10-30
 * @author mhlee
 * @brief 
 * @details
 */

#include <assert.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>

#include "../param/Param.h"
#include "SysLog.h"
#include "../FileMgmt.h"

using namespace std;

#define gettid()    syscall(SYS_gettid)     // there is no glibc wrapper for this system call;;

extern const char*  SOOOA_HOME_ENVNAME;
const char*         SysLog::sysLogFileName = {"sys.log"};
FILE*               SysLog::fp = NULL;
mutex               SysLog::logMutex;

void SysLog::init() {
    char sysLogFilePath[PATH_MAX];
    char sysLogDir[PATH_MAX];

    if (strcmp(SPARAM(SYSLOG_DIR), "") == 0) {
        assert(sprintf(sysLogDir, "%s/log", getenv(SOOOA_HOME_ENVNAME)) != -1);
        assert(sprintf(sysLogFilePath, "%s/log/%s", getenv(SOOOA_HOME_ENVNAME),
                    SysLog::sysLogFileName) != -1);
    } else {
        assert(sprintf(sysLogDir, "%s", SPARAM(SYSLOG_DIR)) != -1);
        assert(sprintf(sysLogFilePath, "%s/%s", SPARAM(SYSLOG_DIR),
            SysLog::sysLogFileName) != -1);
    }

    FileMgmt::checkDir(sysLogDir);

    assert(SysLog::fp == NULL);
    SysLog::fp = fopen(sysLogFilePath, "a");
}

void SysLog::destroy() {
    assert(SysLog::fp);
    fflush(SysLog::fp);
    fclose(SysLog::fp);
    SysLog::fp = NULL;
}

const int SMART_FILENAME_OFFSET = 7;
void SysLog::writeLogHeader(const char* fileName, int lineNum) {
    struct timeval      val;
    struct tm*          tmPtr;
    char                filePath[PATH_MAX];

    assert(SysLog::fp);

    gettimeofday(&val, NULL);
    tmPtr = localtime(&val.tv_sec);
    assert(strlen(fileName) > SMART_FILENAME_OFFSET);
    strcpy(filePath, fileName + SMART_FILENAME_OFFSET); // in order to get rid of "../src/"

    fprintf(SysLog::fp, "[%04d/%02d/%02d %02d:%02d:%02d.%06ld@%s:%d(%d/%d)] ",
        tmPtr->tm_year + 1900, tmPtr->tm_mon + 1,
        tmPtr->tm_mday, tmPtr->tm_hour, tmPtr->tm_min, tmPtr->tm_sec, val.tv_usec,
        filePath, lineNum, (int)getpid(), ((int)gettid() - (int)getpid())); 
}
