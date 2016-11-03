/**
 * @file FileMgmt.h
 * @date 2016-10-31
 * @author mhlee
 * @brief 
 * @details
 */

#ifndef FILEMGMT_H
#define FILEMGMT_H 

#include "common.h"

class FileMgmt {
public: 
                FileMgmt() {}
    virtual    ~FileMgmt() {}

    static void checkDir(const char* path);
    static int  openFile(const char* path, int flag);
private:
    static void makeDir(const char* path);
};

#endif /* FILEMGMT_H */
