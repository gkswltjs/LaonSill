/**
 * @file FileMgmt.cpp
 * @date 2016-10-31
 * @author mhlee
 * @brief 
 * @details
 */

#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "FileMgmt.h"

using namespace std;

void FileMgmt::checkDir(const char* path) {
    // XXX: 일단 에러처리를 assert로만 했음.
    struct stat st;
    if (stat(path, &st) == -1) {
        int err = errno;

        if (err == EACCES) {
            cout << "Permission is denied. directory=" << path << endl;
            assert(0);
        } else if (err == ELOOP) {
            cout << "Too manu symbolic links encountered while traversing the path."
                << " directory=" << path << endl;
            assert(0);
        } else if (err == ENAMETOOLONG) {
            cout << "Directory path is too long. directory=" << path << endl;
            assert(0);
        } else if (err == ENOMEM) {
            cout << "Out of memory (i.e., kernel memory). directory=" << path << endl;
            assert(0);
        } else if (err == ENOTDIR) {
            cout << "A component of the path prefix of path is not a directory." << " directory=" << path << endl;
            assert(0);
        } else if (err == ENOENT) {
            FileMgmt::makeDir(path);
        } else {
            cout << "An unknown error has been occurred when stat directory. direcoty=" << path
                << ", errno=" << err << endl;
            assert(0);
        }
    }
}

void FileMgmt::makeDir(const char* path) { 
    if (mkdir(path, 0700) == -1) {
        int err = errno;

        if (err == EACCES) {
            cout << "The parent directory does not allow write permission to the process, or"
                << " one of the directories in pathname did not allow search permission. "
                << "directory=" << path << endl;
        } else if (err == EDQUOT) {
            cout << "The user's quota of disk blocks or inodes on the filesystem has been"
                << " exhausted. directory=" << path << endl;
        } else if (err == EEXIST) {
            cout << "pathname already exists (not necessarily as a directory). "
                << "This includes the case where pathname is a symbolic link, dangling or not."
                << " directory=" << path << endl;
        } else if (err == EFAULT) {
            cout << "pathname points outside your accessible address space."
                << " directory=" << path << endl;
        } else if (err == ELOOP) {
            cout << "Too many symbolic links were encountered in resolving pathname. "
                << "directory=" << path << endl;
        } else if (err == EMLINK) {
            cout << "The number of links to the parent directory would exceed LINK_MAX. "
                << "directory=" << path << endl;
        } else if (err == ENAMETOOLONG) {
            cout << "pathname was too long. directory=" << path << endl;
        } else if (err == ENOENT) {
            cout << "A directory component in pathname does not exist or is a dangling"
                << " symbolic link. directory=" << path << endl;
        } else if (err == ENOMEM) {
            cout << "Insufficient kernel memory was available. directory=" << path << endl;
        } else if (err == ENOSPC) {
            cout << "The device containing pathname has no room for the new directory. "
                << " directory=" << path << endl;
        } else if (err == ENOSPC) {
            cout << "The new directory cannot be created because the user's disk quota "
                << "is exhausted. directory=" << path << endl;
        } else if (err == ENOTDIR) {
            cout << "A component used as a directory in pathname is not, in fact, a "
                << "directory. directory=" << path << endl;
        } else if (err == EPERM) {
            cout << "The filesystem containing pathname does not support the creation"
                << " of directories. directory=" << path << endl;
        } else if (err == EROFS) {
            cout << "pathname refers to a file on a read-only filesystem. "
                << "directory=" << path << endl;
        } else {
            cout << "An unknown error has been occurred when stat directory. direcoty=" << path
                << ", errno=" << err << endl;
            assert(0);
        }
    }
}
