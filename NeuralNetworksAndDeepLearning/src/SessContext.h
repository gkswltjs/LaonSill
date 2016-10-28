/**
 * @file SessContext.h
 * @date 2016-10-20
 * @author mhlee
 * @brief 
 * @details
 */

#ifndef SESSCONTEXT_H
#define SESSCONTEXT_H 

#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

class SessContext {
public:
    SessContext(int sessId) {
        this->sessId = sessId;
        this->fd = -1;
        this->running = false;
        this->active = false;
    }
    virtual            ~SessContext() {}
    int                 sessId;
    mutex               sessMutex;
    condition_variable  sessCondVar;
    int                 fd;
    bool                running;
    bool                active;     // should be run
};

#endif /* SESSCONTEXT_H */
