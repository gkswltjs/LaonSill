/**
 * @file Client.h
 * @date 2016-10-25
 * @author mhlee
 * @brief 
 * @details
 */

#ifndef CLIENT_H
#define CLIENT_H 

#include <sys/socket.h>

#include "common.h"
#include "Job.h"

class Client {
public:
                        Client() {}
    virtual            ~Client() {}
    static void         clientMain(const char* hostname, int portno);
private:
    static void         pushJob(int fd, char* buf, int jobType, int networkId,
                            int arg1);
    static int          connectRetry(int socketFd, const struct sockaddr *sockAddr,
                            socklen_t sockLen);
};

#endif /* CLIENT_H */
