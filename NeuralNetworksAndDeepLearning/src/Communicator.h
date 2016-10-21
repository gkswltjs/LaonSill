/**
 * @file Communicator.h
 * @date 2016-10-19
 * @author mhlee
 * @brief 
 * @details
 */

#ifndef COMMUNICATOR_H
#define COMMUNICATOR_H 

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <list>
#include <atomic>
#include <map>
#include "SessContext.h"
#include "MessageHeader.h"

using namespace std;

class Communicator {
    enum CommRetType : int {
        Success = 0,
        RecvOnlyHeader,         // Big Message
        RecvFailed,
        RecvConnRefused,
        RecvPeerShutdown,
        SendConnResetByPeer,
        SendFailed,
    };

public:
                                Communicator() {}
    virtual                    ~Communicator() {}
    static void                 launchThreads(int sessCount);
    static void                 cleanupResources();

private: 
    static int                  sessCount;

    static vector<SessContext*> sessContext;
    static vector<thread>       threadPool;
    static thread*              listener;

    // 안 깨워진 쓰레드를 관리하기 위한 변수들.
    static atomic<int>          activeSessCount;
    static atomic<int>          runningSessCount;

    static list<int>            freeSessIdList;
    static mutex                freeSessIdMutex;    // guard freeSessionIDList
    static int                  setSess(int newFd);

    static void                 wakeup(int sessId);
    static void                 releaseSess(int sessId);
    static void                 releaseFd(int sessId);

    static map<int, int>        fdToSessMap;
    static mutex                fdToSessMutex;

    static void                 listenerThread();
    static void                 sessThread(int sessId);

    static void                 serializeMsgHdr(MessageHeader msgHdr, char* msg);
    static void                 deserializeMsgHdr(MessageHeader& msgHdr, char* msg);

    static bool                 handleWelcomeMsg(MessageHeader recvMsgHdr, char* recvMsg,
                                    MessageHeader& replyMsgHdr, char* replyMsg, 
                                    char*& replyBigMsg);
    static bool                 handleHaltMachineMsg(MessageHeader recvMsgHdr, char* recvMsg,
                                    MessageHeader& replyMsgHdr, char* replyMsg, 
                                    char*& replyBigMsg);
    static bool                 handleGoodByeMsg(MessageHeader recvMsgHdr, char* recvMsg,
                                    MessageHeader& replyMsgHdr, char* replyMsg, 
                                    char*& replyBigMsg);

    static CommRetType          recvMessage(int fd, MessageHeader& msgHdr, char* buf,
                                    bool skipMsgPeek);
    static CommRetType          sendMessage(int fd, MessageHeader msgHdr, char* buf);
};

#endif /* COMMUNICATOR_H */
