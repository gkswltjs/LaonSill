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
#include "Serializer.h"

using namespace std;

class Communicator {
public:
    enum CommRetType : int {
        Success = 0,
        RecvOnlyHeader,         // Big Message
        RecvFailed,
        RecvConnRefused,
        RecvPeerShutdown,
        SendConnResetByPeer,
        SendFailed,
    };

                                Communicator() {}
    virtual                    ~Communicator() {}

    static const int            LISTENER_PORT;

    static void                 launchThreads(int sessCount);
    static void                 joinThreads();
    static void                 halt();

    static CommRetType          recvMessage(int fd, MessageHeader& msgHdr, char* buf,
                                    bool skipMsgPeek);
    static CommRetType          sendMessage(int fd, MessageHeader msgHdr, char* buf);
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

    static bool                 handleWelcomeMsg(MessageHeader recvMsgHdr, char* recvMsg,
                                    MessageHeader& replyMsgHdr, char* replyMsg, 
                                    char*& replyBigMsg);
    static bool 		        handleCreateNetworkMsg(MessageHeader recvMsgHdr, char* recvMsg,
            						MessageHeader& replyMsgHdr, char* replyMsg,
                                    char*& replyBigMsg);
    static bool 				handlePushJobMsg(MessageHeader recvMsgHdr, char* recvMsg,
                					MessageHeader& replyMsgHdr, char* replyMsg,
                                    char*& replyBigMsg);
    static bool                 handleHaltMachineMsg(MessageHeader recvMsgHdr, char* recvMsg,
                                    MessageHeader& replyMsgHdr, char* replyMsg,
                                    char*& replyBigMsg);
    static bool                 handleGoodByeMsg(MessageHeader recvMsgHdr, char* recvMsg,
                                    MessageHeader& replyMsgHdr, char* replyMsg,
                                    char*& replyBigMsg);

    static volatile bool        halting;
    static atomic<int>          sessHaltCount;
};

#endif /* COMMUNICATOR_H */
