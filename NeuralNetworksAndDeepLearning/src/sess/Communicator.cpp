/**
 * @file Communicator.cpp
 * @date 2016-10-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <sys/socket.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/types.h>
#include <string.h>
#include <fcntl.h>
#include <arpa/inet.h>

#include "Communicator.h"
#include "Worker.h"
#include "Param.h"
#include "ColdLog.h"
#include "SysLog.h"
#include "LegacyWork.h"
#include "ThreadMgmt.h"

using namespace std;

// XXX: 포트 복원해야 함.
//const int               Communicator::LISTENER_PORT = 20087;
const int               Communicator::LISTENER_PORT = 20088;
const long              SELECT_TIMEVAL_SEC = 0L;
const long              SELECT_TIMEVAL_USEC = 500000L;

int                     Communicator::sessCount;

vector<SessContext*>    Communicator::sessContext;
vector<thread>          Communicator::threadPool;
thread*                 Communicator::listener = NULL;
atomic<int>             Communicator::activeSessCount;
atomic<int>             Communicator::runningSessCount;

list<int>               Communicator::freeSessIdList;
mutex                   Communicator::freeSessIdMutex;

map<int, int>           Communicator::fdToSessMap;
mutex                   Communicator::fdToSessMutex;

volatile bool           Communicator::halting = false;
atomic<int>             Communicator::sessHaltCount;

int Communicator::setSess(int newFd) {
    int sessId;

    unique_lock<mutex> freeSessIdLock(Communicator::freeSessIdMutex);
    if (Communicator::freeSessIdList.empty()) {
        freeSessIdLock.unlock();
        return -1;
    }

    sessId = Communicator::freeSessIdList.front(); 
    Communicator::freeSessIdList.pop_front();
    freeSessIdLock.unlock();

    SessContext*& sessContext = Communicator::sessContext[sessId];
    sessContext->sessId = sessId;
    sessContext->fd = newFd;
    atomic_fetch_add(&Communicator::activeSessCount, 1);

    unique_lock<mutex> fdToSessLock(Communicator::fdToSessMutex);
    Communicator::fdToSessMap[newFd] = sessId;
    fdToSessLock.unlock();

    return sessId;
}

void Communicator::releaseFd(int sessId) {
    SessContext*& sessContext = Communicator::sessContext[sessId];
    unique_lock<mutex> fdToSessLock(Communicator::fdToSessMutex);
    Communicator::fdToSessMap.erase(sessId);
    fdToSessLock.unlock();

    if (close(sessContext->fd) == -1)
        SASSERT(false, "close fd failed");
}

void Communicator::releaseSess(int sessId) {
    SessContext*& sessContext = Communicator::sessContext[sessId];
    sessContext->active = false;
    sessContext->running = false;
    sessContext->fd = -1;

    // 순간적으로 active sess count가 running sess count보다 높아질 수 있다.
    // 하지만, 자주 발생하는 일은 아니며, 그렇게 되더라도 안정장치가 되어 있기 때문에
    // 큰 문제가 되지 않을 것으로 판단된다. 
    // 추후에 성능상 문제가 발생하면 수정하도록 한다.
    atomic_fetch_sub(&Communicator::activeSessCount, 1);
    atomic_fetch_sub(&Communicator::runningSessCount, 1);

    unique_lock<mutex> freeSessIdLock(Communicator::freeSessIdMutex);
    Communicator::freeSessIdList.push_back(sessId);
}

void Communicator::wakeup(int sessId) {
    SessContext*& sessContext = Communicator::sessContext[sessId];
    sessContext->active = true;
    unique_lock<mutex> sessLock(sessContext->sessMutex);
    sessContext->sessCondVar.notify_one();
}

void Communicator::listenerThread() {
    struct sockaddr_in serverAddr;
    int socketFd;
    int maxFdp1;

    socketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd == -1) {
        SASSERT(false, "cannot create socket");
    }

    // XXX: 일단 간단히 아무 이더넷카드를 쓸 수 있도록 하자.
    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddr.sin_port = htons(LISTENER_PORT);

    // (1) bind
    if (bind(socketFd, (struct sockaddr*)&serverAddr, sizeof(struct sockaddr_in)) == -1) {
        int err = errno;
        COLD_LOG(ColdLog::ERROR, true, "bind() failed. errno=%d", err);
        close(socketFd);
        SASSERT(false, "bind() failed. err=%d", err);
    }

    // (2) listen
    if (listen(socketFd, SOMAXCONN) == -1) {
        close(socketFd);
        COLD_LOG(ColdLog::ERROR, true, "listen() failed.");
        SASSERT(false, "listen() failed");
    }

    maxFdp1 = socketFd + 1;

    COLD_LOG(ColdLog::INFO, true, "listener thread starts");

    // (3) main accept loop 
    while (true) {
        struct timeval  selectTimeVal;
        selectTimeVal.tv_sec = SELECT_TIMEVAL_SEC;
        selectTimeVal.tv_usec = SELECT_TIMEVAL_USEC;

        fd_set readFds;
        FD_ZERO(&readFds);
        FD_SET(socketFd, &readFds);

        int selectRet = select(maxFdp1, &readFds, 0, 0, &selectTimeVal);

        if (Communicator::halting)
            break;

        if (selectRet == -1) {
            COLD_LOG(ColdLog::ERROR, true, "select() failed.");
            SASSERT(false, "select() failed");
        }

        // (3-1) check & wakeup hang session thread
        // XXX: 얼마나 자주 발생하는지 측정하고, 성능에 문제가 있으면 수정하자.
        if (atomic_load(&Communicator::activeSessCount) >
            atomic_load(&Communicator::runningSessCount)) {

            map<int, int>::iterator iter; 
            for (iter = Communicator::fdToSessMap.begin();
                iter != Communicator::fdToSessMap.end(); iter++) {

                int sessId = iter->second;
                SessContext*& sessContext = Communicator::sessContext[sessId];
                if (sessContext->active && !sessContext->running)
                    Communicator::wakeup(sessId);
            }
        }
  
        // (3-2) handle new comers
        if (FD_ISSET(socketFd, &readFds)) {
            struct sockaddr_in newSockAddr;
            socklen_t newSockAddrLen = sizeof(newSockAddr);
            int newFd = accept(socketFd, (struct sockaddr *)&newSockAddr, &newSockAddrLen);
            if (newFd == -1) {
                COLD_LOG(ColdLog::ERROR, true, "accept() failed.");
                SASSERT(false, "accept() failed");
            }

            COLD_LOG(ColdLog::INFO, true, "accept socket. newFd=%d", newFd);
            int sessId = Communicator::setSess(newFd);
            if (sessId == -1) {
                // FIXME: should handle error.
                //      session full 오류메세지를 클라이언트에게 전달해야 한다.
                COLD_LOG(ColdLog::WARNING, true, "not enough free session ID");
                SASSERT(false, "not enough free session ID."
                    "This should be handled in the future");
            }

            COLD_LOG(ColdLog::INFO, true, "get session. session ID=%d", sessId);

            Communicator::wakeup(sessId);

            selectRet--;
        }

        // FIXME: exception에 대한 처리 필요
    }

    close(socketFd);
}

bool Communicator::handleWelcomeMsg(MessageHeader recvMsgHdr, char* recvMsg,
    MessageHeader& replyMsgHdr, char* replyMsg, char*& replyBigMsg) {

    replyMsgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    replyMsgHdr.setMsgType(MessageHeader::WelcomeReply);

    SASSERT(MessageHeader::MESSAGE_HEADER_SIZE <= MessageHeader::MESSAGE_DEFAULT_SIZE, "");
    MsgSerializer::serializeMsgHdr(replyMsgHdr, replyMsg);

    return true;
}

bool Communicator::handleHaltMachineMsg(MessageHeader recvMsgHdr, char* recvMsg,
    MessageHeader& replyMsgHdr, char* replyMsg, char*& replyBigMsg) {

    // (1) Worker Thread (Producer& Consumer)를 종료한다.
    Job* haltJob = new Job(Job::HaltMachine);
    Worker::pushJob(haltJob);

    // (2) Listener, Session 쓰레드 들을 모두 종료한다.
    Communicator::halt();       // threads will be eventually halt

    return false;
}

bool Communicator::handleGoodByeMsg(MessageHeader recvMsgHdr, char* recvMsg,
    MessageHeader& replyMsgHdr, char* replyMsg, char*& replyBigMsg) {
    // TODO: 아직 구현 안되었음..
    return false;
}


// CreateNetwork body packet 구성
// |---------------------------------------------------------------|
// | Not specified.. configuration information will be specified   | 
// |                                                               |
// |---------------------------------------------------------------|
//
// CreateNetworkReply body packet 구성
// |------------------|
// | Network Id       |
// | int(4)           |
// |------------------|
bool Communicator::handleCreateNetworkMsg(MessageHeader recvMsgHdr, char* recvMsg,
    MessageHeader& replyMsgHdr, char* replyMsg, char*& replyBigMsg) {

    // (1) create network.
    int networkId = LegacyWork<float>::createNetwork();

    // (2) prepare header
    int msgLen = MessageHeader::MESSAGE_HEADER_SIZE + sizeof(int);
    replyMsgHdr.setMsgLen(msgLen);
    replyMsgHdr.setMsgType(MessageHeader::CreateNetworkReply);

    // (3) fill send buffer;
    char *sendBuffer;
    if (msgLen > MessageHeader::MESSAGE_DEFAULT_SIZE) {
        replyBigMsg = (char*)malloc(msgLen);
        SASSERT(replyBigMsg, "");
        sendBuffer = replyBigMsg;
    } else {
        sendBuffer = replyMsg;
    }

    int offset;
    offset = MsgSerializer::serializeMsgHdr(replyMsgHdr, sendBuffer);
    offset = MsgSerializer::serializeInt(networkId, offset, sendBuffer);

    return true;
}


// PushJob body packet 구성
// +--------+---------+------------+-------------------+----------+
// | JobID  | JobType | JobElemCnt | JobElemTypes      | JobElems |
// | int(4) | int(4)  | int(4)     | int(4)*JobElemCnt | variable |
// +--------+---------+------------+-------------------+----------+
bool Communicator::handlePushJobMsg(MessageHeader recvMsgHdr, char* recvMsg,
    MessageHeader& replyMsgHdr, char* replyMsg, char*& replyBigMsg) {
    // (1) create job
    int             jobID;
    int             jobType;
    int             jobElemCnt;
    int             offset = MessageHeader::MESSAGE_HEADER_SIZE;

    offset = MsgSerializer::deserializeInt(jobID, offset, recvMsg);
    offset = MsgSerializer::deserializeInt(jobType, offset, recvMsg);
    offset = MsgSerializer::deserializeInt(jobElemCnt, offset, recvMsg);

    Job* newJob = new Job((Job::JobType)jobType);

    int *jobElemTypes = NULL;
    if (jobElemCnt > 0) {
        jobElemTypes = (int*)malloc(sizeof(int) * jobElemCnt);
        SASSERT0(jobElemTypes != NULL);
    }

    for (int i = 0 ; i < jobElemCnt; i++) {
        offset = MsgSerializer::deserializeInt(jobElemTypes[i], offset, recvMsg);
    }

    // FIXME: should split source :)
    int         tempArrayCount;
    int         tempInt;
    float       tempFloat;
    float      *tempFloatArray = NULL;
    for (int i = 0; i < jobElemCnt; i++) {
        switch ((Job::JobElemType)jobElemTypes[i]) {
            case Job::IntType:
                offset = MsgSerializer::deserializeInt(tempInt, offset, recvMsg);
                newJob->addJobElem((Job::JobElemType)jobElemTypes[i], 1, (void*)&tempInt);
                break;

            case Job::FloatType:
                offset = MsgSerializer::deserializeFloat(tempFloat, offset, recvMsg);
                newJob->addJobElem((Job::JobElemType)jobElemTypes[i], 1, (void*)&tempFloat);
                break;

            case Job::FloatArrayType:
                offset = MsgSerializer::deserializeInt(tempArrayCount, offset, recvMsg);
                SASSERT0(tempFloatArray == NULL);
                tempFloatArray = (float*)malloc(sizeof(float) * tempArrayCount);
                SASSERT0(tempFloatArray != NULL);

                for (int j = 0; j < tempArrayCount; j++) {
                    offset = MsgSerializer::deserializeFloat(tempFloatArray[j], offset,
                            recvMsg);
                }
                newJob->addJobElem((Job::JobElemType)jobElemTypes[i], tempArrayCount,
                    (void*)tempFloatArray);
                free(tempFloatArray);
                break;

            default:
                SASSERT(false, "invalid job elem type. elem type=%d", jobElemTypes[i]);
                break;
        }
    }

    if (jobElemTypes != NULL)
        free(jobElemTypes);

    // (2) job을 job queue에 넣는다.
    Worker::pushJob(newJob);

    // (3) reply
    replyMsgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    replyMsgHdr.setMsgType(MessageHeader::PushJobReply);

    SASSERT(MessageHeader::MESSAGE_HEADER_SIZE <= MessageHeader::MESSAGE_DEFAULT_SIZE, "");
    MsgSerializer::serializeMsgHdr(replyMsgHdr, replyMsg);

    return true;
}

void Communicator::sessThread(int sessId) {
    bool            continueLoop    = true;

    MessageHeader   recvMsgHdr;
    MessageHeader   replyMsgHdr;
    char*           recvMsg;
    char*           replyMsg;
    char*           recvBigMsg  = NULL;     // 동적할당
    char*           replyBigMsg = NULL;
    CommRetType     recvRet;
    CommRetType     replyRet;

    SessContext*& sessContext   = Communicator::sessContext[sessId];

    recvMsg = (char*)malloc(MessageHeader::MESSAGE_DEFAULT_SIZE);
    SASSERT0(recvMsg);
    replyMsg = (char*)malloc(MessageHeader::MESSAGE_DEFAULT_SIZE);
    SASSERT0(replyMsg);

    COLD_LOG(ColdLog::INFO, true, "session thread #%d starts", sessId);

    // thread main loop
    while (continueLoop) {
        int fd;
        unique_lock<mutex> sessLock(sessContext->sessMutex);

        sessContext->sessCondVar.wait(sessLock, 
            [&sessContext] { return (sessContext->active == true); });
        sessContext->running = true;
        sessLock.unlock();

        atomic_fetch_add(&Communicator::runningSessCount, 1);

        if (Communicator::halting) {
            atomic_fetch_add(&Communicator::sessHaltCount, 1);
            break;
        }

        fd = sessContext->fd;
        COLD_LOG(ColdLog::INFO, true, "session thread #%d wakes up & handle socket(fd=%d)",
                sessId, fd);
        bool continueSocketCommLoop = true;

        // set nonblock socket
        int flag;
        SASSERT(fd != -1, "");
        flag = fcntl(fd, F_GETFL, 0);
        if (flag == -1) {
            int err = errno;
            COLD_LOG(ColdLog::ERROR, true, "fcntl(get flag) is failed. errno=%d", err);
            SASSERT(0, "");
        }
        if (fcntl(fd, F_SETFL, flag | O_NONBLOCK) == -1) {
            int err = errno;
            COLD_LOG(ColdLog::ERROR, true, "fcntl(set flag) is failed. errno=%d", err);
            SASSERT(0, "");
        }

        // XXX: 소스 정리 하자.. depth가 너무 깊다.
        // session main loop
        while (continueSocketCommLoop) {
            // (1) 메세지를 받는다.
            bool useBigRecvMsg = false;
            recvRet = Communicator::recvMessage(fd, recvMsgHdr, recvMsg, false);
            SASSERT((recvRet == Communicator::Success) ||
                (recvRet == Communicator::RecvOnlyHeader), "");

            if (recvRet == Communicator::RecvOnlyHeader) {
                recvBigMsg = (char*)malloc(recvMsgHdr.getMsgLen());
                SASSERT0(recvBigMsg);
                useBigRecvMsg = true;
                recvRet = Communicator::recvMessage(fd, recvMsgHdr, recvBigMsg, true);
            }

            SASSERT(recvRet == Communicator::Success, "");

            // (2) 메세지를 처리한다.
            bool needReply;
            switch (recvMsgHdr.getMsgType()) {
            case MessageHeader::Welcome:
                needReply = Communicator::handleWelcomeMsg(
                    recvMsgHdr, (useBigRecvMsg ? recvBigMsg : recvMsg),
                    replyMsgHdr, replyMsg, replyBigMsg);
                break;
            case MessageHeader::PushJob:
                needReply = Communicator::handlePushJobMsg(
                    recvMsgHdr, (useBigRecvMsg ? recvBigMsg : recvMsg),
                    replyMsgHdr, replyMsg, replyBigMsg);
                break;

            case MessageHeader::CreateNetwork:
                needReply = Communicator::handleCreateNetworkMsg(
                    recvMsgHdr, (useBigRecvMsg ? recvBigMsg : recvMsg),
                    replyMsgHdr, replyMsg, replyBigMsg);
                break;

            case MessageHeader::HaltMachine:
                needReply = Communicator::handleHaltMachineMsg(
                    recvMsgHdr, (useBigRecvMsg ? recvBigMsg : recvMsg),
                    replyMsgHdr, replyMsg, replyBigMsg);
                continueSocketCommLoop = false;
                continueLoop = false;
                break;

            case MessageHeader::GoodBye:
                needReply = Communicator::handleGoodByeMsg(
                    recvMsgHdr, (useBigRecvMsg ? recvBigMsg : recvMsg),
                    replyMsgHdr, replyMsg, replyBigMsg);
                continueSocketCommLoop = false;
                break;

            default:
                SASSERT(!"invalid message header", "");
                break;
            }

            // (3) send reply if necessary
            if (needReply) {
                replyRet = Communicator::sendMessage(fd, replyMsgHdr, 
                    (recvBigMsg == NULL ? replyMsg : replyBigMsg));
            }
            SASSERT(replyRet == Communicator::Success, "");

            // (4) cleanup big msg resource
            if (recvBigMsg != NULL) {
                free(replyBigMsg);
                replyBigMsg = NULL;
            } 

            if (recvBigMsg != NULL) {
                free(recvBigMsg);
                recvBigMsg = NULL;
            }
        }

        Communicator::releaseFd(sessId);
        Communicator::releaseSess(sessId);
    }

    SASSERT0(recvMsg);
    free(recvMsg);
    recvMsg = NULL;

    SASSERT0(replyMsg);
    free(replyMsg);
    replyMsg = NULL;

    COLD_LOG(ColdLog::INFO, true, "session thread #%d ends", sessId);
}

void Communicator::launchThreads(int sessCount) {
    // (1) 초기값 설정
    Communicator::sessCount = sessCount;
    atomic_store(&Communicator::activeSessCount, 0);
    atomic_store(&Communicator::runningSessCount, 0);
    atomic_store(&Communicator::sessHaltCount, 0);

    // (2) ThreadMgmt의 준비가 될때까지 기다린다.
    while (!ThreadMgmt::isReady()) {
        sleep(1);
    }

    // (3) listener thread를 생성한다.
    Communicator::listener = new thread(listenerThread);

    // (4) thread pool을 생성한다. 
    for (int i = 0; i < sessCount; i++) {
        Communicator::sessContext.push_back(new SessContext(i));
        Communicator::freeSessIdList.push_back(i);
    }

    // XXX: 타이밍 이슈가 있음.. order 보호할 수 있는 무언가의 장치가 필요
    sleep(1);
    for (int i = 0; i < sessCount; i++) {
        Communicator::threadPool.push_back(thread(sessThread, i));
    }
}

void Communicator::joinThreads() {
    Communicator::listener->join();
    delete Communicator::listener;
    Communicator::listener = NULL;

    for (int i = 0; i < Communicator::sessCount; i++) {
        Communicator::threadPool[i].join();
    }
}

// XXX: sender/receiver N:1 multiplex 모드로 구현해야 함.
//      일단 우선은 1:1 모드로 구현하자.
Communicator::CommRetType Communicator::recvMessage(
    int fd, MessageHeader& msgHdr, char* buf, bool skipMsgPeek) {
    ssize_t recvRet;

    // (1) peek message & fill message header
    while (!skipMsgPeek) {
        recvRet = recv(fd, buf, MessageHeader::MESSAGE_HEADER_SIZE, MSG_PEEK);

        if (recvRet == 0)
            return Communicator::RecvPeerShutdown; 

        if (recvRet == -1) {
            int err = errno;

            if (err == EAGAIN || err == EWOULDBLOCK || err == EINTR)
                continue;

            if (err == ECONNREFUSED)
                return Communicator::RecvConnRefused;

            COLD_LOG(ColdLog::ERROR, true, "recv(peek message) failed. errno=%d", err);
            SASSERT(0, "");
        }

        if (recvRet == MessageHeader::MESSAGE_HEADER_SIZE)
            break;
    }

    if (!skipMsgPeek) {
        MsgSerializer::deserializeMsgHdr(msgHdr, buf);

        if (msgHdr.getMsgLen() > MessageHeader::MESSAGE_DEFAULT_SIZE)
            return Communicator::RecvOnlyHeader;
    }

    // (2) recv message
    int remain = msgHdr.getMsgLen();
    int offset = 0;
    while (remain != 0) {
        SASSERT(remain >= 0, "");
        recvRet = recv(fd, (void*)((char*)buf + offset), remain, 0);

        if (recvRet == 0)
            return Communicator::RecvPeerShutdown; 

        if (recvRet == -1) {
            int err = errno;

            if (err == EAGAIN || err == EWOULDBLOCK || err == EINTR)
                continue;

            if (err == ECONNREFUSED)
                return Communicator::RecvConnRefused;

            COLD_LOG(ColdLog::ERROR, true, "recv() failed. errno=%d", err);
            SASSERT(0, "");
            //return Communicator::RecvFailed;
        }

        remain -= recvRet;
        offset += recvRet;
    }

    return Communicator::Success; 
}

/**
 * @warning buf에는 이미 serialize 된 메세지가 들어 있어야 한다.
 */
Communicator::CommRetType Communicator::sendMessage(
    int fd, MessageHeader msgHdr, char* buf) {
    ssize_t sendRet;
    int remain = msgHdr.getMsgLen();
    int offset = 0;

    while (remain != 0) {
        SASSERT(remain >= 0, "");
        sendRet = send(fd, (void*)((char*)buf + offset), remain, 0);

        if (sendRet == -1) {
            int err = errno;

            if (err == EAGAIN || err == EWOULDBLOCK || err == EINTR)
                continue;

            if (err == ECONNRESET)
                return Communicator::SendConnResetByPeer;

            COLD_LOG(ColdLog::ERROR, true, "send() failed. errno=%d", err);
            SASSERT(0, "");
        }

        remain -= sendRet;
        offset += sendRet;
    }

    return Communicator::Success; 
}

void Communicator::halt() {
    Communicator::halting = true;

    atomic_fetch_add(&Communicator::sessHaltCount, 1);

    while (atomic_load(&Communicator::sessHaltCount) < Communicator::sessCount) {
        for (int i = 0; i < Communicator::sessCount; i++) {
            Communicator::wakeup(i);
        }

        sleep(1);
    }
}
