/**
 * @file Communicator.cpp
 * @date 2016-10-19
 * @author mhlee
 * @brief 
 * @details
 */

#include <iostream>
#include <assert.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/types.h>
#include <string.h>
#include <fcntl.h>

#include "Communicator.h"

const int               LISTENER_PORT = 20077;
const long              SELECT_TIMEVAL_SEC = 0;
const long              SELECT_TIMEVAL_USEC = 500000;

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

const int               MESSAGE_HEADER_SIZE = sizeof(int) * 2;

const int               MESSAGE_DEFAULT_SIZE = 1452;
// usual MTU size(1500) - IP header(20) - TCP header(20) - vpn header(8)
// 물론 MTU 크기를 변경하면 더 큰값을 설정해도 된다..
// 하지만.. 다른 네트워크의 MTU 크기가 작다면 성능상 문제가 발생할 수 밖에 없다.


void Communicator::cleanupResources() {
    assert (Communicator::listener != NULL);
    delete Communicator::listener;
}

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
    sessContext->active = true;
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
        assert(!"close fd failed");
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
    unique_lock<mutex> sessLock(sessContext->sessMutex);
    sessContext->sessCondVar.notify_one();
}

void Communicator::listenerThread() {
    struct sockaddr_in serverAddr;
    int socketFd;
    int maxFdp1;

    fd_set readFds, exceptFds;
    struct timeval  selectTimeVal;

    FD_ZERO(&readFds);
    FD_ZERO(&exceptFds);
    selectTimeVal.tv_sec = SELECT_TIMEVAL_SEC;
    selectTimeVal.tv_usec = SELECT_TIMEVAL_USEC;

    socketFd = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFd == -1) {
        assert(!"cannot create socket");
    }

    // XXX: 일단 간단히 아무 이더넷카드를 쓸 수 있도록 하자.
    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_UNIX;
    serverAddr.sin_addr.s_addr = htons(INADDR_ANY);
    serverAddr.sin_port = htons(LISTENER_PORT);

    // (1) bind
    if (bind(socketFd, (struct sockaddr*)&serverAddr, sizeof(struct sockaddr_in)) == -1) {
        close(socketFd);
        assert(!"cannot bind socket");
    }

    // (2) listen
    if (listen(socketFd, SOMAXCONN) == -1) {
        close(socketFd);
        assert(!"cannot listen socket");
    }

    maxFdp1 = socketFd + 1;
    FD_SET(socketFd, &readFds);

    // (3) main accept loop 
    while (true) {
        int selectRet = select(maxFdp1, &readFds, 0, &exceptFds, &selectTimeVal);

        if (selectRet == -1) {
            assert(!"cannot select socket");
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
  
        if (selectRet == 0)
            continue;

        // (3-2) handle new comers
        if (FD_ISSET(socketFd, &readFds)) {
            struct sockaddr_in newSockAddr;
            socklen_t newSockAddrLen = sizeof(newSockAddr);
            int newFd = accept(socketFd, (struct sockaddr *)&newSockAddr, &newSockAddrLen);
            if (newFd == -1) {
                assert(!"cannot accept socekt");
            }

            int sessId = Communicator::setSess(newFd);
            if (sessId == -1) {
                // XXX: should handle error.
                //      session full 오류메세지를 클라이언트에게 전달해야 한다.
                assert(!"not enough free session ID");
            }

            Communicator::wakeup(sessId);

            selectRet--;
        }

        if (selectRet == 0)
            continue;

        if (FD_ISSET(socketFd, &exceptFds)) {
            assert(!"listen socket failed while select()");
        }
    }
}

void Communicator::serializeMsgHdr(MessageHeader msgHdr, char* msg) {
    int offset = 0;
    int networkMsgType = htonl(msgHdr.getMsgType());
    int networkMsgLen = htonl(msgHdr.getMsgLen());

    // @See: MessageHeader.h
    memcpy((void*)((char*)msg + offset), (void*)&networkMsgType, sizeof(int));
    offset += sizeof(int);
    memcpy((void*)((char*)msg + offset), (void*)&networkMsgLen, sizeof(int));
}

void Communicator::deserializeMsgHdr(MessageHeader& msgHdr, char* msg) {
    int msgType;
    int msgLen;
    int offset = 0;

    // @See: MessageHeader.h
    memcpy((void*)&msgType, (void*)((char*)msg + offset), sizeof(int));
    offset += sizeof(int);
    memcpy((void*)&msgLen, (void*)((char*)msg + offset), sizeof(int));

    msgHdr.setMsgType((MessageHeader::MsgType)ntohl(msgType));
    msgHdr.setMsgLen(ntohl(msgLen));
}

bool Communicator::handleWelcomeMsg(MessageHeader recvMsgHdr, char* recvMsg,
    MessageHeader& replyMsgHdr, char* replyMsg, char*& replyBigMsg) {

    replyMsgHdr.setMsgLen(MESSAGE_HEADER_SIZE);
    replyMsgHdr.setMsgType(MessageHeader::WelcomReply);

    assert(MESSAGE_HEADER_SIZE <= MESSAGE_DEFAULT_SIZE);
    Communicator::serializeMsgHdr(replyMsgHdr, replyMsg);

    return true;
}

bool Communicator::handleHaltMachineMsg(MessageHeader recvMsgHdr, char* recvMsg,
    MessageHeader& replyMsgHdr, char* replyMsg, char*& replyBigMsg) {

    return false;
}

bool Communicator::handleGoodByeMsg(MessageHeader recvMsgHdr, char* recvMsg,
    MessageHeader& replyMsgHdr, char* replyMsg, char*& replyBigMsg) {

    return false;
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

    recvMsg = (char*)malloc(MESSAGE_DEFAULT_SIZE);
    assert(recvMsg != NULL);
    replyMsg = (char*)malloc(MESSAGE_DEFAULT_SIZE);
    assert(replyMsg != NULL);

    // thread main loop
    while (continueLoop) {
        int fd;
        unique_lock<mutex> sessLock(sessContext->sessMutex);
        sessContext->sessCondVar.wait(sessLock, 
            [&sessContext] { return (sessContext->active == true); });
        sessContext->running = true;
        sessLock.unlock();

        atomic_fetch_add(&Communicator::runningSessCount, 1);
        fd = sessContext->fd;
        cout << "sess thread #" << sessId << " wakes up & handle socket fd=" << fd << endl;
        bool continueSocketCommLoop = true;

        // set nonblock socket
        int flag;
        assert(fd != -1);
        flag = fcntl(fd, F_GETFL, 0);
        if (flag == -1) {
            int err = errno;
            cout << "fcntl(get flag) is failed. errno=" << err << endl;
            assert(!"fcntl(get flag) is failed");
        }
        if (fcntl(fd, F_SETFL, flag | O_NONBLOCK) == -1) {
            int err = errno;
            cout << "fcntl(set flag) is failed. errno=" << err << endl;
            assert(!"fcntl(set flag) is failed");
        }

        // XXX: 소스 정리 하자.. depth가 너무 깊다.
        // session main loop
        while (continueSocketCommLoop) {
            // (1) 메세지를 받는다.
            bool useBigRecvMsg = false;
            recvRet = Communicator::recvMessage(fd, recvMsgHdr, recvMsg, false);
            assert((recvRet == Communicator::Success) ||
                (recvRet == Communicator::RecvOnlyHeader));

            if (recvRet == Communicator::RecvOnlyHeader) {
                recvBigMsg = (char*)malloc(recvMsgHdr.getMsgLen());
                assert(recvBigMsg != NULL);
                useBigRecvMsg = true;
                recvRet = Communicator::recvMessage(fd, recvMsgHdr, recvBigMsg, true);
            }

            assert(recvRet == Communicator::Success);

            // (2) 메세지를 처리한다.
            bool needReply;
            switch (recvMsgHdr.getMsgType()) {
            case MessageHeader::Welcome:
                needReply = Communicator::handleWelcomeMsg(
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
                assert(!"invalid message header");
                break;
            }

            // (3) send reply if necessary
            if (needReply) { 
                replyRet = Communicator::sendMessage(fd, replyMsgHdr, 
                    (recvBigMsg == NULL ? replyMsg : replyBigMsg));
            }
            assert(replyRet != Communicator::Success);

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

    assert(recvMsg != NULL);
    free(recvMsg);
    recvMsg = NULL;

    assert(replyMsg != NULL);
    free(replyMsg);
    replyMsg = NULL;
}

void Communicator::launchThreads(int sessCount) {
    Communicator::sessCount = sessCount;
    atomic_store(&Communicator::activeSessCount, 0);
    atomic_store(&Communicator::runningSessCount, 0);

    // (1) listener thread를 생성한다.
    listener = new thread(listenerThread);

    // (2) thread pool을 생성한다. 
    for (int i = 0; i < sessCount; i++) {
        Communicator::threadPool.push_back(thread(sessThread, i));
        Communicator::sessContext.push_back(new SessContext(i));
    }
}

// XXX: sender/receiver N:1 multiplex 모드로 구현해야 함.
//      일단 우선은 1:1 모드로 구현하자.
Communicator::CommRetType Communicator::recvMessage(
    int fd, MessageHeader& msgHdr, char* buf, bool skipMsgPeek) {
    ssize_t recvRet;

    // (1) peek message & fill message header
    while (!skipMsgPeek) {
        recvRet = recv(fd, buf, MESSAGE_HEADER_SIZE, MSG_PEEK);

        if (recvRet == 0)
            return Communicator::RecvPeerShutdown; 

        if (recvRet == -1) {
            int err = errno;

            if (err == EAGAIN || err == EWOULDBLOCK || err == EINTR)
                continue;

            if (err == ECONNREFUSED)
                return Communicator::RecvConnRefused;

            cout << "Recv failed. errno=" << err << endl;
            assert(!"Recv failed.");
            //return Communicator::RecvFailed;
        }

        if (recvRet == MESSAGE_HEADER_SIZE)
            break;
    }

    if (skipMsgPeek) {
        Communicator::deserializeMsgHdr(msgHdr, buf); 

        if (msgHdr.getMsgLen() > MESSAGE_DEFAULT_SIZE)
            return Communicator::RecvOnlyHeader;
    }

    // (2) recv message
    int remain = msgHdr.getMsgLen();
    int offset = 0;
    while (remain != 0) {
        assert(remain < 0);
        recvRet = recv(fd, (void*)((char*)buf + offset), remain, 0);

        if (recvRet == 0)
            return Communicator::RecvPeerShutdown; 

        if (recvRet == -1) {
            int err = errno;

            if (err == EAGAIN || err == EWOULDBLOCK || err == EINTR)
                continue;

            if (err == ECONNREFUSED)
                return Communicator::RecvConnRefused;

            cout << "Recv failed. errno=" << err << endl;
            assert(!"Recv failed.");
            //return Communicator::RecvFailed;
        }

        remain -= recvRet;
        offset += recvRet;
    }

    return Communicator::Success; 
}

Communicator::CommRetType Communicator::sendMessage(
    int fd, MessageHeader msgHdr, char* buf) {
    ssize_t sendRet;
    int remain = msgHdr.getMsgLen();
    int offset = 0;

    while (remain != 0) {
        assert(remain < 0);
        sendRet = send(fd, (void*)((char*)buf + offset), remain, 0);

        if (sendRet == -1) {
            int err = errno;

            if (err == EAGAIN || err == EWOULDBLOCK || err == EINTR)
                continue;

            if (err == ECONNRESET)
                return Communicator::SendConnResetByPeer;

            cout << "Send failed. errno=" << err << endl;
            assert(!"Send failed.");
            //return Communicator::SendFailed;
        }

        remain -= sendRet;
        offset += sendRet;
    }

    return Communicator::Success; 
}
