/**
 * @file Client.cpp
 * @date 2016-10-25
 * @author mhlee
 * @brief 
 * @details
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <errno.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <arpa/inet.h>

#include "Client.h"
#include "Communicator.h"
#include "Serializer.h"
#include "MessageHeader.h"

const int MAX_RETRY_SECOND = 128;
int Client::connectRetry(int sockFd, const struct sockaddr *sockAddr, socklen_t sockLen) {
    int nsec;

    for (nsec = 1; nsec <= MAX_RETRY_SECOND; nsec <<= 1) {
        if (connect(sockFd, sockAddr, sockLen) == 0) {
            return 0;
        }

        if (nsec <= MAX_RETRY_SECOND/2)
            sleep(nsec);
    }

    return -1;
}

void Client::pushJob(int fd, char* buf, int jobType, int networkId, int arg1) {
    // see handlePushJobMsg()@Communicator.cpp
    // (1) send msg
    MessageHeader msgHdr;
    msgHdr.setMsgType(MessageHeader::PushJob);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE + sizeof(int) * 3);

    int bufOffset = Serializer::serializeMsgHdr(msgHdr, buf);
    bufOffset = Serializer::serializeInt(jobType, bufOffset, buf);
    bufOffset = Serializer::serializeInt(networkId, bufOffset, buf);
    bufOffset = Serializer::serializeInt(arg1, bufOffset, buf);
    Communicator::CommRetType ret = Communicator::sendMessage(fd, msgHdr, buf);
    assert(ret == Communicator::Success);

    // (2) recv msg
    ret = Communicator::recvMessage(fd, msgHdr, buf, false);
    assert(ret == Communicator::Success);
    assert(msgHdr.getMsgType() == MessageHeader::PushJobReply);
}

void Client::clientMain(const char* hostname, int portno) {
    int     sockFd, err;

    // (1) get server info (struct hostent)
    struct hostent *server;
    server = gethostbyname(hostname);
    if (server == NULL) {
        printf("ERROR: no such host as %s\n", hostname);
        exit(0);
    }

    // (2) create socket & connect to the server
    sockFd = socket(AF_INET, SOCK_STREAM, 0);
    assert(sockFd != -1);
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    memcpy((char*)&serverAddr.sin_addr.s_addr, (char*)server->h_addr, server->h_length);
    serverAddr.sin_port = htons(portno);
    if (Client::connectRetry(sockFd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        printf("ERROR: connect failed\n");
        exit(0);
    }

    // (3-1) send welcome msg
    char* buf = (char*)malloc(MessageHeader::MESSAGE_DEFAULT_SIZE);
    assert(buf != NULL);

    MessageHeader msgHdr;
    msgHdr.setMsgType(MessageHeader::Welcome);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    Serializer::serializeMsgHdr(msgHdr, buf);
    Communicator::CommRetType ret = Communicator::sendMessage(sockFd, msgHdr, buf);
    assert(ret == Communicator::Success);

    // (3-2) recv welcome reply msg
    ret = Communicator::recvMessage(sockFd, msgHdr, buf, false);
    assert(ret == Communicator::Success);
    assert(msgHdr.getMsgType() == MessageHeader::WelcomeReply);

    // (4-1) send create network msg
    msgHdr.setMsgType(MessageHeader::CreateNetwork);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    Serializer::serializeMsgHdr(msgHdr, buf);
    ret = Communicator::sendMessage(sockFd, msgHdr, buf);
    assert(ret == Communicator::Success);

    // (4-2) recv create network reply msg & get networkId
    // see handleCreateNetworkMsg()@Communicator.cpp
    ret = Communicator::recvMessage(sockFd, msgHdr, buf, false);
    assert(ret == Communicator::Success);
    assert(msgHdr.getMsgType() == MessageHeader::CreateNetworkReply);
    int networkId;
    Serializer::deserializeInt(networkId, MessageHeader::MESSAGE_HEADER_SIZE, buf);
    cout << "network ID :" << networkId << endl;

    // (5-1) build layer
    Client::pushJob(sockFd, buf, (int)Job::BuildLayer, networkId, 0);
    
    // (5-2) train network
    Client::pushJob(sockFd, buf, (int)Job::TrainNetwork, networkId, 2);
        // 2 means max epoch count

    // (5-3) cleanup layer
    Client::pushJob(sockFd, buf, (int)Job::CleanupLayer, networkId, 0);

    // (6) send Halt Msg
    msgHdr.setMsgType(MessageHeader::HaltMachine);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    Serializer::serializeMsgHdr(msgHdr, buf);
    ret = Communicator::sendMessage(sockFd, msgHdr, buf);
    assert(ret == Communicator::Success);

    // XXX: process should wait until send buffer is empty
    // cleanup resrouce & exit
    close(sockFd);
    free(buf);
}
