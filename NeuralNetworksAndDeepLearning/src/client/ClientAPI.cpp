/**
 * @file ClientAPI.cpp
 * @date 2017-06-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <stdlib.h>
#include <errno.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>

#include "ClientAPI.h"
#include "Communicator.h"
#include "MsgSerializer.h"
#include "MessageHeader.h"
#include "SysLog.h"
#include "Client.h"

using namespace std;

// FIXME: 중복된 코드는 모아서 함수로 정의해서 사용하자.

ClientError ClientAPI::createHandle(ClientHandle& handle, std::string serverHostName,
    int serverPortNum) {

    if (strlen(serverHostName.c_str()) >= MAX_SERVER_HOSTNAME_LEN) {
        return ClientError::TooLongServerHostName;
    }

    handle.sockFD = 0;
    strcpy(handle.serverHostName, serverHostName.c_str());
    handle.serverPortNum = serverPortNum;
    handle.hasSession = false;

    return ClientError::Success;
}

ClientError ClientAPI::getSession(ClientHandle &handle) {
    if (handle.hasSession)
        return ClientError::HaveSessionAlready;

    // (1) get server info (struct hostent)
    struct hostent *server;
    server = gethostbyname(handle.serverHostName);
    if (server == NULL) {
        return ClientError::NoSuchHost;
    }

    // (2) create socket & connect to the server
    handle.sockFD = socket(AF_INET, SOCK_STREAM, 0);
    SASSERT0(handle.sockFD != -1);
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    memcpy((char*)&serverAddr.sin_addr.s_addr, (char*)server->h_addr, server->h_length);
    serverAddr.sin_port = htons(handle.serverPortNum);

    int connectRetry = Client::connectRetry(handle.sockFD, (struct sockaddr*)&serverAddr,
                                            sizeof(serverAddr));
    if (connectRetry == -1) {
        return ClientError::HostConnectionFailed;
    }

    // (3-1) send welcome msg
    cout << "send welcome msg" << endl;
    handle.buffer = (char*)malloc(MessageHeader::MESSAGE_DEFAULT_SIZE);
    if (handle.buffer == NULL) {
        close(handle.sockFD);
        return ClientError::ClientHandleBufferAllocationFailed;
    }
    handle.bufLen = MessageHeader::MESSAGE_DEFAULT_SIZE;

    MessageHeader msgHdr;
    msgHdr.setMsgType(MessageHeader::Welcome);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    MsgSerializer::serializeMsgHdr(msgHdr, handle.buffer);
    Communicator::CommRetType ret = Communicator::sendMessage(handle.sockFD, msgHdr,
        handle.buffer);
    if (ret != Communicator::Success) {
        return ClientError::SendMessageFailed;
    }

    // (3-2) recv welcome reply msg
    cout << "recv welcome msg" << endl;
    ret = Communicator::recvMessage(handle.sockFD, msgHdr, handle.buffer, false);
    SASSERT0(ret == Communicator::Success);
    SASSERT0(msgHdr.getMsgType() == MessageHeader::WelcomeReply);

    handle.hasSession = true;

    return ClientError::Success;
}

ClientError ClientAPI::releaseSession(ClientHandle handle) {

    ClientError retValue = ClientError::Success;

    if (!handle.hasSession)
        return ClientError::NoSession;

    cout << "send goodbye msg" << endl;
    MessageHeader msgHdr;
    msgHdr.setMsgType(MessageHeader::GoodBye);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    MsgSerializer::serializeMsgHdr(msgHdr, handle.buffer);

    Communicator::CommRetType ret;
    ret = Communicator::sendMessage(handle.sockFD, msgHdr, handle.buffer);
    if (ret != Communicator::Success) {
        retValue = ClientError::SendMessageFailed;
    }

    close(handle.sockFD);
    handle.hasSession = false;

    return retValue;
}

ClientError ClientAPI::createNetwork(ClientHandle handle, std::string networkDef,
    NetworkHandle& netHandle) {

    if (!handle.hasSession)
        return ClientError::NoSession;

    cout << "send create-network job" << endl;
    Job* createNetworkJob = new Job(JobType::CreateNetwork);
    createNetworkJob->addJobElem(Job::StringType, strlen(networkDef.c_str()),
        (void*)networkDef.c_str());
    Client::sendJob(handle.sockFD, handle.buffer, createNetworkJob);
    delete createNetworkJob;

    Job* createNetworkReplyJob;
    Client::recvJob(handle.sockFD, handle.buffer, &createNetworkReplyJob);
    SASSERT0(createNetworkReplyJob->getType() == JobType::CreateNetworkReply);
    int networkID = createNetworkReplyJob->getIntValue(0);
    cout << "created network ID : " << networkID << endl;
    delete createNetworkReplyJob;

    return ClientError::Success;
}

ClientError ClientAPI::createNetworkFromFile(ClientHandle handle,
    std::string filePathInServer, NetworkHandle& netHandle) {

    if (!handle.hasSession)
        return ClientError::NoSession;

    cout << "send create-network-from-file job" << endl;
    Job* createNetworkFromFileJob = new Job(JobType::CreateNetworkFromFile);
    createNetworkFromFileJob->addJobElem(Job::StringType, strlen(filePathInServer.c_str()),
        (void*)filePathInServer.c_str());
    Client::sendJob(handle.sockFD, handle.buffer, createNetworkFromFileJob);
    delete createNetworkFromFileJob;

    Job* createNetworkReplyJob;
    Client::recvJob(handle.sockFD, handle.buffer, &createNetworkReplyJob);
    SASSERT0(createNetworkReplyJob->getType() == JobType::CreateNetworkReply);
    int networkID = createNetworkReplyJob->getIntValue(0);
    cout << "created network ID : " << networkID << endl;
    netHandle.networkID = networkID;
    delete createNetworkReplyJob;

    return ClientError::Success;
}
