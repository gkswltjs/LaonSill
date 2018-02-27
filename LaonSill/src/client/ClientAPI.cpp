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
#include "MemoryMgmt.h"

using namespace std;

#define CPPAPI_CHECK_BUFFER(job)                                                        \
    do {                                                                                \
        if (handle.buffer == NULL) {                                                    \
            close(handle.sockFD);                                                       \
            return ClientError::ClientHandleBufferAllocationFailed;                     \
        }                                                                               \
                                                                                        \
        Job* myJob = (Job*) job ;                                                       \
        int bufSize = MessageHeader::MESSAGE_HEADER_SIZE + myJob->getJobSize();         \
        if (bufSize > MessageHeader::MESSAGE_DEFAULT_SIZE) {                            \
            bigBuffer = (char*)malloc(bufSize);                                         \
            if (bigBuffer == NULL) {                                                    \
                close(handle.sockFD);                                                   \
                return ClientError::ClientHandleBufferAllocationFailed;                 \
            }                                                                           \
        }                                                                               \
    } while (0)

#define CPPAPI_CHECK_SEND()                                                             \
    do {                                                                                \
        if (ret != ClientError::Success) {                                              \
            if (bigBuffer != NULL)                                                      \
                free(bigBuffer);                                                        \
            return ClientError::SendJobFailed;                                          \
        }                                                                               \
    } while (0)

#define CPPAPI_CHECK_RECV(job, bufSize)                                                 \
    do {                                                                                \
        if (ret == Communicator::RecvOnlyHeader) {                                      \
            if (bigBuffer)                                                              \
                free(bigBuffer);                                                        \
                                                                                        \
            bigBuffer = (char*)malloc(bufSize);                                         \
                                                                                        \
            if (bigBuffer == NULL) {                                                    \
                close(handle.sockFD);                                                   \
                return ClientError::ClientHandleBufferAllocationFailed;                 \
            }                                                                           \
            ret = Client::recvJob(handle.sockFD, bigBuffer, & job, & bufSize);          \
        } else if (ret != Communicator::Success) {                                      \
            if (bigBuffer != NULL)                                                      \
                free(bigBuffer);                                                        \
            close(handle.sockFD);                                                       \
            return ClientError::RecvJobFailed;                                          \
        }                                                                               \
    } while (0)

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
    ret = Communicator::recvMessage(handle.sockFD, msgHdr, handle.buffer, false);
    if (ret != Communicator::Success)
        return ClientError::RecvMessageFailed;

    SASSERT0(msgHdr.getMsgType() == MessageHeader::WelcomeReply);

    handle.hasSession = true;

    return ClientError::Success;
}

ClientError ClientAPI::releaseSession(ClientHandle handle) {

    ClientError retValue = ClientError::Success;

    if (!handle.hasSession)
        return ClientError::NoSession;

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

    if (handle.buffer != NULL)
        free(handle.buffer);

    return retValue;
}

ClientError ClientAPI::createNetwork(ClientHandle handle, std::string networkDef,
    NetworkHandle& netHandle) {

    if (!handle.hasSession)
        return ClientError::NoSession;

    Job* createNetworkJob = new Job(JobType::CreateNetwork);
    createNetworkJob->addJobElem(Job::StringType, strlen(networkDef.c_str()),
        (void*)networkDef.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(createNetworkJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            createNetworkJob);
    delete createNetworkJob;
    CPPAPI_CHECK_SEND();

    Job* createNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &createNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(createNetworkReplyJob, bufSize);

    SASSERT0(createNetworkReplyJob->getType() == JobType::CreateNetworkReply);
    string networkID = createNetworkReplyJob->getStringValue(0);
    delete createNetworkReplyJob;

    netHandle.networkID = networkID;
    netHandle.created = true;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::createNetworkFromFile(ClientHandle handle,
    std::string filePathInServer, NetworkHandle& netHandle) {

    if (!handle.hasSession)
        return ClientError::NoSession;

    Job* createNetworkFromFileJob = new Job(JobType::CreateNetworkFromFile);
    createNetworkFromFileJob->addJobElem(Job::StringType, strlen(filePathInServer.c_str()),
        (void*)filePathInServer.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(createNetworkFromFileJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            createNetworkFromFileJob);
    delete createNetworkFromFileJob;
    CPPAPI_CHECK_SEND();

    Job* createNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &createNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(createNetworkReplyJob, bufSize);

    SASSERT0(createNetworkReplyJob->getType() == JobType::CreateNetworkReply);
    string networkID = createNetworkReplyJob->getStringValue(0);
    delete createNetworkReplyJob;

    netHandle.networkID = networkID;
    netHandle.created = true;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::destroyNetwork(ClientHandle handle, NetworkHandle& netHandle) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* destroyNetworkJob = new Job(JobType::DestroyNetwork);
    destroyNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
            (void*)netHandle.networkID.c_str());
    int ret = Client::sendJob(handle.sockFD, handle.buffer, destroyNetworkJob);
    delete destroyNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* destroyNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &destroyNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(destroyNetworkReplyJob->getType() == JobType::DestroyNetworkReply);
    delete destroyNetworkReplyJob;

    netHandle.created = false;
    return ClientError::Success;
}

ClientError ClientAPI::buildNetwork(ClientHandle handle, NetworkHandle netHandle,
    int epochs) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* buildNetworkJob = new Job(JobType::BuildNetwork);
    buildNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    buildNetworkJob->addJobElem(Job::IntType, 1, (void*)&epochs);
    int ret = Client::sendJob(handle.sockFD, handle.buffer, buildNetworkJob);
    delete buildNetworkJob;

    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* buildNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &buildNetworkReplyJob, &bufSize);

    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(buildNetworkReplyJob->getType() == JobType::BuildNetworkReply);
    delete buildNetworkReplyJob;

    return ClientError::Success;
}

ClientError ClientAPI::resetNetwork(ClientHandle handle, NetworkHandle netHandle) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* resetNetworkJob = new Job(JobType::ResetNetwork);
    resetNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    int ret = Client::sendJob(handle.sockFD, handle.buffer, resetNetworkJob);
    delete resetNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* resetNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &resetNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(resetNetworkReplyJob->getType() == JobType::ResetNetworkReply);
    delete resetNetworkReplyJob;

    return ClientError::Success;
}

ClientError ClientAPI::runNetwork(ClientHandle handle, NetworkHandle netHandle, 
    bool inference) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runNetworkJob = new Job(JobType::RunNetwork);
    runNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    int inferenceInt = (int)inference;
    runNetworkJob->addJobElem(Job::IntType, 1, (void*)&inferenceInt);
    int ret = Client::sendJob(handle.sockFD, handle.buffer, runNetworkJob);
    delete runNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* runNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &runNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(runNetworkReplyJob->getType() == JobType::RunNetworkReply);
    int success = runNetworkReplyJob->getIntValue(0);
    delete runNetworkReplyJob;

    if (success == 1)
        return ClientError::Success;
    else
        return ClientError::RunNetworkFailed;
}

ClientError ClientAPI::runNetworkMiniBatch(ClientHandle handle, NetworkHandle netHandle,
    bool inference, int miniBatchIdx) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runNetworkJob = new Job(JobType::RunNetworkMiniBatch);
    runNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    int inferenceInt = (int)inference;
    runNetworkJob->addJobElem(Job::IntType, 1, (void*)&inferenceInt);
    runNetworkJob->addJobElem(Job::IntType, 1, (void*)&miniBatchIdx);
    int ret = Client::sendJob(handle.sockFD, handle.buffer, runNetworkJob);
    delete runNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* runNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, handle.buffer, &runNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;
    SASSERT0(runNetworkReplyJob->getType() == JobType::RunNetworkReply);
    int success = runNetworkReplyJob->getIntValue(0);
    delete runNetworkReplyJob;

    if (success == 1)
        return ClientError::Success;
    else
        return ClientError::RunNetworkFailed;
}

ClientError ClientAPI::saveNetwork(ClientHandle handle, NetworkHandle netHandle,
    std::string filePath) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* saveNetworkJob = new Job(JobType::SaveNetwork);
    saveNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    saveNetworkJob->addJobElem(Job::StringType, strlen(filePath.c_str()),
        (void*)filePath.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(saveNetworkJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            saveNetworkJob);
    delete saveNetworkJob;
    CPPAPI_CHECK_SEND();

    Job* saveNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &saveNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(saveNetworkReplyJob, bufSize);

    SASSERT0(saveNetworkReplyJob->getType() == JobType::SaveNetworkReply);
    delete saveNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::loadNetwork(ClientHandle handle, NetworkHandle netHandle,
    std::string filePath) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* loadNetworkJob = new Job(JobType::LoadNetwork);
    loadNetworkJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    loadNetworkJob->addJobElem(Job::StringType, strlen(filePath.c_str()),
        (void*)filePath.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(loadNetworkJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            loadNetworkJob);
    delete loadNetworkJob;
    CPPAPI_CHECK_SEND();

    Job* loadNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &loadNetworkReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(loadNetworkReplyJob, bufSize);

    SASSERT0(loadNetworkReplyJob->getType() == JobType::LoadNetworkReply);
    delete loadNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::getObjectDetection(ClientHandle handle, NetworkHandle netHandle,
    int channel, int height, int width, float* imageData, vector<BoundingBox>& boxArray,
    int coordRelative) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runJob = new Job(JobType::RunNetworkWithInputData);
    runJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    runJob->addJobElem(Job::IntType, 1, (void*)&channel);
    runJob->addJobElem(Job::IntType, 1, (void*)&height);
    runJob->addJobElem(Job::IntType, 1, (void*)&width);
    runJob->addJobElem(Job::IntType, 1, (void*)&coordRelative);

    int imageDataElemCount = channel * height * width;
    runJob->addJobElem(Job::FloatArrayType, imageDataElemCount, imageData);

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
        &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::RunNetworkWithInputDataReply);
    
    int resultBoxCount = runReplyJob->getIntValue(0);
    int elemIdx = 1;
    for (int i = 0; i < resultBoxCount; i++) {
        BoundingBox bbox;
        bbox.top        = runReplyJob->getFloatValue(elemIdx + 0);
        bbox.left       = runReplyJob->getFloatValue(elemIdx + 1);
        bbox.bottom     = runReplyJob->getFloatValue(elemIdx + 2);
        bbox.right      = runReplyJob->getFloatValue(elemIdx + 3);
        bbox.confidence = runReplyJob->getFloatValue(elemIdx + 4);
        bbox.labelIndex = runReplyJob->getIntValue(elemIdx + 5);

        boxArray.push_back(bbox);
        elemIdx += 6;
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::runObjectDetectionWithInput(ClientHandle handle,
    NetworkHandle netHandle, int channel, int height, int width, float* imageData,
    vector<BoundingBox>& boxArray, int baseNetworkType, int needRecovery) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runJob = new Job(JobType::RunObjectDetectionNetworkWithInput);
    runJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    runJob->addJobElem(Job::IntType, 1, (void*)&channel);
    runJob->addJobElem(Job::IntType, 1, (void*)&height);
    runJob->addJobElem(Job::IntType, 1, (void*)&width);
    runJob->addJobElem(Job::IntType, 1, (void*)&baseNetworkType);
    runJob->addJobElem(Job::IntType, 1, (void*)&needRecovery);

    int imageDataElemCount = channel * height * width;
    runJob->addJobElem(Job::FloatArrayType, imageDataElemCount, imageData);

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::RunObjectDetectionNetworkWithInputReply);
    
    int resultBoxCount = runReplyJob->getIntValue(0);

    if (needRecovery && (resultBoxCount == -1)) {
        delete runReplyJob;

        if (bigBuffer != NULL)
            free(bigBuffer);

        return ClientError::RunAdhocNetworkFailed;
    }

    int elemIdx = 1;
    for (int i = 0; i < resultBoxCount; i++) {
        BoundingBox bbox;
        bbox.top        = runReplyJob->getFloatValue(elemIdx + 0);
        bbox.left       = runReplyJob->getFloatValue(elemIdx + 1);
        bbox.bottom     = runReplyJob->getFloatValue(elemIdx + 2);
        bbox.right      = runReplyJob->getFloatValue(elemIdx + 3);
        bbox.confidence = runReplyJob->getFloatValue(elemIdx + 4);
        bbox.labelIndex = runReplyJob->getIntValue(elemIdx + 5);

        boxArray.push_back(bbox);
        elemIdx += 6;
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

ClientError ClientAPI::runClassificationWithInput(ClientHandle handle,
    NetworkHandle netHandle, int channel, int height, int width, float* imageData,
    vector<int>& labelIndexArray, int baseNetworkType, int needRecovery) {
    if (!netHandle.created) 
        return ClientError::NotCreatedNetwork;

    Job* runJob = new Job(JobType::RunClassificationNetworkWithInput);
    runJob->addJobElem(Job::StringType, strlen(netHandle.networkID.c_str()),
        (void*)netHandle.networkID.c_str());
    runJob->addJobElem(Job::IntType, 1, (void*)&channel);
    runJob->addJobElem(Job::IntType, 1, (void*)&height);
    runJob->addJobElem(Job::IntType, 1, (void*)&width);
    runJob->addJobElem(Job::IntType, 1, (void*)&baseNetworkType);
    runJob->addJobElem(Job::IntType, 1, (void*)&needRecovery);

    int imageDataElemCount = channel * height * width;
    runJob->addJobElem(Job::FloatArrayType, imageDataElemCount, imageData);

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::RunClassificationNetworkWithInputReply);
    
    int labelIndex = runReplyJob->getIntValue(0);
    labelIndexArray.push_back(labelIndex);
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (needRecovery && (labelIndex == -1))
        ClientError::RunAdhocNetworkFailed;

    return ClientError::Success;
}

ClientError ClientAPI::getMeasureItemName(ClientHandle handle, string networkID,
    vector<string>& measureItemNames) {

    Job* runJob = new Job(JobType::GetMeasureItemName);
    runJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::GetMeasureItemNameReply);

    int resultItemCount = runReplyJob->getIntValue(0);
    for (int i = 0; i < resultItemCount; i++) {
        string itemName = runReplyJob->getStringValue(i + 1);
        measureItemNames.push_back(itemName);
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (resultItemCount == -1)
        return ClientError::RequestedNetworkNotExist;

    return ClientError::Success;
}

ClientError ClientAPI::getMeasures(ClientHandle handle, string networkID,
    bool forwardSearch, int start, int count, int* startIterNum, int* dataCount,
    int* curIterNum, int* totalIterNum, float* data) {

    int forward = (int)forwardSearch;

    Job* runJob = new Job(JobType::GetMeasures);
    runJob->addJobElem(Job::StringType, strlen(networkID.c_str()), (void*)networkID.c_str());
    runJob->addJobElem(Job::IntType, 1, (void*)&forward);
    runJob->addJobElem(Job::IntType, 1, (void*)&start);
    runJob->addJobElem(Job::IntType, 1, (void*)&count);

    char* bigBuffer = NULL;
    CPPAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer, runJob);
    delete runJob;
    CPPAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(handle.sockFD, bigBuffer ? bigBuffer : handle.buffer,
            &runReplyJob, &bufSize);
    CPPAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::GetMeasuresReply);

    int measureCount = runReplyJob->getIntValue(0);
    (*dataCount) = measureCount;
    (*startIterNum) = runReplyJob->getIntValue(1);
    (*curIterNum) = runReplyJob->getIntValue(2);
    (*totalIterNum) = runReplyJob->getIntValue(3);

    if ((*dataCount) > 0) {
        float *measures = runReplyJob->getFloatArray(4);
        memcpy(data, measures, sizeof(float) * measureCount);
    }

    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (measureCount == -1)
        return ClientError::RequestedNetworkNotExist;

    return ClientError::Success;
}
