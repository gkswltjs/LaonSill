/**
 * @file CapiClientAPI.cpp
 * @date 2017-09-27
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
#include "CapiClientAPI.h"

using namespace std;

#define CAPI_CHECK_BUFFER(job)                                                          \
    do {                                                                                \
        if (buffer == NULL) {                                                           \
            close(sockFD);                                                              \
            return ClientError::ClientHandleBufferAllocationFailed;                     \
        }                                                                               \
                                                                                        \
        Job* myJob = (Job*) job ;                                                       \
        int bufSize = MessageHeader::MESSAGE_HEADER_SIZE + myJob->getJobSize();         \
        if (bufSize > MessageHeader::MESSAGE_DEFAULT_SIZE) {                            \
            bigBuffer = (char*)malloc(bufSize);                                         \
            if (bigBuffer == NULL) {                                                    \
                close(sockFD);                                                          \
                return ClientError::ClientHandleBufferAllocationFailed;                 \
            }                                                                           \
        }                                                                               \
    } while (0)

#define CAPI_CHECK_SEND()                                                               \
    do {                                                                                \
        if (ret != ClientError::Success) {                                              \
            if (bigBuffer != NULL)                                                      \
                free(bigBuffer);                                                        \
            return ClientError::SendJobFailed;                                          \
        }                                                                               \
    } while (0)

#define CAPI_CHECK_RECV(job, bufSize)                                                   \
    do {                                                                                \
        if (ret == Communicator::RecvOnlyHeader) {                                      \
            if (bigBuffer)                                                              \
                free(bigBuffer);                                                        \
                                                                                        \
            bigBuffer = (char*)malloc(bufSize);                                         \
                                                                                        \
            if (bigBuffer == NULL) {                                                    \
                close(sockFD);                                                          \
                return ClientError::ClientHandleBufferAllocationFailed;                 \
            }                                                                           \
            ret = Client::recvJob(sockFD, bigBuffer, & job, & bufSize);                 \
        } else if (ret != Communicator::Success) {                                      \
            if (bigBuffer != NULL)                                                      \
                free(bigBuffer);                                                        \
            close(sockFD);                                                              \
            return ClientError::RecvJobFailed;                                          \
        }                                                                               \
    } while (0)

extern "C" int testYo(int a, char* name, float b) {
    printf("name : %s\n", name);
    cout << "name yo : " << name << endl;
    return a * int(b);
}

extern "C" int getSession(int *hasSession, char* serverHostName, int serverPortNum,
    int *sockFD, char* buffer) { 
    if (*hasSession)
        return ClientError::HaveSessionAlready;

    // (1) get server info (struct hostent)
    struct hostent *server;
    server = gethostbyname(serverHostName);
    if (server == NULL) {
        return ClientError::NoSuchHost;
    }

    // (2) create socket & connect to the server
    (*sockFD) = socket(AF_INET, SOCK_STREAM, 0);
    SASSERT0((*sockFD) != -1);
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(struct sockaddr_in));
    serverAddr.sin_family = AF_INET;
    memcpy((char*)&serverAddr.sin_addr.s_addr, (char*)server->h_addr, server->h_length);
    serverAddr.sin_port = htons(serverPortNum);

    int connectRetry = Client::connectRetry((*sockFD), (struct sockaddr*)&serverAddr,
                                            sizeof(serverAddr));
    if (connectRetry == -1) {
        return ClientError::HostConnectionFailed;
    }

    // (3-1) send welcome msg
    if (buffer == NULL) {
        close((*sockFD));
        return ClientError::ClientHandleBufferAllocationFailed;
    }

    MessageHeader msgHdr;
    msgHdr.setMsgType(MessageHeader::Welcome);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    MsgSerializer::serializeMsgHdr(msgHdr, buffer);
    Communicator::CommRetType ret = Communicator::sendMessage((*sockFD), msgHdr, buffer);
    if (ret != Communicator::Success) {
        return ClientError::SendMessageFailed;
    }

    // (3-2) recv welcome reply msg
    ret = Communicator::recvMessage((*sockFD), msgHdr, buffer, false);
    if (ret != Communicator::Success)
        return Communicator::RecvFailed;

    SASSERT0(msgHdr.getMsgType() == MessageHeader::WelcomeReply);

    (*hasSession) = 1;  //true

    return int(ClientError::Success);
}

extern "C" int releaseSession(int sockFD, char* buffer, int* hasSession) {
    ClientError retValue = ClientError::Success;

    if (!(*hasSession))
        return ClientError::NoSession;

    MessageHeader msgHdr;
    msgHdr.setMsgType(MessageHeader::GoodBye);
    msgHdr.setMsgLen(MessageHeader::MESSAGE_HEADER_SIZE);
    MsgSerializer::serializeMsgHdr(msgHdr, buffer);

    Communicator::CommRetType ret;
    ret = Communicator::sendMessage(sockFD, msgHdr, buffer);
    if (ret != Communicator::Success) {
        retValue = ClientError::SendMessageFailed;
    }

    close(sockFD);
    (*hasSession) = 0; // false

    return retValue;
}

extern "C" int createNetwork(int sockFD, int hasSession, char* buffer, char* networkDef,
    char* networkID) {

    if (!hasSession)
        return ClientError::NoSession;

    Job* createNetworkJob = new Job(JobType::CreateNetwork);
    createNetworkJob->addJobElem(Job::StringType, strlen(networkDef), (void*)networkDef);

    char* bigBuffer = NULL;
    CAPI_CHECK_BUFFER(createNetworkJob);

    int ret = Client::sendJob(sockFD, bigBuffer ? bigBuffer : buffer, createNetworkJob);
    delete createNetworkJob;
    CAPI_CHECK_SEND();

    Job* createNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, bigBuffer ? bigBuffer : buffer, &createNetworkReplyJob,
            &bufSize);
    CAPI_CHECK_RECV(createNetworkReplyJob, bufSize);

    SASSERT0(createNetworkReplyJob->getType() == JobType::CreateNetworkReply);
    strcpy(networkID, createNetworkReplyJob->getStringValue(0).c_str());
    delete createNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

extern "C" int createNetworkFromFile(int sockFD, int hasSession, char* buffer,
    char* filePathInServer, char *networkID) {

    if (!hasSession)
        return ClientError::NoSession;

    Job* createNetworkFromFileJob = new Job(JobType::CreateNetworkFromFile);
    createNetworkFromFileJob->addJobElem(Job::StringType, strlen(filePathInServer),
        (void*)filePathInServer);

    if (buffer == NULL) {
        close(sockFD);
        return ClientError::ClientHandleBufferAllocationFailed;
    }

    char* bigBuffer = NULL;
    CAPI_CHECK_BUFFER(createNetworkFromFileJob);

    int ret = Client::sendJob(sockFD, bigBuffer ? bigBuffer : buffer, createNetworkFromFileJob);
    delete createNetworkFromFileJob;
    CAPI_CHECK_SEND();

    Job* createNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, bigBuffer ? bigBuffer : buffer, &createNetworkReplyJob,
            &bufSize);
    CAPI_CHECK_RECV(createNetworkReplyJob, bufSize);

    SASSERT0(createNetworkReplyJob->getType() == JobType::CreateNetworkReply);
    strcpy(networkID, createNetworkReplyJob->getStringValue(0).c_str());
    delete createNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

extern "C" int destroyNetwork(int sockFD, char* buffer, int isCreated, char *networkID) {
    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* destroyNetworkJob = new Job(JobType::DestroyNetwork);
    destroyNetworkJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    int ret = Client::sendJob(sockFD, buffer, destroyNetworkJob);
    delete destroyNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* destroyNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, buffer, &destroyNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;

    SASSERT0(destroyNetworkReplyJob->getType() == JobType::DestroyNetworkReply);
    delete destroyNetworkReplyJob;

    return ClientError::Success;
}

extern "C" int buildNetwork(int sockFD, char* buffer, int isCreated, char *networkID,
        int epochs) {
    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* buildNetworkJob = new Job(JobType::BuildNetwork);
    buildNetworkJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    buildNetworkJob->addJobElem(Job::IntType, 1, (void*)&epochs);
    int ret = Client::sendJob(sockFD, buffer, buildNetworkJob);
    delete buildNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* buildNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, buffer, &buildNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;

    SASSERT0(buildNetworkReplyJob->getType() == JobType::BuildNetworkReply);
    delete buildNetworkReplyJob;

    return ClientError::Success;
}

extern "C" int resetNetwork(int sockFD, char* buffer, int isCreated, char *networkID) {
    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* resetNetworkJob = new Job(JobType::ResetNetwork);
    resetNetworkJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    int ret = Client::sendJob(sockFD, buffer, resetNetworkJob);
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    delete resetNetworkJob;

    Job* resetNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, buffer, &resetNetworkReplyJob, &bufSize);
    if (ret != ClientError::Success)
        return ClientError::RecvJobFailed;

    SASSERT0(resetNetworkReplyJob->getType() == JobType::ResetNetworkReply);
    delete resetNetworkReplyJob;

    return ClientError::Success;
}

extern "C" int runNetwork(int sockFD, char* buffer, int isCreated, char* networkID,
        int inference) {
    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* runNetworkJob = new Job(JobType::RunNetwork);
    runNetworkJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    runNetworkJob->addJobElem(Job::IntType, 1, (void*)&inference);
    int ret = Client::sendJob(sockFD, buffer, runNetworkJob);
    delete runNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* runNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, buffer, &runNetworkReplyJob, &bufSize);
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

extern "C" int runNetworkMiniBatch(int sockFD, char* buffer, int isCreated, char* networkID,
    int inference, int miniBatchIdx) {

    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* runNetworkJob = new Job(JobType::RunNetworkMiniBatch);
    runNetworkJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    runNetworkJob->addJobElem(Job::IntType, 1, (void*)&inference);
    runNetworkJob->addJobElem(Job::IntType, 1, (void*)&miniBatchIdx);
    int ret = Client::sendJob(sockFD, buffer, runNetworkJob);
    delete runNetworkJob;
    if (ret != ClientError::Success)
        return ClientError::SendJobFailed;

    Job* runNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, buffer, &runNetworkReplyJob, &bufSize);
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

extern "C" int saveNetwork(int sockFD, char* buffer, int isCreated, char* networkID,
        char* filePath) {
    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* saveNetworkJob = new Job(JobType::SaveNetwork);
    saveNetworkJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    saveNetworkJob->addJobElem(Job::StringType, strlen(filePath), (void*)filePath);

    char* bigBuffer = NULL;
    CAPI_CHECK_BUFFER(saveNetworkJob);

    int ret = Client::sendJob(sockFD, bigBuffer ? bigBuffer : buffer, saveNetworkJob);
    delete saveNetworkJob;
    CAPI_CHECK_SEND();

    Job* saveNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, bigBuffer ? bigBuffer : buffer, &saveNetworkReplyJob,
            &bufSize);
    CAPI_CHECK_RECV(saveNetworkReplyJob, bufSize);

    SASSERT0(saveNetworkReplyJob->getType() == JobType::SaveNetworkReply);
    delete saveNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

extern "C" int loadNetwork(int sockFD, char* buffer, int isCreated, char* networkID,
    char* filePath) {
    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* loadNetworkJob = new Job(JobType::LoadNetwork);
    loadNetworkJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    loadNetworkJob->addJobElem(Job::StringType, strlen(filePath), (void*)filePath);

    char* bigBuffer = NULL;
    CAPI_CHECK_BUFFER(loadNetworkJob);

    int ret = Client::sendJob(sockFD, bigBuffer ? bigBuffer : buffer, loadNetworkJob);
    delete loadNetworkJob;
    CAPI_CHECK_SEND();

    Job* loadNetworkReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, bigBuffer ? bigBuffer : buffer, &loadNetworkReplyJob,
            &bufSize);
    CAPI_CHECK_RECV(loadNetworkReplyJob, bufSize);

    SASSERT0(loadNetworkReplyJob->getType() == JobType::LoadNetworkReply);
    delete loadNetworkReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

extern "C" int getObjectDetection(int sockFD, char* buffer, int isCreated, char* networkID,
    int channel, int height, int width, float* imageData, BoundingBox* boxArray,
    int maxBoxCount, int coordRelative) {
    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* runJob = new Job(JobType::RunNetworkWithInputData);
    runJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    runJob->addJobElem(Job::IntType, 1, (void*)&channel);
    runJob->addJobElem(Job::IntType, 1, (void*)&height);
    runJob->addJobElem(Job::IntType, 1, (void*)&width);
    runJob->addJobElem(Job::IntType, 1, (void*)&coordRelative);

    int imageDataElemCount = channel * height * width;
    runJob->addJobElem(Job::FloatArrayType, imageDataElemCount, imageData);

    char* bigBuffer = NULL;
    CAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(sockFD, bigBuffer ? bigBuffer : buffer, runJob);
    delete runJob;
    CAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, bigBuffer ? bigBuffer : buffer, &runReplyJob, &bufSize);
    CAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::RunNetworkWithInputDataReply);
    
    int resultBoxCount = runReplyJob->getIntValue(0);
    int elemIdx = 1;
    for (int i = 0; i < resultBoxCount; i++) {
        if (i == maxBoxCount)
            break;

        BoundingBox bbox;
        bbox.top        = runReplyJob->getFloatValue(elemIdx + 0);
        bbox.left       = runReplyJob->getFloatValue(elemIdx + 1);
        bbox.bottom     = runReplyJob->getFloatValue(elemIdx + 2);
        bbox.right      = runReplyJob->getFloatValue(elemIdx + 3);
        bbox.confidence = runReplyJob->getFloatValue(elemIdx + 4);
        bbox.labelIndex = runReplyJob->getIntValue(elemIdx + 5);

        boxArray[i] = bbox;
        elemIdx += 6;
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

extern "C" int runObjectDetectionWithInput(int sockFD, char* buffer, int isCreated,
    char* networkID, int channel, int height, int width, float* imageData,
    BoundingBox* boxArray, int maxBoxCount, int networkType, int needRecovery) {
    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* runJob = new Job(JobType::RunObjectDetectionNetworkWithInput);
    runJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    runJob->addJobElem(Job::IntType, 1, (void*)&channel);
    runJob->addJobElem(Job::IntType, 1, (void*)&height);
    runJob->addJobElem(Job::IntType, 1, (void*)&width);
    runJob->addJobElem(Job::IntType, 1, (void*)&networkType);
    runJob->addJobElem(Job::IntType, 1, (void*)&needRecovery);

    int imageDataElemCount = channel * height * width;
    runJob->addJobElem(Job::FloatArrayType, imageDataElemCount, imageData);

    char* bigBuffer = NULL;
    CAPI_CHECK_BUFFER(runJob);
    int ret = Client::sendJob(sockFD, bigBuffer ? bigBuffer : buffer, runJob);
    delete runJob;
    CAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, bigBuffer ? bigBuffer : buffer, &runReplyJob, &bufSize);
    CAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::RunObjectDetectionNetworkWithInputReply);

    int resultBoxCount = runReplyJob->getIntValue(0);

    if ((resultBoxCount == -1) && needRecovery) {
        delete runReplyJob;

        if (bigBuffer != NULL)
            free(bigBuffer);

        return ClientError::RunAdhocNetworkFailed;
    }


    int elemIdx = 1;
    for (int i = 0; i < resultBoxCount; i++) {
        if (i == maxBoxCount)
            break;

        BoundingBox bbox;
        bbox.top        = runReplyJob->getFloatValue(elemIdx + 0);
        bbox.left       = runReplyJob->getFloatValue(elemIdx + 1);
        bbox.bottom     = runReplyJob->getFloatValue(elemIdx + 2);
        bbox.right      = runReplyJob->getFloatValue(elemIdx + 3);
        bbox.confidence = runReplyJob->getFloatValue(elemIdx + 4);
        bbox.labelIndex = runReplyJob->getIntValue(elemIdx + 5);

        boxArray[i] = bbox;
        elemIdx += 6;
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    return ClientError::Success;
}

extern "C" int runClassificationWithInput(int sockFD, char* buffer, int isCreated,
    char* networkID, int channel, int height, int width, float* imageData,
    int networkType, int needRecovery, int* labelIndex) {

    if (!isCreated) 
        return ClientError::NotCreatedNetwork;

    Job* runJob = new Job(JobType::RunClassificationNetworkWithInput);
    runJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    runJob->addJobElem(Job::IntType, 1, (void*)&channel);
    runJob->addJobElem(Job::IntType, 1, (void*)&height);
    runJob->addJobElem(Job::IntType, 1, (void*)&width);
    runJob->addJobElem(Job::IntType, 1, (void*)&networkType);
    runJob->addJobElem(Job::IntType, 1, (void*)&needRecovery);

    int imageDataElemCount = channel * height * width;
    runJob->addJobElem(Job::FloatArrayType, imageDataElemCount, imageData);

    char* bigBuffer = NULL;
    CAPI_CHECK_BUFFER(runJob);

    int ret = Client::sendJob(sockFD, bigBuffer ? bigBuffer : buffer, runJob);
    delete runJob;
    
    CAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, bigBuffer ? bigBuffer : buffer, &runReplyJob, &bufSize);
    
    CAPI_CHECK_RECV(runReplyJob, bufSize);

    SASSERT0(runReplyJob->getType() == JobType::RunClassificationNetworkWithInputReply);
    
    (*labelIndex) = runReplyJob->getIntValue(0);
    
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (needRecovery && (*labelIndex) == -1)
        return ClientError::RunAdhocNetworkFailed;

    return ClientError::Success;
}

extern "C" int getMeasureItemName(int sockFD, char* buffer, char* networkID,
        int maxItemCount, char** measureItemNames, int* measureItemCount) {
    
    Job* runJob = new Job(JobType::GetMeasureItemName);
    runJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);

    char* bigBuffer = NULL;
    CAPI_CHECK_BUFFER(runJob);
    int ret = Client::sendJob(sockFD, bigBuffer ? bigBuffer : buffer, runJob);
    delete runJob;
    CAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, bigBuffer ? bigBuffer : buffer, &runReplyJob, &bufSize);
    CAPI_CHECK_RECV(runReplyJob, bufSize);
    SASSERT0(runReplyJob->getType() == JobType::GetMeasureItemNameReply);

    int resultItemCount = runReplyJob->getIntValue(0);

    if (resultItemCount == -1)
        (*measureItemCount) = 0;
    else if (resultItemCount > maxItemCount)
        (*measureItemCount) = maxItemCount;
    else
        (*measureItemCount) = resultItemCount;

    for (int i = 0; i < (*measureItemCount); i++) {

        string itemName = runReplyJob->getStringValue(i + 1);
        strcpy(measureItemNames[i], itemName.c_str());
    }
    delete runReplyJob;

    if (bigBuffer != NULL)
        free(bigBuffer);

    if (resultItemCount == -1)
        return ClientError::RequestedNetworkNotExist;

    return ClientError::Success;
}

extern "C" int getMeasures(int sockFD, char* buffer, char* networkID, int forwardSearch, 
        int start, int count, int* startIterNum, int* dataCount, int* curIterNum, 
        int* totalIterNum, float* data) {

    Job* runJob = new Job(JobType::GetMeasures);
    runJob->addJobElem(Job::StringType, strlen(networkID), (void*)networkID);
    runJob->addJobElem(Job::IntType, 1, (void*)&forwardSearch);
    runJob->addJobElem(Job::IntType, 1, (void*)&start);
    runJob->addJobElem(Job::IntType, 1, (void*)&count);

    char* bigBuffer = NULL;
    CAPI_CHECK_BUFFER(runJob);
    int ret = Client::sendJob(sockFD, bigBuffer ? bigBuffer : buffer, runJob);
    delete runJob;
    CAPI_CHECK_SEND();

    Job* runReplyJob;
    int bufSize = 0;
    ret = Client::recvJob(sockFD, bigBuffer ? bigBuffer : buffer, &runReplyJob, &bufSize);
    CAPI_CHECK_RECV(runReplyJob, bufSize);
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
