/**
 * @file ClientAPI.h
 * @date 2017-06-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef CLIENTAPI_H
#define CLIENTAPI_H 

#include <string>
#include <vector>

#define MAX_SERVER_HOSTNAME_LEN             (1024)

typedef struct ClientHandle_s {
    int     sockFD;
    char    serverHostName[MAX_SERVER_HOSTNAME_LEN];
    int     serverPortNum;
    char*   buffer;
    int     bufLen;
    bool    hasSession;

    ClientHandle_s() {
        buffer = NULL;
        bufLen = 0;
        hasSession = false;
    }
} ClientHandle;

typedef enum ClientError_s {
    Success = 0,
    TooLongServerHostName,
    NoSuchHost,
    HostConnectionFailed,
    ClientHandleBufferAllocationFailed,
    ClientHandleBufferReallocationFailed,
    ClientHandleInternalMemoryAllocationFailed,
    SendMessageFailed,
    RecvMessageFailed,
    HaveSessionAlready,
    NoSession,
    NotCreatedNetwork,
    RunNetworkFailed,
    RequestedNetworkNotExist,
    SessErrorMax
} ClientError;

typedef struct NetworkHandle_s {
    std::string     networkID;
    bool            created;

    NetworkHandle_s() {
        networkID = "";
        created = false;
    }
} NetworkHandle;

typedef struct BoundingBox_s {
    float top;
    float left;
    float bottom;
    float right;
    float confidence;
} BoundingBox;

class ClientAPI {
public: 
                        ClientAPI() {}
    virtual            ~ClientAPI() {}

    static ClientError      createHandle(ClientHandle& handle, std::string serverHostName,
                                         int serverPortNum);


    static ClientError      getSession(ClientHandle& handle);
    static ClientError      releaseSession(ClientHandle handle);

    static ClientError      createNetwork(ClientHandle handle, std::string networkDef,
                                          NetworkHandle& netHandle);
    static ClientError      createNetworkFromFile(ClientHandle handle,
                                                  std::string filePathInServer,
                                                  NetworkHandle& netHandle);
    static ClientError      destroyNetwork(ClientHandle handle, NetworkHandle& netHandle);
    static ClientError      buildNetwork(ClientHandle handle, NetworkHandle netHandle,
                                         int epochs);
    static ClientError      resetNetwork(ClientHandle handle, NetworkHandle netHandle);
    static ClientError      runNetwork(ClientHandle handle, NetworkHandle netHandle,
                                       bool inference);
    static ClientError      runNetworkMiniBatch(ClientHandle handle, NetworkHandle netHandle,
                                                bool inference, int miniBatchIdx);
    static ClientError      saveNetwork(ClientHandle handle, NetworkHandle netHandle,
                                        std::string filePath);
    static ClientError      loadNetwork(ClientHandle handle, NetworkHandle netHandle,
                                        std::string filePath);

    static ClientError      getObjectDetection(ClientHandle handle, NetworkHandle netHandle,
                                int channel, int height, int width, float* imageData,
                                std::vector<BoundingBox>& boxArray, int coordRelative=0);

    static ClientError      getMeasureItemName(ClientHandle handle, std::string networkID,
                                std::vector<std::string>& measureItemNames);  

    static ClientError      getMeasures(ClientHandle handle, std::string networkID,
                                bool forwardSearch, int start, int count, 
                                int* startIterNum, int* dataCount, float* data);
};
                            
#endif /* CLIENTAPI_H */
