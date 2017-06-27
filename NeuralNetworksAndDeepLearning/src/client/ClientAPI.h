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
    SessErrorMax
} ClientError;

typedef struct NetworkHandle_s {
    int     networkID;
    bool    created;

    NetworkHandle_s() {
        networkID = -1;
        created = false;
    }
} NetworkHandle;

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
};
#endif /* CLIENTAPI_H */