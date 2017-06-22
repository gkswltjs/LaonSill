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
    SessErrorMax
} ClientError;

typedef struct NetworkHandle_s {
    int networkID;
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
};
#endif /* CLIENTAPI_H */
