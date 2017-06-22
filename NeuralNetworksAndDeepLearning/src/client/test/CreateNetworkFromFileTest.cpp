/**
 * @file CreateNetworkFromFileTest.cpp
 * @date 2017-06-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string>
#include <iostream>

#include "CreateNetworkFromFileTest.h"
#include "common.h"
#include "StdOutLog.h"
#include "ClientAPI.h"
#include "Communicator.h"
#include "SysLog.h"

using namespace std;

#define NETWORK_FILEPATH       ("../src/plan/test/network.conf.test")

bool CreateNetworkFromFileTest::runSimpleTest() {
    ClientError ret;
    ClientHandle handle;
    NetworkHandle netHandle;

    ret = ClientAPI::createHandle(handle, "localhost", Communicator::LISTENER_PORT); 
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::getSession(handle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::createNetworkFromFile(handle, NETWORK_FILEPATH, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::destroyNetwork(handle, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    return true;
}

bool CreateNetworkFromFileTest::runTest() {
    bool result = runSimpleTest();
    if (result) {
        STDOUT_LOG("*  - simple create network from file test is success");
    } else {
        STDOUT_LOG("*  - simple create network from file test is failed");
        return false;
    }

    return true;
}
