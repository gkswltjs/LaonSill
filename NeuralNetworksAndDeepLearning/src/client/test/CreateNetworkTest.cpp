/**
 * @file CreateNetworkTest.cpp
 * @date 2017-06-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <fstream>

#include "CreateNetworkTest.h"
#include "common.h"
#include "StdOutLog.h"
#include "ClientAPI.h"
#include "Communicator.h"
#include "SysLog.h"

using namespace std;

#define NETWORK_FILEPATH       ("../src/plan/test/network.conf.test")

bool CreateNetworkTest::runSimpleTest() {
    ifstream ifs(NETWORK_FILEPATH);
    string content((istreambuf_iterator<char>(ifs)), (istreambuf_iterator<char>()));

    ClientError ret;
    ClientHandle handle;
    NetworkHandle netHandle;

    ret = ClientAPI::createHandle(handle, "localhost", Communicator::LISTENER_PORT); 
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::getSession(handle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::createNetwork(handle, content, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    return true;
}

bool CreateNetworkTest::runTest() {
    bool result = runSimpleTest();
    if (result) {
        STDOUT_LOG("*  - simple create network test is success");
    } else {
        STDOUT_LOG("*  - simple create network test is failed");
        return false;
    }

    return true;
}
