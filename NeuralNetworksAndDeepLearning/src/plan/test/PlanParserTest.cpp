/**
 * @file PlanParserTest.cpp
 * @date 2017-05-12
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include "PlanParserTest.h"
#include "PlanParser.h"
#include "common.h"
#include "StdOutLog.h"

using namespace std;

#define PLAN_PARSER_TEST_NETWORK_FILEPATH       ("../src/plan/test/network.conf.test")

bool PlanParserTest::runParseTest() {
    PlanParser::loadNetwork(string(PLAN_PARSER_TEST_NETWORK_FILEPATH));

    return true;
}

bool PlanParserTest::runTest() {
    bool result = runParseTest();
    if (result) {
        STDOUT_LOG("*  - simple plan parse test is success");
    } else {
        STDOUT_LOG("*  - simple plan parse test is failed");
        return false;
    }

    return true;
}
