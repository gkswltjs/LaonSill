/**
 * @file PlanParser.cpp
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#include "jsoncpp/json/json.h"
#include "PlanParser.h"
#include "SysLog.h"

using namespace std;

int PlanParser::loadNetwork(string filePath) {

    filebuf fb;
    if (fb.open(filePath.c_str(), ios::in) == NULL) {
        SASSERT(false, "cannot open cluster confifuration file. file path=%s",
            filePath.c_str());
    }

    Json::Value rootValue;
    istream is(&fb);
    Json::Reader reader;
    bool parse = reader.parse(is, rootValue);

    if (!parse) {
        SASSERT(false, "invalid json-format file. file path=%s. error message=%s",
            filePath.c_str(), reader.getFormattedErrorMessages().c_str());
    }

    Json::Value layerList = rootValue["layers"];
    for (int i = 0; i < layerList.size(); i++) {
        Json::Value layer = layerList[i];
    }

    Json::Value networkConfList = rootValue["configs"];

    fb.close();
}
