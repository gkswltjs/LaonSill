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

#include <vector>
#include <string>

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

    // (1) get layer property
    Json::Value layerList = rootValue["layers"];
    for (int i = 0; i < layerList.size(); i++) {
        Json::Value layer = layerList[i];

        vector<string> keys = layer.getMemberNames();

        for (int j = 0; j < keys.size(); j++) {
            string key = keys[j];
            Json::Value val = layer[key.c_str()];
            
            switch(val.type()) {
                case Json::booleanValue:
                    cout << "key : " << key << ", value : " << val.asBool() << endl;
                    break;

                case Json::intValue:
                    cout << "key : " << key << ", value : " << val.asInt() << endl;
                    break;

                case Json::uintValue:
                    cout << "key : " << key << ", value : " << val.asUInt() << endl;
                    break;

                case Json::realValue:
                    cout << "key : " << key << ", value : " << val.asDouble() << endl;
                    break;

                case Json::stringValue:
                    cout << "key : " << key << ", value : " << val.asCString() << endl;
                    break;

                case Json::arrayValue:
                    cout << "key : " << key << ", array size : " << val.size() << endl;
                    break;

#if 0   // XXX: not now, but we will be support this type
                case Json::objectValue:
                    break;
#endif

                default:
                    SASSERT(false, "unsupported json-value. type=%d", val.type());
                    break;
            }
        }
    }

    // (2) get network property
    Json::Value networkConfList = rootValue["configs"];

    fb.close();
}
