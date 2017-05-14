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
#include "PropMgmt.h"
#include "SysLog.h"
#include "LayerPropList.h"

using namespace std;

int PlanParser::loadNetwork(string filePath) {
    // (1) 우선 network configuration file 파싱부터 진행
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
   
    // 파싱에 사용할 임시 변수들
    bool boolValue;
    int64_t int64Value;
    uint64_t uint64Value;
    double doubleValue;
    string stringValue;
    vector<bool> boolArrayValue;
    vector<int64_t> int64ArrayValue;
    vector<uint64_t> uint64ArrayValue;
    vector<double> doubleArrayValue;
    vector<string> stringArrayValue;
    Json::Value arrayValue;


    // (2) 파싱에 문제가 없어보이니.. 네트워크 ID 생성
    Network<float>* network = new Network<float>();
    int networkID = network->getNetworkID();

    // (3) fill layer property
    Json::Value layerList = rootValue["layers"];
    for (int i = 0; i < layerList.size(); i++) {
        Json::Value layer = layerList[i];
        vector<string> keys = layer.getMemberNames();

        // XXX: 예외처리 해야 한다!!!!
        int layerID = layer["id"].asInt();
        string layerType = layer["layer"].asCString();

        LayerProp* newProp = 
            LayerPropList::createLayerProp(networkID, layerID, layerType.c_str());

        // fill prop
        for (int j = 0; j < keys.size(); j++) {
            string key = keys[j];
            Json::Value val = layer[key.c_str()];

            if (strcmp(key.c_str(), "id") == 0)
                continue;
            if (strcmp(key.c_str(), "layer") == 0)
                continue;

            switch(val.type()) {
                case Json::booleanValue:
                    boolValue = val.asBool();
                    LayerPropList::setProp((void*)newProp->prop, layerType.c_str(),
                            key.c_str(), (void*)&boolValue);
                    break;

                case Json::intValue:
                    int64Value = val.asInt64();
                    LayerPropList::setProp((void*)newProp->prop, layerType.c_str(),
                            key.c_str(), (void*)&int64Value);
                    break;

                case Json::realValue:
                    doubleValue = val.asDouble();
                    LayerPropList::setProp((void*)newProp->prop, layerType.c_str(),
                            key.c_str(), (void*)&doubleValue);
                    break;

                case Json::stringValue:
                    stringValue = val.asCString();
                    LayerPropList::setProp((void*)newProp->prop, layerType.c_str(),
                            key.c_str(), (void*)&stringValue);
                    break;

                case Json::arrayValue:
                    // peek 1st value's type
                    SASSERT0(val.size() > 0);
                    arrayValue = val[0];
                    if (arrayValue.type() == Json::booleanValue) {
                        for (int k = 0; k < val.size(); k++) {
                            arrayValue = val[k];
                            boolArrayValue.push_back(arrayValue.asBool());
                        }
                        LayerPropList::setProp((void*)newProp->prop, layerType.c_str(),
                            key.c_str(), (void*)&boolArrayValue);
                    } else if (arrayValue.type() == Json::intValue) {
                        for (int k = 0; k < val.size(); k++) {
                            arrayValue = val[k];
                            int64ArrayValue.push_back(arrayValue.asInt64());
                        }
                        LayerPropList::setProp((void*)newProp->prop, layerType.c_str(),
                            key.c_str(), (void*)&int64ArrayValue);
                    } else if (arrayValue.type() == Json::realValue) {
                        for (int k = 0; k < val.size(); k++) {
                            arrayValue = val[k];
                            doubleArrayValue.push_back(arrayValue.asDouble());
                        }
                        LayerPropList::setProp((void*)newProp->prop, layerType.c_str(),
                            key.c_str(), (void*)&doubleArrayValue);
                    } else if (arrayValue.type() == Json::stringValue) {
                        for (int k = 0; k < val.size(); k++) {
                            arrayValue = val[k];
                            stringArrayValue.push_back(arrayValue.asString());
                        }
                        LayerPropList::setProp((void*)newProp->prop, layerType.c_str(),
                            key.c_str(), (void*)&stringArrayValue);
                    } else {
                        SASSERT(false, "Unsupported sub-type for array type. sub_type=%d",
                            (int)arrayValue.type());
                    }
                    break;

                default:
                    SASSERT(false, "unsupported json-value. type=%d", val.type());
                    break;
            }
        }
        PropMgmt::insertLayerProp(newProp);
    }

    // (2) get network property
    _NetworkProp *networkProp = new _NetworkProp();
    Json::Value networkConfDic = rootValue["configs"];

    vector<string> keys = networkConfDic.getMemberNames();
    for (int i = 0; i < keys.size(); i++) {
        string key = keys[i];
        Json::Value val = networkConfDic[key.c_str()];

        switch(val.type()) {
            case Json::booleanValue:
                boolValue = val.asBool();
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&boolValue);
                break;

            case Json::intValue:
                int64Value = val.asInt64();
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&int64Value);
                break;

            case Json::realValue:
                doubleValue = val.asDouble();
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&doubleValue);
                break;

            case Json::stringValue:
                stringValue = val.asCString();
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&stringValue);
                break;

            case Json::arrayValue:
                // peek 1st value's type
                SASSERT0(val.size() > 0);
                arrayValue = val[0];
                if (arrayValue.type() == Json::booleanValue) {
                    for (int k = 0; k < val.size(); k++) {
                        arrayValue = val[k];
                        boolArrayValue.push_back(arrayValue.asBool());
                    }
                    NetworkProp::setProp(networkProp, key.c_str(), (void*)&boolArrayValue);
                } else if (arrayValue.type() == Json::intValue) {
                    for (int k = 0; k < val.size(); k++) {
                        arrayValue = val[k];
                        int64ArrayValue.push_back(arrayValue.asInt64());
                    }
                    NetworkProp::setProp(networkProp, key.c_str(), (void*)&int64ArrayValue);
                } else if (arrayValue.type() == Json::realValue) {
                    for (int k = 0; k < val.size(); k++) {
                        arrayValue = val[k];
                        doubleArrayValue.push_back(arrayValue.asDouble());
                    }
                    NetworkProp::setProp(networkProp, key.c_str(), (void*)&doubleArrayValue);
                } else if (arrayValue.type() == Json::stringValue) {
                    for (int k = 0; k < val.size(); k++) {
                        arrayValue = val[k];
                        stringArrayValue.push_back(arrayValue.asString());
                    }
                    NetworkProp::setProp(networkProp, key.c_str(), (void*)&stringArrayValue);
                } else {
                    SASSERT(false, "Unsupported sub-type for array type. sub_type=%d",
                        (int)arrayValue.type());
                }
                break;

            default:
                SASSERT(false, "unsupported json-value. type=%d", val.type());
                break;
        }
    }
    PropMgmt::insertNetworkProp(networkID, networkProp);

    fb.close();
}
