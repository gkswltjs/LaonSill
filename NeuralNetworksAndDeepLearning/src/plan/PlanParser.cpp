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

#include "PlanParser.h"
#include "PropMgmt.h"
#include "SysLog.h"
#include "LayerPropList.h"

using namespace std;

void PlanParser::setPropValue(Json::Value val, bool isLayer, string layerType, string key,
    void* prop) {
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

    bool isStructType = false;
    string property;
    string subProperty;
    size_t pos = key.find('.');
    if (pos != string::npos) { 
        property = key.substr(0, pos);
        subProperty = key.substr(pos+1);
        isStructType = true;
        SASSERT0(isLayer);  // layer property만 현재 struct type을 지원한다.
    }

    _NetworkProp* networkProp;
    if (!isLayer) {
        networkProp = (_NetworkProp*)prop;
    }

    switch(val.type()) {
    case Json::booleanValue:
        boolValue = val.asBool();
        if (isLayer && !isStructType) {
            LayerPropList::setProp(prop, layerType.c_str(), key.c_str(), (void*)&boolValue);
        } else if (isLayer && isStructType) {
            LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                subProperty.c_str(), (void*)&boolValue);
        } else {
            NetworkProp::setProp(networkProp, key.c_str(), (void*)&boolValue);
        }
        break;

    case Json::intValue:
        int64Value = val.asInt64();
        if (isLayer && !isStructType) {
            LayerPropList::setProp(prop, layerType.c_str(), key.c_str(), (void*)&int64Value);
        } else if (isLayer && isStructType) {
            LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                subProperty.c_str(), (void*)&int64Value);
        } else {
            NetworkProp::setProp(networkProp, key.c_str(), (void*)&int64Value);
        }
        break;

    case Json::realValue:
        doubleValue = val.asDouble();
        if (isLayer && !isStructType) {
            LayerPropList::setProp(prop, layerType.c_str(), key.c_str(), (void*)&doubleValue);
        } else if (isLayer && isStructType) {
            LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                subProperty.c_str(), (void*)&doubleValue);
        } else {
            NetworkProp::setProp(networkProp, key.c_str(), (void*)&doubleValue);
        }
        break;

    case Json::stringValue:
        stringValue = val.asCString();
        if (isLayer && !isStructType) {
            LayerPropList::setProp(prop, layerType.c_str(), key.c_str(), (void*)&stringValue);
        } else if (isLayer && isStructType) {
            LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                subProperty.c_str(), (void*)&stringValue);
        } else {
            NetworkProp::setProp(networkProp, key.c_str(), (void*)&stringValue);
        }
        break;

    case Json::arrayValue:
        // peek 1st value's type
        SASSERT0(val.size() > 0);
        arrayValue = val[0];
        if (arrayValue.type() == Json::booleanValue) {
            boolArrayValue = {};
            for (int k = 0; k < val.size(); k++) {
                arrayValue = val[k];
                boolArrayValue.push_back(arrayValue.asBool());
            }
            if (isLayer && !isStructType) {
                LayerPropList::setProp(prop, layerType.c_str(), key.c_str(),
                    (void*)&boolArrayValue);
            } else if (isLayer && isStructType) {
                LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                    subProperty.c_str(), (void*)&boolArrayValue);
            } else {
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&boolArrayValue);
            }
        } else if (arrayValue.type() == Json::intValue) {
            int64ArrayValue = {};
            for (int k = 0; k < val.size(); k++) {
                arrayValue = val[k];
                int64ArrayValue.push_back(arrayValue.asInt64());
            }
            if (isLayer && !isStructType) {
                LayerPropList::setProp(prop, layerType.c_str(), key.c_str(),
                    (void*)&int64ArrayValue);
            } else if (isLayer && isStructType) {
                LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                    subProperty.c_str(), (void*)&int64ArrayValue);
            } else {
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&int64ArrayValue);
            }
        } else if (arrayValue.type() == Json::realValue) {
            doubleArrayValue = {};
            for (int k = 0; k < val.size(); k++) {
                arrayValue = val[k];
                doubleArrayValue.push_back(arrayValue.asDouble());
            }
            if (isLayer && !isStructType) {
                LayerPropList::setProp(prop, layerType.c_str(), key.c_str(),
                    (void*)&doubleArrayValue);
            } else if (isLayer && isStructType) {
                LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                    subProperty.c_str(), (void*)&doubleArrayValue);
            } else {
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&doubleArrayValue);
            }
        } else if (arrayValue.type() == Json::stringValue) {
            stringArrayValue = {};
            for (int k = 0; k < val.size(); k++) {
                arrayValue = val[k];
                stringArrayValue.push_back(arrayValue.asString());
            }
            
            if (isLayer && !isStructType) {
                LayerPropList::setProp(prop, layerType.c_str(), key.c_str(),
                    (void*)&stringArrayValue);
            } else if (isLayer && isStructType) {
                LayerPropList::setPropStruct(prop, layerType.c_str(), property.c_str(),
                    subProperty.c_str(), (void*)&stringArrayValue);
            } else {
                NetworkProp::setProp(networkProp, key.c_str(), (void*)&stringArrayValue);
            }
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

void PlanParser::handleInnerLayer(int networkID, Json::Value vals, string parentLayerType,
    void* parentProp) {
    vector<int64_t> innerIDList;

    for (int i = 0; i < vals.size(); i++) {
        Json::Value innerLayer = vals[i];

        int layerID = innerLayer["id"].asInt();
        SASSERT(layerID < LOGICAL_PLAN_MAX_USER_DEFINED_LAYERID,
            "layer ID should less than %d. layer ID=%d",
            LOGICAL_PLAN_MAX_USER_DEFINED_LAYERID, layerID);
        string layerType = innerLayer["layer"].asCString();

        LayerProp* innerProp = 
            LayerPropList::createLayerProp(networkID, layerID, layerType.c_str());

        vector<string> keys = innerLayer.getMemberNames();

        for (int j = 0; j < keys.size(); j++) {
            string key = keys[j];
            Json::Value val = innerLayer[key.c_str()];

            if (strcmp(key.c_str(), "id") == 0)
                continue;
            if (strcmp(key.c_str(), "layer") == 0)
                continue;

            setPropValue(val, true, layerType, key,  (void*)innerProp->prop);
        }

        // new prop를 설정.
        PropMgmt::insertLayerProp(innerProp);

        innerIDList.push_back((int64_t)layerID);
    }

    LayerPropList::setProp(parentProp, parentLayerType.c_str(), "innerLayerIDs",
        (void*)&innerIDList);
}

// XXX: 함수 하나가 엄청 길다... 흠.. 나중에 소스 좀 정리하자..
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
   
    // logical plan을 만들기 위한 변수들
    map<int, PlanBuildDef> planDefMap;

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
        SASSERT(layerID < LOGICAL_PLAN_MAX_USER_DEFINED_LAYERID,
            "layer ID should less than %d. layer ID=%d",
            LOGICAL_PLAN_MAX_USER_DEFINED_LAYERID, layerID);

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

            if (strcmp(key.c_str(), "innerLayer") == 0) {
                handleInnerLayer(networkID, val, layerType, newProp->prop);
                continue;
            }

            setPropValue(val, true, layerType, key,  (void*)newProp->prop);
        }

        // new prop를 설정.
        PropMgmt::insertLayerProp(newProp);
       
        // plandef 맵에 추가
        SASSERT(planDefMap.find(layerID) == planDefMap.end(),
            "layer ID has been declared redundant. layer ID=%d", layerID);
        PlanBuildDef newPlanDef;
        newPlanDef.layerID = layerID;
        newPlanDef.layerType = LayerPropList::getLayerType(layerType.c_str());   // TODO:

        vector<string> inputs = LayerPropList::getInputs(layerType.c_str(), newProp->prop);
        vector<string> outputs = LayerPropList::getOutputs(layerType.c_str(), newProp->prop);
        vector<bool> propDowns =
            LayerPropList::getPropDowns(layerType.c_str(), newProp->prop); 

        for (int j = 0; j < inputs.size(); j++) {
            newPlanDef.inputs.push_back(inputs[j]);
        }

        for (int j = 0; j < outputs.size(); j++) {
            newPlanDef.outputs.push_back(outputs[j]);
        }

        for (int j = 0; j < propDowns.size(); j++) {
            newPlanDef.propDowns.push_back(propDowns[j]);
        }

        newPlanDef.isDonator = LayerPropList::isDonator(layerType.c_str(), newProp->prop);
        newPlanDef.isReceiver = LayerPropList::isReceiver(layerType.c_str(), newProp->prop);
        newPlanDef.donatorID = LayerPropList::getDonatorID(layerType.c_str(), newProp->prop);
        newPlanDef.learnable = LayerPropList::isLearnable(layerType.c_str(), newProp->prop);

        planDefMap[layerID] = newPlanDef;
    }


    // (2) get network property
    _NetworkProp *networkProp = new _NetworkProp();
    Json::Value networkConfDic = rootValue["configs"];

    vector<string> keys = networkConfDic.getMemberNames();
    for (int i = 0; i < keys.size(); i++) {
        string key = keys[i];
        Json::Value val = networkConfDic[key.c_str()];

        setPropValue(val, false, "", key,  (void*)networkProp);
    }
    PropMgmt::insertNetworkProp(networkID, networkProp);

    LogicalPlan::build(networkID, planDefMap);

    fb.close();

    return networkID;
}
