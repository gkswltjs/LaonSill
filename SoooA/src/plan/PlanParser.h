/**
 * @file PlanParser.h
 * @date 2017-05-04
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef PLANPARSER_H
#define PLANPARSER_H 

#include <string>

#include "jsoncpp/json/json.h"

#include "LogicalPlan.h"

class PlanParser {
public: 
    PlanParser() {}
    virtual ~PlanParser() {}

    static int loadNetwork(std::string filePath);
    static int loadNetworkByJSONString(std::string jsonString);
    static void buildNetwork(int networkID, Json::Value val);
private:
    static void setPropValue(Json::Value val, bool isLayer, std::string layerType,
        std::string key, void* prop);
    static std::vector<int64_t> handleInnerLayer(int networkID, Json::Value vals,
        std::string parentLayerType, void* parentProp);
};
#endif /* PLANPARSER_H */
