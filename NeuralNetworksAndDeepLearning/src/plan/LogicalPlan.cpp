/**
 * @file LogicalPlan.cpp
 * @date 2017-05-10
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <algorithm>

#include "LogicalPlan.h"
#include "SysLog.h"
#include "StdOutLog.h"
#include "Layer.h"
#include "LayerPropList.h"
#include "PropMgmt.h"
#include "WorkContext.h"

using namespace std;

map<int, LogicalPlan*> LogicalPlan::lpMap;

void LogicalPlan::cleanup(int networkID) {
    SASSERT(LogicalPlan::lpMap.find(networkID) != LogicalPlan::lpMap.end(),
        "There is no network ID for the logical plan you are trying to delete."
        " networkID=%d", networkID);

    LogicalPlan* lp = LogicalPlan::lpMap[networkID];
    LogicalPlan::lpMap.erase(networkID);

    delete lp;
}

PlanDef* LogicalPlan::findPlanDef(LogicalPlan* lp, int planID) {
    for (int i = 0; i < lp->ppDefs.size(); i++) {
        if (lp->ppDefs[i].planID == planID)
            return &lp->ppDefs[i];
    }

    SASSERT(false, "cannot find plandef for requesting plan ID. planID=%d", planID);
}

// XXX: the number of codes for this function is too long!!!!!!! split it
//     build()함수는 아래와 같은 일들을 수행한다.
//  (1) 각 레이어의 정의(PlanDef)를 토대로 해야할 세부 plan들을 생성
//  (2) 각 세부 plan들간의 관계(ex 의존성)를 설정
//  (3) 특수 레이어 케이스(ex. split layer, inplace layer) 처리
//     - inplace layer : 자신의 인풋과 아웃풋이 동일한 경우
//     - split layer : A, B, C 3개의 레이어가 존재하는 경우에..
//                     A의 output이 B,C의 input이 되는 경우를 의미
void LogicalPlan::build(int networkID, map<int, PlanBuildDef> planDefMap) {
    // (1) fill input2ID & output2ID map
    map<string, vector<int>> input2IDMap;   // tensor name을 기준으로 input ID map
    map<string, vector<int>> output2IDMap;  // tensor name을 기준으로 output ID map

    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;
        PlanBuildDef value = it->second;

        for (int i = 0; i < value.inputs.size(); i++) {
            string inputKey = value.inputs[i];
            if (output2IDMap.find(inputKey) == output2IDMap.end()) {
                output2IDMap[inputKey] = {};
            }

            output2IDMap[inputKey].push_back(key);
        }

        for (int i = 0; i < value.outputs.size(); i++) {
            string outputKey = value.outputs[i];
            if (input2IDMap.find(outputKey) == input2IDMap.end()) {
                input2IDMap[outputKey] = {};
            }

            input2IDMap[outputKey].push_back(key);
        }
    }

    // (1-1) sort
    for (map<string, vector<int>>::iterator it = input2IDMap.begin();
        it != input2IDMap.end(); ++it) {
        string key = it->first;
        sort(input2IDMap[key].begin(), input2IDMap[key].end());
    }

    for (map<string, vector<int>>::iterator it=output2IDMap.begin(); it!=output2IDMap.end();
            ++it) {
        string key = it->first;
        sort(output2IDMap[key].begin(), output2IDMap[key].end());
    }

    // (2) fill propDowns
    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;
        PlanBuildDef value = it->second;
  
        int inputCount = value.inputs.size();
        int propDownCount = value.propDowns.size();
        for (int i = 0; i < inputCount - propDownCount; i++) {
            planDefMap[key].propDowns.push_back(true);
        }
    }

    // (3) generate plans
    LogicalPlan* lp = new LogicalPlan(networkID);

    // (3-1) generate forward plans
    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;

        PlanDef newPlanDef;
        PlanBuildDef planBuildDef = it->second;

        newPlanDef.planID = LP_FORWARD_PLANID(key);
        newPlanDef.planType = PLANTYPE_FORWARD;

        newPlanDef.layerID = key;
        newPlanDef.layerType = planBuildDef.layerType;

        newPlanDef.depCount = (int)planBuildDef.inputs.size();
        newPlanDef.notifyList = {};

        for (int i = 0; i < planBuildDef.outputs.size(); i++) {
            string outputName = planBuildDef.outputs[i];

            // 아웃풋이 다른 레이어의 인풋인지 확인한다. 
            // 만약 아웃풋이 다른 레이어의 인풋이 아니라면 output layer이다.
            // 아웃풋 레이어에서는 동일 레이어의 backward plan에게 notify 하면 된다.
            if (output2IDMap.find(outputName) == output2IDMap.end()) {
                newPlanDef.notifyList.push_back(LP_BACKWARD_PLANID(key));
                continue;
            }
            
            // inplace에 대해서 확인
            vector<int> IDList = output2IDMap[outputName];
            bool isInplace = false;
            for (int i = 0; i < IDList.size() - 1; i++) {
                if (key == IDList[i]) {
                    isInplace = true;
                    newPlanDef.notifyList.push_back(LP_FORWARD_PLANID(IDList[i+1]));
                    break;
                }
            }

            bool isSplit = ((output2IDMap.find(outputName) != output2IDMap.end()) &&
                    (output2IDMap[outputName].size() > 0) &&
                    (input2IDMap[outputName].size() > 0) &&
                    (input2IDMap[outputName].size() < output2IDMap[outputName].size()));

            if (isSplit && isInplace) {
                newPlanDef.notifyList.pop_back();
            } else if (!isSplit && !isInplace) {
                int nextPlanID = LP_FORWARD_PLANID(output2IDMap[outputName][0]);
                newPlanDef.notifyList.push_back(nextPlanID);
            }
        }

        lp->ppDefs.push_back(newPlanDef);
    }

    // (3-2) generate backward plans
    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;

        PlanDef newPlanDef;
        PlanBuildDef planBuildDef = it->second;

        newPlanDef.planID = LP_BACKWARD_PLANID(key);
        newPlanDef.planType = PLANTYPE_BACKWARD;

        newPlanDef.layerID = key;
        newPlanDef.layerType = planBuildDef.layerType;

        newPlanDef.depCount = (int)planBuildDef.outputs.size();
        newPlanDef.notifyList = {};

        int propDownCount = 0; 
        for (int i = 0; i < planBuildDef.inputs.size(); i++) {
            // learnable layer의 경우에 update plan에게 notify해줘야 한다.
            if (planBuildDef.learnable && planBuildDef.propDowns[i]) {
                newPlanDef.notifyList.push_back(LP_UPDATE_PLANID(key)); 
            }

            string inputName = planBuildDef.inputs[i];

            // 인풋이 다른 레이어의 아웃풋인지 확인한다. 
            // 만약 인풋이 다른 레이어의 아웃풋이 아니라면 input layer이다.
            // input layer는 딱히 알려줘야할 대상은 없다.
            if (input2IDMap.find(inputName) == input2IDMap.end()) {
                continue;
            }
                
            // input이 곧 자신인 경우(inplace)에 대해서 확인하고, 그에따른 처리를 한다.
            vector<int> IDList = output2IDMap[inputName];
            bool isInplace = false;
            for (int i = IDList.size() - 1; i > 0; i--) {
                if (key == IDList[i]) {
                    isInplace = true;
                    newPlanDef.notifyList.push_back(LP_BACKWARD_PLANID(IDList[i-1]));
                    break;
                }
            }

            bool isSplit = ((output2IDMap.find(inputName) != output2IDMap.end()) &&
                    (output2IDMap[inputName].size() > 0) &&
                    (input2IDMap[inputName].size() > 0) &&
                    (input2IDMap[inputName].size() < output2IDMap[inputName].size()));

            if (isSplit && isInplace) {
                newPlanDef.notifyList.pop_back();
            } else if (!isSplit && !isInplace) {
                int nextPlanID = LP_BACKWARD_PLANID(input2IDMap[inputName][0]);
                newPlanDef.notifyList.push_back(nextPlanID);
            }
        }

        lp->ppDefs.push_back(newPlanDef);
    }

    // (3-3) generate update plans
    for (map<int, PlanBuildDef>::iterator it = planDefMap.begin(); it != planDefMap.end();
        ++it) {
        int key = it->first;

        PlanDef newPlanDef;
        PlanBuildDef planBuildDef = it->second;

        newPlanDef.planID = LP_UPDATE_PLANID(key);
        newPlanDef.planType = PLANTYPE_UPDATE;

        newPlanDef.layerID = key;
        newPlanDef.layerType = planBuildDef.layerType;

        newPlanDef.notifyList = {};
        int depCount = 0;
        for (int i = 0; i < planBuildDef.inputs.size(); i++) {
            // learnable layer의 경우에 update plan에게 notify해줘야 한다.
            if (planBuildDef.learnable && planBuildDef.propDowns[i]) {
                depCount++;
            }
        }

        if (depCount > 0) {
            newPlanDef.depCount = depCount;
            lp->ppDefs.push_back(newPlanDef);
        }
    }

    // (3-4) generate split layer
    int curLayerID = LOGICAL_PLAN_MAX_USER_DEFINED_LAYERID;

    for (map<string, vector<int>>::iterator it = input2IDMap.begin();
        it != input2IDMap.end(); ++it) {
        string key = it->first;
        vector<int> inputIDs = it->second;

        if (output2IDMap.find(key) == output2IDMap.end())
            continue;
        vector<int> outputIDs = output2IDMap[key];

        if (inputIDs.size() == 0 || outputIDs.size() == 0)
            continue;

        if (inputIDs.size() == outputIDs.size())
            continue;

        SASSERT0(inputIDs.size() < outputIDs.size());

        // (3-4-0) prepare names for split layer
        int splitOutputCount = outputIDs.size() - inputIDs.size() + 1;
        vector<string> splitLayerOutputDataNames;
        vector<string> splitLayerInputDataNames;
        string splitLayerName = key + "_split";
        char splitLayerTempDataName[64];

        for (int i = 0; i < splitOutputCount; i++) {
            sprintf(splitLayerTempDataName, "%s_split_%d", key.c_str(), i);
            splitLayerOutputDataNames.push_back(string(splitLayerTempDataName));
        }
        splitLayerInputDataNames.push_back(key);

        // (3-4-1) generate split layer's forward plan
        PlanDef newPlanDefForward;
        newPlanDefForward.layerID = curLayerID;
        newPlanDefForward.planID = LP_FORWARD_PLANID(newPlanDefForward.layerID);
        newPlanDefForward.planType = PLANTYPE_FORWARD;
        newPlanDefForward.layerType = (int)Layer<float>::Split;
        newPlanDefForward.depCount = 1;

        newPlanDefForward.notifyList = {};
        for (int i = 0; i < splitOutputCount; i++) {
            int splitOutputID = LP_FORWARD_PLANID(outputIDs[outputIDs.size() - i - 1]);
            PlanDef* splitOutputPlanDef = LogicalPlan::findPlanDef(lp, splitOutputID);
            newPlanDefForward.notifyList.push_back(splitOutputID);
        }
        int splitInputID = LP_FORWARD_PLANID(inputIDs[inputIDs.size() - 1]);
        PlanDef* splitInputPlanDef = LogicalPlan::findPlanDef(lp, splitInputID);

        splitInputPlanDef->notifyList.push_back(newPlanDefForward.planID);

        // (3-4-2) generate split layer's backward plan
        PlanDef newPlanDefBackward;
        newPlanDefBackward.layerID = curLayerID;
        newPlanDefBackward.planID = LP_BACKWARD_PLANID(newPlanDefBackward.layerID);
        newPlanDefBackward.planType = PLANTYPE_BACKWARD;
        newPlanDefBackward.layerType = (int)Layer<float>::Split;

        splitOutputCount = outputIDs.size() - inputIDs.size() + 1;
        newPlanDefBackward.depCount = splitOutputCount;

        for (int i = 0; i < splitOutputCount; i++) {
            int splitOutputID = LP_BACKWARD_PLANID(outputIDs[outputIDs.size() - i - 1]);
            PlanDef* splitOutputPlanDef = LogicalPlan::findPlanDef(lp, splitOutputID);

            splitOutputPlanDef->notifyList.push_back(newPlanDefBackward.planID);
        }

        newPlanDefBackward.notifyList = {};
        splitInputID = LP_BACKWARD_PLANID(inputIDs[inputIDs.size() - 1]);
        splitInputPlanDef = LogicalPlan::findPlanDef(lp, splitInputID);
        //splitInputPlanDef->depCount += 1;
        newPlanDefBackward.notifyList.push_back(splitInputID);

        lp->ppDefs.push_back(newPlanDefForward);
        lp->ppDefs.push_back(newPlanDefBackward);

        // (3-4-3) create split layer's prop
        LayerProp* newProp = LayerPropList::createLayerProp(networkID, curLayerID,
            "Split");
        LayerPropList::setProp((void*)newProp->prop, "Split", "id", (void*)&curLayerID);
        LayerPropList::setProp((void*)newProp->prop, "Split", "name",
                (void*)&splitLayerName);

        LayerPropList::setProp((void*)newProp->prop, "Split", "input",
            (void*)&splitLayerInputDataNames);
        LayerPropList::setProp((void*)newProp->prop, "Split", "output",
            (void*)&splitLayerOutputDataNames);
        PropMgmt::insertLayerProp(newProp);

        // (3-4-4) change the data names of the layers associated with the split layer
        for (int i = 0; i < splitOutputCount; i++) {
            int splitOutputID = LP_BACKWARD_PLANID(outputIDs[outputIDs.size() - i - 1]);
            PlanDef* splitOutputPlanDef = LogicalPlan::findPlanDef(lp, splitOutputID);
            WorkContext::updateLayer(networkID, splitOutputPlanDef->layerID);

            SLPROP(Split, input).push_back(splitLayerOutputDataNames[i]);
        }

        curLayerID++;
    }

    SASSERT(LogicalPlan::lpMap.find(networkID) == LogicalPlan::lpMap.end(),
        "network ID has been declared redundant. network ID=%d", networkID);

    LogicalPlan::lpMap[networkID] = lp;
}

void LogicalPlan::printPlanDef(int networkID) {
    SASSERT(LogicalPlan::lpMap.find(networkID) != LogicalPlan::lpMap.end(),
        "There is no logical plan for the requested network ID. network ID=%d",
        networkID);

    LogicalPlan* lp = LogicalPlan::lpMap[networkID];

    for (int i = 0; i < lp->ppDefs.size(); i++) {
        char tempBuf[1024];
        int pos = 0;

        if (lp->ppDefs[i].notifyList.size() == 0) {
            strcpy(tempBuf, "None");
        } else {
            for (int j = 0; j < lp->ppDefs[i].notifyList.size(); j++) {
                pos += sprintf(tempBuf + pos, "%d ", lp->ppDefs[i].notifyList[j]);
            }
        }

        STDOUT_BLOCK(cout << "planID : " << lp->ppDefs[i].planID << 
            ", planType : " << lp->ppDefs[i].planType <<
            ", layer ID : " << lp->ppDefs[i].layerID <<
            ", layerType : " << lp->ppDefs[i].layerType <<
            ", depCount : " << lp->ppDefs[i].depCount << 
            " notify List : " << tempBuf << endl;);
    }
}

LogicalPlan* LogicalPlan::getLogicalPlan(int networkID) {
    SASSERT(LogicalPlan::lpMap.find(networkID) != LogicalPlan::lpMap.end(),
        "There is no logical plan for the requested network ID. network ID=%d",
        networkID);

    LogicalPlan* lp = LogicalPlan::lpMap[networkID];
    return lp;
}