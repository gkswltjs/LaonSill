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

using namespace std;

#define LP_FORWARD_PLANID(id)       (id * 3 + 0)
#define LP_BACKWARD_PLANID(id)      (id * 3 + 1)
#define LP_UPDATE_PLANID(id)        (id * 3 + 2)

map<int, LogicalPlan*> LogicalPlan::lpMap;

void LogicalPlan::cleanup(int networkID) {
    SASSERT(LogicalPlan::lpMap.find(networkID) != LogicalPlan::lpMap.end(),
        "There is no network ID for the logical plan you are trying to delete."
        " networkID=%d", networkID);

    LogicalPlan* lp = LogicalPlan::lpMap[networkID];
    LogicalPlan::lpMap.erase(networkID);

    delete lp;
}

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
            
            // output이 곧 자신인 경우(inplace)에 대해서 확인하고, 그에따른 처리를 한다.
            vector<int> IDList = input2IDMap[outputName];
            bool isInplace = false;
            for (int i = 0; i < IDList.size() - 1; i++) {
                if (key == IDList[i]) {
                    isInplace = true;
                    newPlanDef.notifyList.push_back(LP_FORWARD_PLANID(IDList[i+1]));
                    break;
                }
            }

            if (!isInplace) {
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

            if (!isInplace) {
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
