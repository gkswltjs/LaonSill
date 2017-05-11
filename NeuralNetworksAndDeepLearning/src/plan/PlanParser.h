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

#include "LogicalPlan.h"

class PlanParser {
public: 
    PlanParser() {}
    virtual ~PlanParser() {}

    static int loadNetwork(std::string filePath);
};
#endif /* PLANPARSER_H */
