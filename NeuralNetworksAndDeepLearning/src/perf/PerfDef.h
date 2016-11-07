/**
 * @file PerfDef.h
 * @date 2016-11-07
 * @author mhlee
 * @brief 
 * @details
 */

#ifndef PERFDEF_H
#define PERFDEF_H 

#include <string.h>

#include <vector>

#include "../common.h"
#include "PerfArgDef.h"

class PerfDef {
public: 
    PerfDef(bool jobScope, bool useTime, bool useAvgTime, bool useMaxTime) {
        this->jobScope = jobScope;
        this->useTime = useTime;
        this->useAvgTime = useAvgTime;
        this->useMaxTime = useMaxTime;
    }
    virtual ~PerfDef() {}

    bool jobScope;
    bool useTime;
    bool useAvgTime;
    bool useMaxTime;
    std::vector<PerfArgDef*> argArray;

    void addArgs(PerfArgDef* argDef) {
        this->argArray.push_back(argDef);
    }
};
#endif /* PERFDEF_H */
