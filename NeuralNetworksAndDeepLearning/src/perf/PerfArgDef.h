/**
 * @file PerfArgDef.h
 * @date 2016-11-07
 * @author mhlee
 * @brief 
 * @details
 */

#ifndef PERFARGDEF_H
#define PERFARGDEF_H 

#include <string.h>

#include "../common.h"

#define PERFARGDEF_ARGNAME_MAXSIZE          (64)
#define PERFARGDEF_DESC_MAXSIZE             (256)
#define PERFARGDEF_TYPENAME_MAXSIZE         (32)

class PerfArgDef {
public: 
    PerfArgDef(const char *argName, const char* typeName, const char* desc) {
        strcpy(this->argName, argName);
        strcpy(this->typeName, typeName);
        strcpy(this->desc, desc);
    }
    virtual ~PerfArgDef() {}

    // use "string" instead of "fixed char array"?
    char    argName[PERFARGDEF_ARGNAME_MAXSIZE];
    char    typeName[PERFARGDEF_TYPENAME_MAXSIZE];
    char    desc[PERFARGDEF_DESC_MAXSIZE];
};
#endif /* PERFARGDEF_H */
