#!/usr/bin/env python

"""genPerf.py: """

import json;

def checkParamProperty(perfDic, perf, propertyName):
    if not propertyName in perfDic[perf]:
        print "ERROR: perf %s does not have %s property" % (perf, propertyName)
        exit(-1)

# (1) load perfDef.json
try:
    jsonFile = open('perfDef.json', 'r')
    perfDic = json.load(jsonFile)

except Exception as e:
    print str(e)
    exit(-1)

finally:
    jsonFile.close()

# (2) check perfDef syntax
for perf in perfDic:
    checkParamProperty(perfDic, perf, "DESC")
    checkParamProperty(perfDic, perf, "SCOPE")
    checkParamProperty(perfDic, perf, "USETIME")
    checkParamProperty(perfDic, perf, "USEAVGTIME")
    checkParamProperty(perfDic, perf, "USEMAXTIME")
    checkParamProperty(perfDic, perf, "ARGS")

# (3) generate header file
headerTopSentences = [\
"/**",\
" * @file PerfList.h",\
" * @author mhlee",\
" * @brief performance list module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
"#ifndef PERFLIST_H_",\
"#define PERFLIST_H_",\
"",\
"#include <stdint.h>",\
"#include <vector>",\
"#include <map>",\
"#include <string>",\
"",\
'#include "../common.h"',\
'#include "PerfArgDef.h"',\
'#include "PerfDef.h"',\
"",\
"class PerfList {",\
"public:",\
"    PerfList() {}",\
"    virtual ~PerfList() {}",\
"",\
]

headerBottomSentences = [\
"    static void    fillPerfDefMap(std::map<std::string, PerfDef*>& perfDefMap);\n",\
"};",\
"",\
"#endif /* PERFLIST_H_ */"]

typeDic = {\
    "UINT8" : "uint8_t", "INT8" : "int8_t","UINT16" : "uint16_t", "INT16" : "int16_t",\
    "UINT32" : "uint32_t", "INT32" : "int32_t","UINT64" : "uint64_t", "INT64" : "int64_t",\
    "FLOAT" : "float", "DOUBLE" : "double", "LONGDOUBLE" : "long double"\
}

try:
    headerFile = open('PerfList.h', 'w+')

    for line in headerTopSentences:
        headerFile.write(line + "\n")

    for perf in perfDic:
        # (1) parse performance def
        desc = perfDic[perf]["DESC"]

        jobScope = False
        if perfDic[perf]["SCOPE"] == "JOB":
            jobScope = True

        useTime = perfDic[perf]["USETIME"]

        if useTime == False:
            useAvgTime = False
            useMaxTime = False
        else:
            useAvgTime = perfDic[perf]["USEAVGTIME"]
            useMaxTime = perfDic[perf]["USEMAXTIME"]

        perfArgs = perfDic[perf]["ARGS"]
        newArgList = []
        for perfArg in perfArgs:
            if perfArg[1] in typeDic:
                typeString = typeDic[perfArg[1]]
            else:
                print "ERROR: invalid args type(%s) for perf(%s)" %\
                    (perfDic[perf]["TYPE"], perf)
                exit(-1)
            newArgList.append((perfArg[0], typeString, perfArg[2])) 

        # (2) generate performance comment
        headerFile.write('    // PERF NAME : %s\n' % perf)

        # (3) generate variables
        if jobScope == True:
            volStr = "thread_local"
        else:
            volStr = "volatile"

        headerFile.write("    static %s long _%sCount;\n" % (volStr, perf))
        if useTime == True:
            headerFile.write("    static %s double _%sTime;\n" % (volStr, perf))
        if useAvgTime == True:
            headerFile.write("    static %s double _%sAvgTime;\n" % (volStr, perf))
        if useMaxTime == True:
            headerFile.write("    static %s double _%sMaxTime;\n" % (volStr, perf))

        for newArg in newArgList:
            headerFile.write("    static %s %s _%s_%s;\n" %\
                (volStr, newArg[1], perf, newArg[0]))

        headerFile.write('\n')

    for line in headerBottomSentences:
        headerFile.write(line + "\n")

except Exception as e:
    print str(e)
    exit(-1)

finally:
    headerFile.close()

# (4) generate source file
sourceTopSentences = [\
"/**",\
" * @file PerfList.cpp",\
" * @author mhlee",\
" * @brief performance list module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
'#include "PerfList.h"',\
"",\
"using namespace std;",\
"",\
""]

perfDefList = []

try:
    sourceFile = open('PerfList.cpp', 'w+')

    for line in sourceTopSentences:
        sourceFile.write(line + "\n")

    for perf in perfDic:
        # (1) parse performance def
        desc = perfDic[perf]["DESC"]

        jobScope = False
        if perfDic[perf]["SCOPE"] == "JOB":
            jobScope = True

        useTime = perfDic[perf]["USETIME"]

        if useTime == False:
            useAvgTime = False
            useMaxTime = False
        else:
            useAvgTime = perfDic[perf]["USEAVGTIME"]
            useMaxTime = perfDic[perf]["USEMAXTIME"]

        perfArgs = perfDic[perf]["ARGS"]
        newArgList = []
        for perfArg in perfArgs:
            if perfArg[1] in typeDic:
                typeString = typeDic[perfArg[1]]
            else:
                print "ERROR: invalid args type(%s) for perf(%s)" %\
                    (perfDic[perf]["TYPE"], perf)
                exit(-1)
            newArgList.append((perfArg[0], typeString, perfArg[2])) 

        # (2) generate performance comment
        sourceFile.write('// PERF NAME : %s\n' % perf)

        # (3) generate variables
        if jobScope == True:
            volStr = "thread_local"
        else:
            volStr = "volatile"

        sourceFile.write("%s long PerfList::_%sCount = 0L;\n" % (volStr, perf))
        if useTime == True:
            sourceFile.write("%s double PerfList::_%sTime = 0;\n" % (volStr, perf))
        if useAvgTime == True:
            sourceFile.write("%s double PerfList::_%sAvgTime = 0;\n" % (volStr, perf))
        if useMaxTime == True:
            sourceFile.write("%s double PerfList::_%sMaxTime = 0;\n" % (volStr, perf))

        for newArg in newArgList:
            sourceFile.write("%s %s PerfList::_%s_%s = 0;\n" %\
                (volStr, newArg[1], perf, newArg[0]))

        sourceFile.write('\n')
        perfDefList.append((perf, jobScope, useTime, useAvgTime, useMaxTime, newArgList))

    # (12) prepare fillPerfDefMap func() 
    sourceFile.write("void PerfList::fillPerfDefMap(map<string, PerfDef*>& perfDefMap) {\n")

    isFirst = True
    for perfDef in perfDefList:
        if isFirst == True:
            isFirst = False
        else:
            sourceFile.write('\n')
        sourceFile.write('    PerfDef* perfDef%s = new PerfDef(%s, %s, %s, %s);\n'\
            % (str(perfDef[0]), str(perfDef[1]).lower(), str(perfDef[2]).lower(),\
                str(perfDef[3]).lower(), str(perfDef[4]).lower()))
        
        for newArg in perfDef[5]:
            sourceFile.write('    perfDef%s->addArgs(new PerfArgDef("%s", "%s", "%s"));\n'\
                % (str(perfDef[0]), str(newArg[0]), str(newArg[1]), str(newArg[2])))
        sourceFile.write('    perfDefMap["%s"] = perfDef%s;\n'\
            % (str(perfDef[0]), str(perfDef[0])))
            
    sourceFile.write("}\n\n")
    
except Exception as e:
    print str(e)
    exit(-1)
finally:
    sourceFile.close()
