#!/usr/bin/env python

"""genParam.py: """

import json;

def checkParamProperty(paramDic, param, propertyName):
    if not propertyName in paramDic[param]:
        print "ERROR: param %s does not have %s property" % (param, propertyName)
        exit(-1)

# (1) load paramDef.json
try:
    jsonFile = open('paramDef.json', 'r')
    paramDic = json.load(jsonFile)

except Exception as e:
    print str(e)
    exit(-1)

finally:
    jsonFile.close()

# (2) check paramDef syntax
for param in paramDic:
    checkParamProperty(paramDic, param, "DESC")
    checkParamProperty(paramDic, param, "MANDATORY")
    checkParamProperty(paramDic, param, "MUTABLE")
    checkParamProperty(paramDic, param, "SCOPE")
    checkParamProperty(paramDic, param, "TYPE")
    checkParamProperty(paramDic, param, "RANGE")
    checkParamProperty(paramDic, param, "DEFAULT")

if not "SESS_COUNT" in paramDic:
    print "ERROR: SESS_COUNT parameter does not exist"
    exit(-1)

# (3) generate header file
headerTopSentences = [\
"/**",\
" * @file Param.h",\
" * @author mhlee",\
" * @brief parameter mgmt module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
"#ifndef PARAM_H_",\
"#define PARAM_H_",\
"",\
"#include <stdint.h>",\
"",\
"#define SysParam(n)            Param::_##n",\
"#define SessParam(n)           Param::_sess_##n",\
"#define ParamDesc(n)           Param::_desc_##n",\
"",\
"class Param {",\
"public:",\
"    Param() {}",\
"    virtual ~Param() {}",\
"",\
]

headerBottomSentences = [\
"",\
"};",\
"",\
"#endif /* PARAM_H_ */"]

typeDic = {\
    "UINT8" : "uint8_t", "INT8" : "int8_t",\
    "UINT16" : "uint16_t", "INT16" : "int16_t",\
    "UINT32" : "uint32_t", "INT32" : "int32_t",\
    "UINT64" : "uint64_t", "INT64" : "int64_t",\
    "BOOL" : "bool", "FLOAT" : "float",\
    "DOUBLE" : "double", "LONGDOUBLE" : "long double",\
}

try:
    headerFile = open('Param.h', 'w+')

    for line in headerTopSentences:
        headerFile.write(line + "\n")

    for param in paramDic:
        sessScope = False

        if paramDic[param]["SCOPE"] == "SESSION":
            sessScope = True

        if paramDic[param]["MUTABLE"] == True:
            headerFile.write("    static volatile ")
        else:
            headerFile.write("    static ")

        if paramDic[param]["TYPE"] in typeDic:
            headerFile.write("%s " % typeDic[paramDic[param]["TYPE"]])
        elif "CHAR" in paramDic[param]["TYPE"]:
            # XXX: needs error-check
            arrayCount = (int)(paramDic[param]["TYPE"].replace(")", "$")\
                                .replace("(", "$").split("$")[1])
            headerFile.write("char[%d] " % arrayCount)
        else:
            print "ERROR: invalid param type(%s) for param(%s)" %\
                (paramDic[param]["TYPE"], param)
            exit(-1)

        headerFile.write("_%s;\n" % param)

        if sessScope == True:
            if paramDic[param]["MUTABLE"] == True:
                headerFile.write("    static volatile thread_local ")
            else:
                headerFile.write("    static thread_local ")

            if paramDic[param]["TYPE"] in typeDic:
                headerFile.write("%s " % typeDic[paramDic[param]["TYPE"]])
            elif "CHAR" in paramDic[param]["TYPE"]:
                # XXX: needs error-check
                arrayCount = (int)(paramDic[param]["TYPE"].replace(")", "$")\
                                    .replace("(", "$").split("$")[1])
                headerFile.write("char[%d] " % arrayCount)
            else:
                print "ERROR: invalid param type(%s) for param(%s)" %\
                    (paramDic[param]["TYPE"], param)
                exit(-1)

            headerFile.write("_sess_%s;\n" % param)

        headerFile.write('    static const char* _desc_%s;\n' % param)

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
" * @file Param.cpp",\
" * @author mhlee",\
" * @brief parameter mgmt module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
'#include "Param.h"',\
""]

try:
    sourceFile = open('Param.cpp', 'w+')

    for line in sourceTopSentences:
        sourceFile.write(line + "\n")

    for param in paramDic:
        sessScope = False

        if paramDic[param]["SCOPE"] == "SESSION":
            sessScope = True

        if paramDic[param]["MUTABLE"] == True:
            sourceFile.write("volatile ")

        if paramDic[param]["TYPE"] in typeDic:
            sourceFile.write("%s " % typeDic[paramDic[param]["TYPE"]])
        elif "CHAR" in paramDic[param]["TYPE"]:
            # XXX: needs error-check
            arrayCount = (int)(paramDic[param]["TYPE"].replace(")", "$")\
                                .replace("(", "$").split("$")[1])
            sourceFile.write("char[%d] " % arrayCount)
        else:
            print "ERROR: invalid param type(%s) for param(%s)" %\
                (paramDic[param]["TYPE"], param)
            exit(-1)

        sourceFile.write("Param::_%s = %s;\n" % (param, str(paramDic[param]["DEFAULT"])))

        if sessScope == True:
            if paramDic[param]["MUTABLE"] == True:
                sourceFile.write("volatile thread_local ")
            else:
                sourceFile.write("thread_local ")

            if paramDic[param]["TYPE"] in typeDic:
                sourceFile.write("%s " % typeDic[paramDic[param]["TYPE"]])
            elif "CHAR" in paramDic[param]["TYPE"]:
                # XXX: needs error-check
                arrayCount = (int)(paramDic[param]["TYPE"].replace(")", "$")\
                                    .replace("(", "$").split("$")[1])
                sourceFile.write("char[%d] " % arrayCount)
            else:
                print "ERROR: invalid param type(%s) for param(%s)" %\
                    (paramDic[param]["TYPE"], param)
                exit(-1)

            sourceFile.write("Param::_sess_%s = %s;\n" %\
                (param, str(paramDic[param]["DEFAULT"])))

        sourceFile.write('const char* Param::_desc_%s = {"%s"};\n' %\
            (param, paramDic[param]["DESC"]))
    
except Exception as e:
    print str(e)
    exit(-1)
finally:
    sourceFile.close()
