#!/usr/bin/env python

"""genNProp.py: """

import json;

####################################### Modify here ##########################################
# if you want to use specific custom type, you should insert header file that the custom type 
# is defined into headerFileList.
headerFileList = ["NetworkConfig.h"]
##############################################################################################

def checkParamProperty(propDic, prop, propertyName):
    if not propertyName in propDic[prop]:
        print "ERROR: prop %s does not have %s property" % (prop, propertyName)
        exit(-1)

# (1) load propDef.json
try:
    jsonFile = open('networkPropDef.json', 'r')
    propDic = json.load(jsonFile)

except Exception as e:
    print str(e)
    exit(-1)

finally:
    jsonFile.close()

# (3) generate header file
headerTopSentences = [\
"/**",\
" * @file NetworkProp.h",\
" * @author moonhoen lee",\
" * @brief network property module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
"#ifndef NETWORKPROP_H_",\
"#define NETWORKPROP_H_",\
"",\
"#include <stdint.h>",\
"#include <string.h>",\
"",\
'#include "common.h"',\
'#include "SysLog.h"',\
"",\
]

headerClassDefSentences = [\
"class NetworkProp {",\
"public : ",\
"    NetworkProp() {}",\
"    virtual ~NetworkProp() {}",\
]

headerBottomSentences = [\
"};",\
"",\
"#endif /* NETWORKPROP_H_ */",\
]

# (4) generate source file
sourceTopSentences = [\
"/**",\
" * @file NetworkProp.cpp",\
" * @author moonhoen lee",\
" * @brief network property module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
'#include "NetworkProp.h"',\
'#include "SysLog.h"',\
"",\
]

try:
    headerFile = open('NetworkProp.h', 'w+')
    sourceFile = open('NetworkProp.cpp', 'w+')

    # write top sentences
    for line in headerTopSentences:
        headerFile.write(line + "\n")

    for headerFileName in headerFileList:
        headerFile.write('#include "%s"\n' % headerFileName)
    headerFile.write('\n')

    for line in sourceTopSentences:
        sourceFile.write(line + "\n")
    
    # write structure
    headerFile.write('typedef struct NetworkProp_s {\n')

    varList = propDic["VARS"]
    for var in varList:
        if '[' in var[1]:
            splited = var[1].replace(']', '@').replace('[', '@').split('@')
            headerFile.write('    %s _%s_[%s];\n' % (splited[0], var[0], splited[1]))
        else:
            headerFile.write('    %s _%s_;\n' % (var[1], var[0]))

    headerFile.write('\n    NetworkProp_s() {\n')

    for var in varList:
        if '[' in var[1]:
            headerFile.write('        strcpy(_%s_, %s);\n' % (var[0], var[2]))
        else:
            headerFile.write('        _%s_ = (%s)%s;\n' % (var[0], var[1], var[2]))
    headerFile.write('\n    }\n')
    headerFile.write('} _NetworkProp;\n\n')
  
    # write class
    for line in headerClassDefSentences:
        headerFile.write(line + "\n")

    headerFile.write("    static void setProp(_NetworkProp* target, ")
    headerFile.write(" const char* property, void* value);\n\n")
    sourceFile.write("void NetworkProp::setProp(_NetworkProp* target, ")
    sourceFile.write(" const char* property, void* value) {\n")

    isFirstCond = True

    for var in varList:
        if isFirstCond:
            sourceFile.write('    if (strcmp(property, "%s") == 0) {\n' % var[0])
            isFirstCond = False
        else:
            sourceFile.write(' else if (strcmp(property, "%s") == 0) {\n' % var[0])

        if '[' in var[1]:
            sourceFile.write('        strcpy(target->_%s_, (const char*)value);\n'\
                % var[0])
        else:
            sourceFile.write('        memcpy((void*)&target->_%s_, value, sizeof(%s));\n'\
                % (var[0], var[1]))
        sourceFile.write('    }')
                
    sourceFile.write(' else {\n')
    sourceFile.write('        SASSERT(false, "invalid network property.')
    sourceFile.write(' property=%s"')
    sourceFile.write(', property);\n    }\n}\n\n')

    for line in headerBottomSentences:
        headerFile.write(line + "\n")

except Exception as e:
    print str(e)
    exit(-1)

finally:
    headerFile.close()
