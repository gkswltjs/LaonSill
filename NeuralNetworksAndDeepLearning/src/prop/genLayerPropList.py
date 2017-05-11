#!/usr/bin/env python

"""genLayerProp.py: """

import json;

####################################### Modify here ##########################################
# if you want to use specific custom type, you should insert header file that the custom type 
# is defined into headerFileList.
headerFileList = ["LayerConfig.h"]
##############################################################################################

def checkParamProperty(propDic, prop, propertyName):
    if not propertyName in propDic[prop]:
        print "ERROR: prop %s does not have %s property" % (prop, propertyName)
        exit(-1)

# (1) load propDef.json
try:
    jsonFile = open('layerPropDef.json', 'r')
    propDic = json.load(jsonFile)

except Exception as e:
    print str(e)
    exit(-1)

finally:
    jsonFile.close()

# (2) check propDef syntax
for prop in propDic:
    checkParamProperty(propDic, prop, "DESC")
    checkParamProperty(propDic, prop, "PARENT")
    checkParamProperty(propDic, prop, "LEVEL")
    checkParamProperty(propDic, prop, "VARS")

# (3) generate header file
headerTopSentences = [\
"/**",\
" * @file LayerPropList.h",\
" * @author moonhoen lee",\
" * @brief layer property structure list module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
"#ifndef LAYERPROPLIST_H_",\
"#define LAYERPROPLIST_H_",\
"",\
"#include <stdint.h>",\
"#include <string.h>",\
"",\
'#include "common.h"',\
'#include "SysLog.h"',\
"",\
]

headerClassDefSentences = [\
"class LayerPropList {",\
"public : ",\
"    LayerPropList() {}",\
"    virtual ~LayerPropList() {}",\
]

headerBottomSentences = [\
"};",\
"",\
"#endif /* LAYERPROPLIST_H_ */",\
]

# (4) generate source file
sourceTopSentences = [\
"/**",\
" * @file LayerPropList.cpp",\
" * @author moonhoen lee",\
" * @brief layer property structure list module",\
" * @warning",\
" *  The file is auto-generated.",\
" *  Do not modify the file!!!!",\
" */",\
"",\
'#include "LayerPropList.h"',\
'#include "SysLog.h"',\
"",\
]

varDic = dict()     # key : layer name(prop), value : vars
levelDic = dict()   # key : level, layer name(prop) list
maxLevel = -1

try:
    headerFile = open('LayerPropList.h', 'w+')
    sourceFile = open('LayerPropList.cpp', 'w+')

    # write top sentences
    for line in headerTopSentences:
        headerFile.write(line + "\n")

    for headerFileName in headerFileList:
        headerFile.write('#include "%s"\n' % headerFileName)
    headerFile.write('\n')

    for line in sourceTopSentences:
        sourceFile.write(line + "\n")

    # fill levelDic
    for prop in propDic:
        # (1) parse property structure def
        level = propDic[prop]["LEVEL"]
        if level not in levelDic:
            levelDic[level] = []

        levelDic[level].append(prop)

        if level > maxLevel:
            maxLevel = level
    
    # write structure
    for level in range(maxLevel + 1):
        propList = levelDic[level]

        for prop in propList:
            if prop in varDic:
                print "ERROR: duplicate prop(layer name). prop=%s" % prop
                exit(-1)

            varDic[prop] = []
           
            # fill parent var
            parent = propDic[prop]["PARENT"]
            if len(parent) > 1:
                if parent not in varDic:
                    print "ERROR: specified parent is not defined. prop=%s, parent=%s"\
                        % (prop, parent)
                    exit(-1)


                for var in varDic[parent]:
                    varDic[prop].append(var)

            for var in propDic[prop]["VARS"]:
                varDic[prop].append(var) 

        # (2) generate comment for property layer name
        headerFile.write('// property layer name : %s\n' % prop)

        headerFile.write('typedef struct %sPropLayer_s {\n' % prop)
        for var in varDic[prop]:
            if '[' in var[1]:
                splited = var[1].replace(']', '@').replace('[', '@').split('@')
                headerFile.write('    %s _%s_[%s];\n' % (splited[0], var[0], splited[1]))
            else:
                headerFile.write('    %s _%s_;\n' % (var[1], var[0]))

        headerFile.write('\n    %sPropLayer_s() {\n' % prop)

        for var in varDic[prop]:
            if '[' in var[1]:
                headerFile.write('        strcpy(_%s_, %s);\n' % (var[0], var[2]))
            else:
                headerFile.write('        _%s_ = (%s)%s;\n' % (var[0], var[1], var[2]))
        headerFile.write('\n    }\n')

        headerFile.write('} _%sPropLayer;\n\n' % prop)

  
    # write class
    for line in headerClassDefSentences:
        headerFile.write(line + "\n")

    headerFile.write("    static void setProp(void* target, const char* layer,")
    headerFile.write(" const char* property, void* value);\n\n")
    headerFile.write("private:\n")
    for level in range(maxLevel + 1):
        propList = levelDic[level]

        for prop in propList:
            headerFile.write(\
                "    static void set%s(void* target, const char* property, void* value);\n"\
                % prop)
            sourceFile.write(\
            "void LayerPropList::set%s(void* target, const char* property, void* value) {\n"\
                % prop)
            sourceFile.write('    _%sPropLayer* obj = (_%sPropLayer*)target;\n\n'\
                % (prop, prop))

            isFirstCond = True

            for var in varDic[prop]:
                if isFirstCond:
                    sourceFile.write('    if (strcmp(property, "%s") == 0) {\n' % var[0])
                    isFirstCond = False
                else:
                    sourceFile.write(' else if (strcmp(property, "%s") == 0) {\n' % var[0])

                if '[' in var[1]:
                    sourceFile.write('        strcpy(obj->_%s_, (const char*)value);\n'\
                        % var[0])
                else:
                    sourceFile.write('        memcpy((void*)&obj->_%s_, value, sizeof(%s));\n'\
                        % (var[0], var[1]))
                sourceFile.write('    }')
            sourceFile.write(' else {\n')
            sourceFile.write('        SASSERT(false, "invalid property.')
            sourceFile.write(' layer name=%s, property=%s"')
            sourceFile.write(', "%s", property);\n    }\n}\n\n' % prop) 

    sourceFile.write("void LayerPropList::setProp(void *target, const char* layer,")
    sourceFile.write(" const char* property, void* value) {\n")
    for level in range(maxLevel + 1):
        propList = levelDic[level]

        isFirstCond = True
        for prop in propList:
            if isFirstCond:
                sourceFile.write('    if (strcmp(layer, "%s") == 0) {\n' % prop)
                isFirstCond = False
            else:
                sourceFile.write(' else if (strcmp(layer, "%s") == 0) {\n' % prop)
            sourceFile.write('        set%s(target, property, value);\n' % prop) 
            sourceFile.write('    }')
    sourceFile.write(' else {\n')
    sourceFile.write('        SASSERT(false, "invalid layer. layer name=%s"')
    sourceFile.write(', layer);\n    }\n}\n\n')

    for line in headerBottomSentences:
        headerFile.write(line + "\n")

except Exception as e:
    print str(e)
    exit(-1)

finally:
    headerFile.close()
