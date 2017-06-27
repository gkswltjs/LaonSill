#!/usr/bin/env python
import os
import shutil

###########################################################################################
# Settings
#  - you can modify below things
###########################################################################################

configNameList = ["Debug", "DebugClient", "Release", "ReleaseClient"]

symbolDic = dict()  # key : configure name, value : symbol list
symbolDic["Debug"]              = ["GPU_MODE", "DEBUG_MODE"]
symbolDic["DebugClient"]        = ["GPU_MODE", "DEBUG_MODE", "CLIENT_MODE"]
symbolDic["Release"]            = ["GPU_MODE"]
symbolDic["ReleaseClient"]      = ["GPU_MODE", "CLIENT_MODE"]

libList = ["cudart", "opencv_core", "cublas", "cudnn", "boost_system", "boost_filesystem",\
         "opencv_highgui", "opencv_imgproc", "opencv_features2d", "z", \
 	 "boost_iostreams", "X11", "openblas", "blas", "gpu_nms", "lmdb"]

targetNameDic = dict()  # key : configure name, value : target name
targetNameDic["Debug"]          = "SoooaServerDebug"
targetNameDic["DebugClient"]    = "SoooaClientDebug"
targetNameDic["Release"]        = "SoooaServer"
targetNameDic["ReleaseClient"]  = "SoooaClient"

dirNameDic = dict() # key : configure name, value : directory name
dirNameDic["Debug"]             = "DebugGen"
dirNameDic["DebugClient"]       = "DebugClientGen"
dirNameDic["Release"]           = "ReleaseGen"
dirNameDic["ReleaseClient"]     = "ReleaseClientGen"

subDirList = []     # directories under src directory

supportArch             = "compute_60"
supportCode             = "sm_60"

sourceHomeDirEnvName    = 'SOOOA_BUILD_PATH'

incEnvVarList = ["INC_PATH_GNUPLOT", "INC_PATH_CIMG", "INC_PATH_OPENCV",\
                "INC_PATH_BLAS"]

###########################################################################################
# Codes
#  - do not modify below things
###########################################################################################

def createDirectory(configName):
    try:
        srcHomeDir = os.environ[sourceHomeDirEnvName]
      
        buildDirPath = srcHomeDir + '/%s' % dirNameDic[configName]
        if os.path.exists(buildDirPath):
            print 'ERROR: %s is already exists. remove it first and then build'\
                % buildDirPath
            exit(-1)

        os.mkdir(buildDirPath)

        return buildDirPath
    except Exception as e:
        print 'ERROR: createDirectory()'
        print str(e)
        exit(-1)

def generateMakefile(configName, dirPath):
    try:
        newFile = open(dirPath + '/makefile', 'w+')
        newFile.write('#################################################################\n')
        newFile.write('# Automatically-generated file. Do not edit!\n')
        newFile.write('#################################################################\n\n')
        newFile.write('RM := rm -rf\n\n')

        #include things
        newFile.write('-include sources.mk\n')
        for subDir in subDirList:
            newFile.write('-include %s/subdir.mk\n' % subDir)
        newFile.write('\n')

        newFile.write('OS_SUFFIX := $(subst Linux,linux,$(subst Darwin/x86_64,darwin,\
$(shell uname -s)/$(shell uname -m)))\n')
        newFile.write('-include objects.mk\n')


        #include conditional things
        newFile.write('ifneq ($(MAKECMDGOALS),clean)\n')
        newFile.write('ifneq ($(strip $(CU_DEPS)),)\n')
        newFile.write('-include $(CU_DEPS)\n')
        newFile.write('endif\n')
        newFile.write('ifneq ($(strip $(C++_DEPS)),)\n')
        newFile.write('-include $(C++_DEPS)\n')
        newFile.write('endif\n')
        newFile.write('ifneq ($(strip $(C_DEPS)),)\n')
        newFile.write('-include $(C_DEPS)\n')
        newFile.write('endif\n')
        newFile.write('ifneq ($(strip $(CC_DEPS)),)\n')
        newFile.write('-include $(CC_DEPS)\n')
        newFile.write('endif\n')
        newFile.write('ifneq ($(strip $(CPP_DEPS)),)\n')
        newFile.write('-include $(CPP_DEPS)\n')
        newFile.write('endif\n')
        newFile.write('ifneq ($(strip $(CXX_DEPS)),)\n')
        newFile.write('-include $(CXX_DEPS)\n')
        newFile.write('endif\n')
        newFile.write('ifneq ($(strip $(C_UPPER_DEPS)),)\n')
        newFile.write('-include $(C_UPPER_DEPS)\n')
        newFile.write('endif\n')
        newFile.write('endif\n\n')

        #include GLUT LIBS
        newFile.write('ifeq ($(shell uname -s),Darwin)\n')
        newFile.write('GLUT_LIBS := -Xlinker -framework -Xlinker GLUT -Xlinker -framework\
 -Xlinker OpenGL\n')
        newFile.write('else\n')
        newFile.write('GLUT_LIBS := -lGL -lGLU -lglut\n')
        newFile.write('endif\n\n')

        # all taget
        newFile.write('all: %s\n\n' % targetNameDic[configName])

        # tool invocation
        newFile.write('%s: $(OBJS) $(USER_OBJS)\n' % targetNameDic[configName])
        newFile.write("\t@echo 'Building target: $@'\n")
        newFile.write("\t@echo 'Invoking: NVCC Linker'\n")
        newFile.write('\tnvcc --cudart static -Xlinker --export-dynamic\
 --relocatable-device-code=false -gencode arch=%s,code=%s -link -o "%s" \
$(OBJS) $(USER_OBJS) $(LIBS)\n' % (supportArch, supportCode, targetNameDic[configName]))
        newFile.write("\t@echo 'Finished building target: $@'\n")
        newFile.write("\t@echo ' '\n\n")

        # Other Targets
        newFile.write("clean:\n")
        newFile.write("\t-$(RM) $(CU_DEPS)$(OBJS)$(C++_DEPS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)\
$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) %s\n" % targetNameDic[configName])
        newFile.write("\t-@echo ' '\n\n")

        #PHONY & SECONDARY
        newFile.write(".PHONY: all clean dependents\n")
        newFile.write(".SECONDARY:\n")

        newFile.close() 
    
    except Exception as e:
        print 'ERROR: generateMakefile()'
        print str(e)
        exit(-1)

def generateObjects(dirPath):
    try:
        newFile = open(dirPath + '/objects.mk', 'w+')
        newFile.write('#################################################################\n')
        newFile.write('# Automatically-generated file. Do not edit!\n')
        newFile.write('#################################################################\n\n')
        newFile.write('USER_OBJS :=\n\n')
        newFile.write('LIBS :=')
        for lib in libList:
            newFile.write(' -l%s' % lib)
        newFile.write('\n')
        newFile.close() 
    except Exception as e:
        print 'ERROR: generateObjects()'
        print str(e)
        exit(-1)

def generateSources(dirPath):
    try:
        newFile = open(dirPath + '/sources.mk', 'w+')
        newFile.write('#################################################################\n')
        newFile.write('# Automatically-generated file. Do not edit!\n')
        newFile.write('#################################################################\n\n')

        newFile.write('O_SRCS := \n')
        newFile.write('CPP_SRCS := \n')
        newFile.write('C_UPPER_SRCS := \n')
        newFile.write('C_SRCS := \n')
        newFile.write('S_UPPER_SRCS := \n')
        newFile.write('OBJ_SRCS := \n')
        newFile.write('CU_SRCS := \n')
        newFile.write('ASM_SRCS := \n')
        newFile.write('CXX_SRCS := \n')
        newFile.write('C++_SRCS := \n')
        newFile.write('CC_SRCS := \n')
        newFile.write('CU_DEPS := \n')
        newFile.write('OBJS := \n')
        newFile.write('C++_DEPS := \n')
        newFile.write('C_DEPS := \n')
        newFile.write('CC_DEPS := \n')
        newFile.write('CPP_DEPS := \n')
        newFile.write('EXECUTABLES := \n')
        newFile.write('CXX_DEPS := \n')
        newFile.write('C_UPPER_DEPS := \n')
        newFile.write('\n')

        newFile.write('SUBDIRS := \\\n')
        for subDir in subDirList:
            newFile.write('%s \\\n' % subDir)    

        newFile.close() 
    except Exception as e:
        print 'ERROR: generateSources()'
        print str(e)
        exit(-1)

def addSubDir(dirPath, relPath):
    try:
        for fileName in os.listdir(dirPath):
            filePath = dirPath + '/%s' % fileName
            if not os.path.isdir(filePath):
                continue

            addDirPath = relPath + '/%s' % fileName
            subDirList.append(addDirPath)
            addSubDir(filePath, addDirPath)

    except Exception as e:
        print 'ERROR: addSubDir()'
        print str(e)
        exit(-1)

def generateSubMakefile(configName, dirPath, relPath):
    srcHomeDir = os.environ[sourceHomeDirEnvName]
    buildDirPath = srcHomeDir + '/%s/%s' % (dirNameDic[configName], relPath)

    try:
        os.mkdir(buildDirPath)

        newFile = open(buildDirPath + '/subdir.mk', 'w+')
        newFile.write('#################################################################\n')
        newFile.write('# Automatically-generated file. Do not edit!\n')
        newFile.write('#################################################################\n\n')

        # search cpp, cu files
        cppFiles    = []
        objFiles    = []
        cuFiles     = []

        for fileName in os.listdir(dirPath):
            filePath = dirPath + '/%s' % fileName
            if os.path.isdir(filePath):
                continue

            ext = os.path.splitext(fileName)[-1]
            prefix = os.path.splitext(fileName)[0]
            if ext == '.cpp':
                cppFiles.append(prefix)
                objFiles.append(prefix)

            if ext == '.cu':
                cuFiles.append(prefix)
                objFiles.append(prefix)

        # declare cpp, cu files
        if len(cppFiles) > 0:
            newFile.write('CPP_SRCS += ')
            for prefix in cppFiles:
                newFile.write('\\\n../%s/%s.cpp' % (relPath, prefix))
            newFile.write('\n\n')

        if len(cuFiles) > 0:
            newFile.write('CU_SRCS += ')
            for prefix in cuFiles:
                newFile.write('\\\n../%s/%s.cu' % (relPath, prefix))
            newFile.write('\n\n')

        if len(cuFiles) > 0:
            newFile.write('CU_DEPS += ')
            for prefix in cuFiles:
                newFile.write('\\\n./%s/%s.d' % (relPath, prefix))
            newFile.write('\n\n')

        if len(objFiles) > 0:
            newFile.write('OBJS += ')
            for prefix in objFiles:
                newFile.write('\\\n./%s/%s.o' % (relPath, prefix))
            newFile.write('\n\n')

        if len(cppFiles) > 0:
            newFile.write('CPP_DEPS += ')
            for prefix in cppFiles:
                newFile.write('\\\n./%s/%s.d' % (relPath, prefix))
            newFile.write('\n\n')

        # build cpp files
        if len(cppFiles) > 0:
            # (1) make dep
            newFile.write(relPath + '/%.o: ../' + relPath + '/%.cpp\n')
            newFile.write("\t@echo 'Building file: $<'\n")
            newFile.write("\t@echo 'Invoking: NVCC compiler'\n")
            newFile.write("\tnvcc ")

            for symbol in symbolDic[configName]:
                newFile.write('-D%s ' % symbol)

            newFile.write("-I/usr/local/cuda/include ")     # cuda include path
            for incEnvVar in incEnvVarList:
                newFile.write("-I%s " % os.environ[incEnvVar])
            
            for incPath in subDirList:
                newFile.write("-I%s " % os.path.join(srcHomeDir, incPath))
    
            newFile.write("-G -g ")
            if "Debug" in configName:
                newFile.write("-O0 ")
            else:
                newFile.write("-O3 ")
            newFile.write("-Xcompiler -Wno-format-zero-length -std=c++11 ")
            newFile.write("-gencode arch=%s,code=%s " % (supportArch, supportCode))

            newFile.write('-odir "%s" ' % relPath)
            newFile.write('-M -o "$(@:%.o=%.d)" "$<"\n')

            # (2) make object
            newFile.write("\tnvcc ")

            for symbol in symbolDic[configName]:
                newFile.write('-D%s ' % symbol)

            newFile.write("-I/usr/local/cuda/include ")     # cuda include path
            for incEnvVar in incEnvVarList:
                newFile.write("-I%s " % os.environ[incEnvVar])
            
            for incPath in subDirList:
                newFile.write("-I%s " % os.path.join(srcHomeDir, incPath))

            newFile.write("-G -g ")
            if "Debug" in configName:
                newFile.write("-O0 ")
            else:
                newFile.write("-O3 ")
            newFile.write("-Xcompiler -Wno-format-zero-length -std=c++11 ")
            newFile.write("-gencode arch=%s,code=%s " % (supportArch, supportCode))

            newFile.write('--compile -x c++ -o  "$@" "$<"\n')

            newFile.write("\t@echo 'Finished building: $<'\n")
            newFile.write("\t@echo ' '\n\n")

        # build cu files
            # (1) make dep
            newFile.write(relPath + '/%.o: ../' + relPath + '/%.cu\n')
            newFile.write("\t@echo 'Building file: $<'\n")
            newFile.write("\t@echo 'Invoking: NVCC compiler'\n")
            newFile.write("\tnvcc ")

            for symbol in symbolDic[configName]:
                newFile.write('-D%s ' % symbol)

            newFile.write("-I/usr/local/cuda/include ")     # cuda include path
            for incEnvVar in incEnvVarList:
                newFile.write("-I%s " % os.environ[incEnvVar])
            
            for incPath in subDirList:
                newFile.write("-I%s " % os.path.join(srcHomeDir, incPath))
    
            newFile.write("-G -g ")
            if "Debug" in configName:
                newFile.write("-O0 ")
            else:
                newFile.write("-O3 ")
            newFile.write("-Xcompiler -Wno-format-zero-length -std=c++11 ")
            newFile.write("-gencode arch=%s,code=%s " % (supportArch, supportCode))

            newFile.write('-odir "%s" ' % relPath)
            newFile.write('-M -o "$(@:%.o=%.d)" "$<"\n')

            # (2) make object
            newFile.write("\tnvcc ")

            for symbol in symbolDic[configName]:
                newFile.write('-D%s ' % symbol)

            newFile.write("-I/usr/local/cuda/include ")     # cuda include path
            for incEnvVar in incEnvVarList:
                newFile.write("-I%s " % os.environ[incEnvVar])
            
            for incPath in subDirList:
                newFile.write("-I%s " % os.path.join(srcHomeDir, incPath))

            newFile.write("-G -g ")
            if "Debug" in configName:
                newFile.write("-O0 ")
            else:
                newFile.write("-O3 ")
            newFile.write("-Xcompiler -Wno-format-zero-length -std=c++11 ")
            newFile.write("-gencode arch=%s,code=%s " % (supportArch, supportCode))

            newFile.write('--compile -x cu -o  "$@" "$<"\n')

            newFile.write("\t@echo 'Finished building: $<'\n")
            newFile.write("\t@echo ' '\n\n")

        newFile.close() 
    except Exception as e:
        print 'ERROR: generateSources()'
        print str(e)
        exit(-1)

###########################################################################################
# Main Function
###########################################################################################

# (1) check env varaible of source home dir
try:
    if not sourceHomeDirEnvName in os.environ:
        print 'ERROR: you mush specify $%s' % sourceHomeDirEnvName
        exit(-1)

    for incEnvVar in incEnvVarList:
        if not incEnvVar in os.environ:
            print 'ERROR: you mush specify $%s' % incEnvVar
            exit(-1)
        
    if not "INC_PATH_CIMG" in os.environ:
        print 'ERROR: you mush specify $INC_PATH_GNUPLOT\n'
        exit(-1)
    
    srcHomeDir = os.environ[sourceHomeDirEnvName]
    if not os.path.exists(srcHomeDir):
        print 'ERROR: %s does not exist. check $%s'\
            % (srcHomeDir, sourceHomeDirEnvName)
        exit(-1)

    if not os.path.isdir(srcHomeDir):
        print 'ERROR: %s is not directory. check $%s'\
            % (srcHomeDir, sourceHomeDirEnvName)
        exit(-1)

    if not os.path.exists(srcHomeDir + '/src'):
        print 'ERROR: %s/src does not exist. check $%s'\
            % (srcHomeDir + '/src', sourceHomeDirEnvName)
        exit(-1)

    if not os.path.isdir(srcHomeDir + '/src'):
        print 'ERROR: %s/src is not directory. check $%s'\
            % (srcHomeDir, sourceHomeDirEnvName)
        exit(-1)
except Exception as e:
    print 'ERROR: while check source home dir'
    print str(e)
    exit(-1)

# (2) add sub directories
subDirList.append('src')
addSubDir(srcHomeDir + '/src', 'src')

# (3) generate makefiles
for configName in configNameList:
    buildDirPath = createDirectory(configName)

    generateMakefile(configName, buildDirPath)
    generateObjects(buildDirPath)
    generateSources(buildDirPath)

    for subDir in subDirList:
        generateSubMakefile(configName, srcHomeDir + '/%s' % subDir, subDir)

