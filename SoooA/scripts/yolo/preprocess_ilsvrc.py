#!/usr/bin/python

import os

ILSVRC_ROOT_PATH = "/data/ilsvrc12_train/"
ILSVRC_CLASS_FILENAME = "class.txt"
ILSVRC_META_FILENAME = "ilsvrc.txt"

classDic = dict()

classFilePath = ILSVRC_ROOT_PATH + ILSVRC_CLASS_FILENAME
f = open(classFilePath, "r")
lines = f.readlines()

for line in lines:
    elems = line.split(',') 
    folderName = elems[0]
    classID = elems[1]
    classDic[folderName] = classID

f.close()

metaFilePath = ILSVRC_ROOT_PATH + ILSVRC_META_FILENAME
f = open(metaFilePath, "w")

isFirst = True

for folderName in classDic:
    classID = classDic[folderName]
    dirName = ILSVRC_ROOT_PATH + folderName

    fileNames = os.listdir(dirName)
    for fileName in fileNames:
        if "JPEG" in fileName:
            if isFirst:
                isFirst = False
            else:
                f.write('\n')
            filePath = dirName + "/" + fileName
            f.write("%s %d" % (filePath, int(classID)))

f.close()
