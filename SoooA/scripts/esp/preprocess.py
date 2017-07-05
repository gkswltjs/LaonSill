#!/usr/bin/python
#
# FIXME: should check error but .... it is a very trivial program & I'm too lazy :)
#
import os

espLabelDirPath = "/data/ESP-ImageSet/LABELS"
espKeywordsFilePath = "/data/ESP-ImageSet/keywords.txt"

def preprocess():
    keywords = []

    fileKeyword = open(espKeywordsFilePath, 'w')
    fileNameList = os.listdir(espLabelDirPath)
    for fileName in fileNameList:
        labelFile = open(espLabelDirPath + "/" + fileName, 'r')
        lines = labelFile.readlines()

        for line in lines:
            keyword = line[0:-1]
            if keyword not in keywords:
                keywords.append(keyword)

        labelFile.close()

    for keyword in keywords:
        fileKeyword.write(keyword + '\n')

    fileKeyword.close()


preprocess()
