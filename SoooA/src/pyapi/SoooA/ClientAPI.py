# -*- coding: utf-8 -*-

from ctypes import *
libSoooA = CDLL('libSoooAClient.so.1.0.1')

MAX_MESAURE_ITEMCOUNT=20
MAX_MEASURE_ITEMNAMELEN=64

# libSoooAClient에서 제공하는 함수들을 준비한다.
funcTestYo = libSoooA.testYo
funcTestYo.argtypes = [c_int, c_char_p, c_float]

funcGetSession = libSoooA.getSession
funcGetSession.argtypes = [POINTER(c_int), c_char_p, c_int, POINTER(c_int), c_char_p]

funcReleaseSession = libSoooA.releaseSession
funcReleaseSession.argtypes = [c_int, c_char_p, POINTER(c_int)]

funcCreateNetwork = libSoooA.createNetwork
funcCreateNetwork.argtypes = [c_int, c_int, c_char_p, c_char_p, c_char_p]

funcCreateNetworkFromFile = libSoooA.createNetworkFromFile
funcCreateNetworkFromFile.argtypes = [c_int, c_int, c_char_p, c_char_p, c_char_p]

funcDestroyNetwork = libSoooA.destroyNetwork
funcDestroyNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p]

funcBuildNetwork = libSoooA.buildNetwork
funcBuildNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]

funcResetNetwork = libSoooA.resetNetwork
funcResetNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p]

funcRunNetwork = libSoooA.runNetwork
funcRunNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]

funcRunNetworkMiniBatch = libSoooA.runNetworkMiniBatch
funcRunNetworkMiniBatch.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int]

funcSaveNetwork = libSoooA.saveNetwork
funcSaveNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_char_p]

funcLoadNetwork = libSoooA.loadNetwork
funcLoadNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_char_p]


class BoundingBox(Structure):
    _fields_ = [("top", c_float), ("left", c_float), ("bottom", c_float), ("right", c_float),
                ("confidence", c_float)]

funcGetObjectDetection = libSoooA.getObjectDetection
funcGetObjectDetection.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int,
                                   c_int, POINTER(c_float), c_void_p, c_int, c_int]

funcGetMeasureItemName = libSoooA.getMeasureItemName
funcGetMeasureItemName.argtypes = [c_int, c_char_p, c_char_p, c_int,
    POINTER(POINTER(c_char)), POINTER(c_int)]

funcGetMeasures = libSoooA.getMeasures
funcGetMeasures.argtypes = [c_int, c_char_p, c_char_p, c_int, c_int, c_int, POINTER(c_int),
    POINTER(c_int), POINTER(c_float)]

class ClientHandle:
    def __init__(self):
        self.hasSession = c_int(0)
        self.bufLen = 32 * 1024 * 1024          ## XXX: 32MB. should be fixed
        self.sockFD = c_int(-1)
        self.serverHostName = c_char_p("localhost")
        self.serverPortNum = c_int(20088)

        self.networkID = create_string_buffer(37)   # uuid size : 32 + 4 + 1
        self.isCreated = c_int(0)

    def createHandle(self, serverHostName = "localhost", serverPortNum=20088,
            bufLen = 32 * 1024 * 1024):
        self.serverHostName = c_char_p(serverHostName)
        self.serverPortNum = c_int(serverPortNum)
        self.bufLen = bufLen
        self.buffer = create_string_buffer(self.bufLen)
        return 0    # success

    def getSession(self):
        ret = funcGetSession(byref(self.hasSession), self.serverHostName,
            self.serverPortNum, byref(self.sockFD), self.buffer)
        return ret

    def releaseSession(self):
        ret = funcReleaseSession(self.sockFD, self.buffer, byref(self.hasSession))
        return ret

    def createNetwork(self, networkDef): 
        ret = funcCreateNetwork(self.sockFD, self.hasSession, self.buffer,
                c_char_p(networkDef), self.networkID)
        self.isCreated = c_int(1)
        return ret

    def createNetworkFromFile(self, filePathInServer):
        ret = funcCreateNetworkFromFile(self.sockFD, self.hasSession, self.buffer,
                c_char_p(filePathInServer), self.networkID)
        self.isCreated = c_int(1)
        return ret

    def destroyNetwork(self):
        ret = funcDestroyNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID)
        self.isCreated = c_int(0)
        return ret

    def buildNetwork(self, epochs):
        ret = funcBuildNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID,
                c_int(epochs))
        return ret

    def resetNetwork(self):
        ret = funcResetNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID)
        return ret

    def runNetwork(self, inference):
        ret = funcRunNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID,
                c_int(inference))
        return ret

    def runNetworkMiniBatch(self, inference, miniBatchIdx):
        ret = funcRunNetworkMiniBatch(self.sockFD, self.buffer, self.isCreated,
                self.networkID, c_int(inference), c_int(miniBatchIdx))
        return ret

    def saveNetwork(self, filePath):
        ret = funcSaveNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID,
                c_char_p(filePath))
        return ret

    def loadNetwork(self, filePath):
        ret = funcLoadNetwork(self.sockFD, self.buffer, self.isCreated, self.networkID,
                c_char_p(filePath))
        return ret

    def getObjectDetection(self, channel, height, width, imageData, maxBoxCount,
            coordRelative):
        imageDataArray = (c_float * len(imageData))(*imageData)
        bboxArray = (BoundingBox * maxBoxCount)()

        ret = funcGetObjectDetection(self.sockFD, self.buffer, self.isCreated, 
                self.networkID, c_int(channel), c_int(height), c_int(width),
                imageDataArray, bboxArray, c_int(maxBoxCount), c_int(coordRelative))

        result_box = []
        for bbox in bboxArray:
            if bbox.confidence > 0.000001:
                result_box.append([bbox.top, bbox.left, bbox.bottom, bbox.right,
                        bbox.confidence])
        return ret, result_box

    def getMeasureItemName(self, networkID):
        itemCount = c_int(-1)        
        itemNameArray = (POINTER(c_char) * MAX_MESAURE_ITEMCOUNT)()
        for i in range(MAX_MESAURE_ITEMCOUNT):
            itemNameArray[i] = create_string_buffer(MAX_MEASURE_ITEMNAMELEN)
        
        ret = funcGetMeasureItemName(self.sockFD, self.buffer, c_char_p(networkID),
                c_int(MAX_MESAURE_ITEMCOUNT), itemNameArray, byref(itemCount))

        result = []
        for i in range(itemCount.value):
            value = cast(itemNameArray[i], c_char_p).value
            result.append(value);

        return ret, result

    def getMeasures(self, networkID, itemCount, forwardSearch, start, count):
        assert itemCount > 0

        startIterNum = c_int(-1)
        dataCount = c_int(-1)
        measureArray = (c_float * (itemCount * count))()
       
        ret = funcGetMeasures(self.sockFD, self.buffer, c_char_p(networkID),
                c_int(int(forwardSearch)), c_int(start), c_int(count), byref(startIterNum),
                byref(dataCount), measureArray)

        result = []

        for i in range(dataCount.value / itemCount):
            curr_result = []
            for j in range(itemCount):
                index = i * itemCount + j
                curr_result.append(measureArray[index])
            result.append(curr_result)

        return ret, startIterNum.value, result
