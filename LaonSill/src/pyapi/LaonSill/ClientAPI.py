# -*- coding: utf-8 -*-

from ctypes import *
libLaonSill = CDLL('libLaonSillClient.so.1.0.1')

MAX_MESAURE_ITEMCOUNT=20
MAX_MEASURE_ITEMNAMELEN=64

# libLaonSillClient에서 제공하는 함수들을 준비한다.
funcTestYo = libLaonSill.testYo
funcTestYo.argtypes = [c_int, c_char_p, c_float]

funcGetSession = libLaonSill.getSession
funcGetSession.argtypes = [POINTER(c_int), c_char_p, c_int, POINTER(c_int), c_char_p]

funcReleaseSession = libLaonSill.releaseSession
funcReleaseSession.argtypes = [c_int, c_char_p, POINTER(c_int)]

funcCreateNetwork = libLaonSill.createNetwork
funcCreateNetwork.argtypes = [c_int, c_int, c_char_p, c_char_p, c_char_p]

funcCreateNetworkFromFile = libLaonSill.createNetworkFromFile
funcCreateNetworkFromFile.argtypes = [c_int, c_int, c_char_p, c_char_p, c_char_p]

funcDestroyNetwork = libLaonSill.destroyNetwork
funcDestroyNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p]

funcBuildNetwork = libLaonSill.buildNetwork
funcBuildNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]

funcResetNetwork = libLaonSill.resetNetwork
funcResetNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p]

funcRunNetwork = libLaonSill.runNetwork
funcRunNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]

funcRunNetworkMiniBatch = libLaonSill.runNetworkMiniBatch
funcRunNetworkMiniBatch.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int]

funcSaveNetwork = libLaonSill.saveNetwork
funcSaveNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_char_p]

funcLoadNetwork = libLaonSill.loadNetwork
funcLoadNetwork.argtypes = [c_int, c_char_p, c_int, c_char_p, c_char_p]


class BoundingBox(Structure):
    _fields_ = [("top", c_float), ("left", c_float), ("bottom", c_float), ("right", c_float),
                ("confidence", c_float), ("class_id", c_int)]

funcGetObjectDetection = libLaonSill.getObjectDetection
funcGetObjectDetection.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int,
                                   c_int, POINTER(c_float), c_void_p, c_int, c_int]


funcRunObjectDetectionWithInput = libLaonSill.runObjectDetectionWithInput
funcRunObjectDetectionWithInput.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int,
                                   c_int, POINTER(c_float), c_void_p, c_int, c_int]


funcRunClassificationWithInput = libLaonSill.runClassificationWithInput
funcRunClassificationWithInput.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int, c_int,
                                   c_int, POINTER(c_float), c_int, POINTER(c_int)]

funcGetMeasureItemName = libLaonSill.getMeasureItemName
funcGetMeasureItemName.argtypes = [c_int, c_char_p, c_char_p, c_int,
    POINTER(POINTER(c_char)), POINTER(c_int)]

funcGetMeasures = libLaonSill.getMeasures
funcGetMeasures.argtypes = [c_int, c_char_p, c_char_p, c_int, c_int, c_int, POINTER(c_int),
    POINTER(c_int), POINTER(c_float)]

class ClientHandle:
    def __init__(self):
        self.hasSession = c_int(0)
        self.bufLen = 1452      # see LaonSill MessageHeader.h source
        self.sockFD = c_int(-1)
        self.serverHostName = c_char_p("localhost")
        self.serverPortNum = c_int(20088)

        self.networkID = create_string_buffer(37)   # uuid size : 32 + 4 + 1
        self.isCreated = c_int(0)

    def createHandle(self, serverHostName = "localhost", serverPortNum=20088):
        self.serverHostName = c_char_p(serverHostName)
        self.serverPortNum = c_int(serverPortNum)
        self.buffer = create_string_buffer(self.bufLen)
        return 0    # success

    def getSession(self):
        ret = funcGetSession(byref(self.hasSession), self.serverHostName,
            self.serverPortNum, byref(self.sockFD), self.buffer)
        return ret

    def setSession(self, networkID):
        self.networkID = c_char_p(networkID)
        self.isCreated = c_int(1)

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
                        bbox.confidence, bbox.class_id])
        return ret, result_box

    def runObjectDetectionWithInput(self, channel, height, width, imageData, maxBoxCount,
            networkType):
        imageDataArray = (c_float * len(imageData))(*imageData)
        bboxArray = (BoundingBox * maxBoxCount)()

        ret = funcRunObjectDetectionWithInput(self.sockFD, self.buffer, self.isCreated, 
                self.networkID, c_int(channel), c_int(height), c_int(width),
                imageDataArray, bboxArray, c_int(maxBoxCount), c_int(networkType))

        result_box = []
        for bbox in bboxArray:
            if bbox.confidence > 0.000001:
                result_box.append([bbox.top, bbox.left, bbox.bottom, bbox.right,
                        bbox.confidence, bbox.class_id])
        return ret, result_box

    def runClassificationWithInput(self, channel, height, width, imageData, networkType):
        imageDataArray = (c_float * len(imageData))(*imageData)
        result = []
        label_index = c_int(-1)

        ret = funcRunClassificationWithInput(self.sockFD, self.buffer, self.isCreated, 
                self.networkID, c_int(channel), c_int(height), c_int(width),
                imageDataArray, c_int(networkType), byref(label_index))

        return ret, label_index.value

    def getMeasureItemName(self, networkID):
        itemCount = c_int(-1)        
        itemNameArray = (POINTER(c_char) * MAX_MESAURE_ITEMCOUNT)()
        for i in range(MAX_MESAURE_ITEMCOUNT):
            itemNameArray[i] = create_string_buffer(MAX_MEASURE_ITEMNAMELEN)
        
        ret = funcGetMeasureItemName(self.sockFD, self.buffer, c_char_p(networkID),
                c_int(MAX_MESAURE_ITEMCOUNT), itemNameArray, byref(itemCount))

        if ret != 0:
            return ret, []

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

        if ret != 0:
            ret, -1, []

        result = []

        for i in range(dataCount.value / itemCount):
            curr_result = []
            for j in range(itemCount):
                index = i * itemCount + j
                curr_result.append(measureArray[index])
            result.append(curr_result)

        return ret, startIterNum.value, result
