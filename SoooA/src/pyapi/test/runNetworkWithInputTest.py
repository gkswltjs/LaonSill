#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from SoooA.ClientAPI import *

res = 512
coordRelative = 1

# (1) SoooA Client 핸들을 생성한다.
print "create handle"
handle = ClientHandle()
ret = handle.createHandle()

# (2) 세션을 얻는다.
print "get session"
ret = handle.getSession()

# (3) 네트워크를 생성한다.
NETWORK_FILEPATH = os.path.join(os.environ["SOOOA_SOURCE_PATH"], "examples", "SSD",
        "ssd_512_infer_live.json")
print "create network. network filepath : ", NETWORK_FILEPATH
handle.createNetworkFromFile(NETWORK_FILEPATH)

# (4) 네트워크를 빌드한다.
print "build network (epoch=2)"
handle.buildNetwork(2)

# (5) 오브젝트 디텍션 루프
TESTIMAGE_BASE_FILEPATH = os.path.join(os.environ["SOOOA_SOURCE_PATH"], "client", "test")
for i in range(4):
    imagePath = os.path.join(TESTIMAGE_BASE_FILEPATH, "%d.jpg" % (i + 1))

    img = cv2.imread(imagePath)
    height, width = img.shape[:2]

    resized_img = cv2.resize(img, (res, res))
    converted_img = np.array(resized_img).astype('f')

    ret, bboxes = handle.getObjectDetection(3, res, res, resized_img.flatten(), 20,
            coordRelative)

    for bbox in bboxes:
        if coordRelative == 1:
            left = int(bbox[0] * width)
            top = int(bbox[1] * height)
            right = int(bbox[2] * width)
            bottom = int(bbox[3] * height)
        else:
            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[2])
            bottom = int(bbox[3])

        cv2.rectangle(resized_img, (left, top), (right, bottom), (255, 255, 0))
        
    cv2.imshow('image', resized_img)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

# (6) 자원을 정리합니다.
print "cleanup resources"
handle.destroyNetwork()
handle.releaseSession()

