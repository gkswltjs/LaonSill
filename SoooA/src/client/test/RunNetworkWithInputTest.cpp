/**
 * @file RunNetworkWithInputTest.cpp
 * @date 2017-07-13
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "RunNetworkWithInputTest.h"
#include "common.h"
#include "StdOutLog.h"
#include "ClientAPI.h"
#include "Communicator.h"
#include "SysLog.h"


using namespace std;

#define NETWORK_FILEPATH            SPATH("examples/frcnn/frcnn_test_live.json")
#define TESTIMAGE_BASE_FILEPATH     SPATH("client/test/")

bool RunNetworkWithInputTest::runSimpleTest() {
    ClientError ret;
    ClientHandle handle;
    NetworkHandle netHandle;

    ret = ClientAPI::createHandle(handle, "localhost", Communicator::LISTENER_PORT); 
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::getSession(handle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::createNetworkFromFile(handle, string(NETWORK_FILEPATH), netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::buildNetwork(handle, netHandle, 2);
    SASSERT0(ret == ClientError::Success);

    const bool showResult = false;
    const string windowName = "result";
    int imageChannel;
    int height;
    int width;
    float* imageData;
    char scoreBuf[20];

    if (showResult) {
    	cv::namedWindow(windowName);
    }

    for (int i = 0; i < 4; i++) {
        // the content of the image should be filled according to the image.
        string path = TESTIMAGE_BASE_FILEPATH + to_string(i + 1) + ".jpg";
        cv::Mat image = cv::imread(path); 
        imageChannel = image.channels();
        height = image.rows;
        width = image.cols;
        image.convertTo(image, CV_32FC3);
        imageData = (float*)image.data;

        vector<BoundingBox> bboxArray;
        STDOUT_LOG("request object detection job");
        ret = ClientAPI::getObjectDetection(handle, netHandle, imageChannel, height, width,
            imageData, bboxArray);
        SASSERT0(ret == ClientError::Success);

        STDOUT_LOG(" bounding box count : %d", (int)bboxArray.size());
        for (int j = 0; j < bboxArray.size(); j++) {
            STDOUT_LOG(" rect #%d : (%d, %d, %d, %d), confidence : %f", j, 
                bboxArray[j].top, bboxArray[j].left, bboxArray[j].bottom, bboxArray[j].right,
                bboxArray[j].confidence);

            if (showResult) {
				cv::rectangle(image, cv::Point(bboxArray[j].left, bboxArray[j].top),
						cv::Point(bboxArray[j].right, bboxArray[j].bottom), cv::Scalar(255, 0, 0), 2);

				sprintf(scoreBuf, "%f", bboxArray[j].confidence);
				cv::putText(image, string(scoreBuf), cv::Point(bboxArray[j].left,
						bboxArray[j].top + 15.0f), 2, 0.5f, cv::Scalar(255, 0, 0));
            }
        }

        if (showResult) {
			image.convertTo(image, CV_8UC3);
			cv::imshow(windowName, image);
			cv::waitKey(0);
        }

        ret = ClientAPI::resetNetwork(handle, netHandle);
        SASSERT0(ret == ClientError::Success);
    }

    if (showResult) {
    	cv::destroyWindow(windowName);
    }


    ret = ClientAPI::destroyNetwork(handle, netHandle);
    SASSERT0(ret == ClientError::Success);
    ret = ClientAPI::releaseSession(handle);
    SASSERT0(ret == ClientError::Success);

    return true;
}


bool RunNetworkWithInputTest::runTest() {
    bool result = runSimpleTest();
    if (result) {
        STDOUT_LOG("*  - simple telcoware(run network with input) test is success");
    } else {
        STDOUT_LOG("*  - simple telcoware(run network with input) test is failed");
        return false;
    }

    return true;
}
