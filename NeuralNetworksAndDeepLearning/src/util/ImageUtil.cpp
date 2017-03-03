/**
 * @file ImageUtil.cpp
 * @date 2017-02-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <time.h>
#include <sys/time.h>

#include <string>

#include <opencv2/opencv.hpp>

#include "SysLog.h"
#include "Param.h"
#include "ImageUtil.h"
#include "FileMgmt.h"

using namespace std;

template<typename Dtype>
static cv::Mat makeImage(const Dtype* data, int nthImage, int channel, int row, int col);

template<typename Dtype>
cv::Mat makeImage(const Dtype* data, int nthImage, int channel, int row,
    int col) {
    // FIXME: 현재는 unsigned char 형으로만 이미지를 표현하고 있습니다.
    //        추후에 변경이 필요할 수 있습니다.
    //        data는 0.0 ~ 1.0 범위의 데이터입니다. 기본 RGB 값에서 255를 나눈 형태 입니다.
    //        따라서 unsigned char로 표현하는 경우에 다시 255를 곱합니다.
    //        Data => row major
    SASSERT0((channel == 3) || (channel == 1));
    int imageType;

    if (channel == 3) {
        imageType = CV_8UC3;
    } else {    // channel = 1
        imageType = CV_8UC1;
    }

    cv::Mat testImage(row, col, imageType);
    int channelElemCount = row * col;
    int imageElemCount = channelElemCount * channel;
    int baseIndex = imageElemCount * nthImage;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            int index = i * col + j;
            testImage.at<cv::Vec3b>(i, j)[0] =
                data[baseIndex + index + channelElemCount * 2] * 255;
            testImage.at<cv::Vec3b>(i, j)[1] =
                data[baseIndex + index + channelElemCount * 1] * 255;
            testImage.at<cv::Vec3b>(i, j)[2] =
                data[baseIndex + index + channelElemCount * 0] * 255;
        }
    }

    return testImage;
}

template<typename Dtype>
void ImageUtil<Dtype>::showImage(const Dtype* data, int nthImage, int channel, int row,
    int col) {
    cv::Mat testImage = makeImage(data, nthImage, channel, row, col);

    cv::imshow("test image", testImage);
    cv::waitKey(0);
}

template<typename Dtype>
void ImageUtil<Dtype>::saveImage(const Dtype* data, int imageCount, int channel, int row,
    int col) {

    struct timeval val;
    struct tm* tmPtr;

    gettimeofday(&val, NULL);
    tmPtr = localtime(&val.tv_sec);

    char timeStr[1024];
    sprintf(timeStr, "%04d%02d%02d_%02d%02d%02d_%06ld",
        tmPtr->tm_year + 1900, tmPtr->tm_mon + 1, tmPtr->tm_mday, tmPtr->tm_hour,
        tmPtr->tm_min, tmPtr->tm_sec, val.tv_usec);

    string folderPath = string(SPARAM(IMAGEUTIL_SAVE_DIR)) + "/" + string(timeStr);
    
    FileMgmt::checkDir(folderPath.c_str());

    for (int i = 0; i < imageCount; i++) {
        char imageName[1024];
        sprintf(imageName, "%d.jpg", i);

        string filePath = folderPath + "/" + string(imageName);
        cv::Mat newImage = makeImage(data, i, channel, row, col);
        cv::imwrite(filePath, newImage);
    }
}

template class ImageUtil<float>;
