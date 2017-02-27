/**
 * @file ImageUtil.cpp
 * @date 2017-02-27
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <opencv2/opencv.hpp>

#include "SysLog.h"
#include "ImageUtil.h"

using namespace std;

template<typename Dtype>
void ImageUtil<Dtype>::showImage(const Dtype* data, int nthImage, int channel, int row,
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

    cv::imshow("test image", testImage);
    cv::waitKey(0);
}
