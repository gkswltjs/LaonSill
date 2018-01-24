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
#include <opencv2/highgui/highgui.hpp>

#include "SysLog.h"
#include "Param.h"
#include "ImageUtil.h"
#include "FileMgmt.h"
#include "IO.h"

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
#if 0
            testImage.at<cv::Vec3b>(i, j)[0] =
                (int)((data[baseIndex + index + channelElemCount * 2] + 1.0) * 127.5);
            testImage.at<cv::Vec3b>(i, j)[1] =
                (int)((data[baseIndex + index + channelElemCount * 1] + 1.0) * 127.5);
            testImage.at<cv::Vec3b>(i, j)[2] =
                (int)((data[baseIndex + index + channelElemCount * 0] + 1.0) * 127.5);
#else
            testImage.at<cv::Vec3b>(i, j)[0] =
                (int)(data[baseIndex + index + channelElemCount * 2] + 123.0);
            testImage.at<cv::Vec3b>(i, j)[1] =
                (int)(data[baseIndex + index + channelElemCount * 1] + 117.0);
            testImage.at<cv::Vec3b>(i, j)[2] =
                (int)(data[baseIndex + index + channelElemCount * 0] + 104.0);
#endif
        }
    }

    return testImage;
}

template<typename Dtype>
void ImageUtil<Dtype>::showImage(const Dtype* data, int nthImage, int channel, int row,
    int col) {
    cv::Mat testImage = makeImage(data, nthImage, channel, row, col);

    cv::imshow("test image", testImage);
    //cv::imwrite("/home/monhoney/yoyo.jpg", testImage);
    cv::waitKey(0);
}

template<typename Dtype>
void ImageUtil<Dtype>::saveImage(const Dtype* data, int imageCount, int channel, int row,
    int col, string folderName) {

    struct timeval val;
    struct tm* tmPtr;

    gettimeofday(&val, NULL);
    tmPtr = localtime(&val.tv_sec);

    string folderPath;
    if (strcmp(folderName.c_str(),  "") == 0) {
        char timeStr[1024];
        sprintf(timeStr, "%04d%02d%02d_%02d%02d%02d_%06ld",
            tmPtr->tm_year + 1900, tmPtr->tm_mon + 1, tmPtr->tm_mday, tmPtr->tm_hour,
            tmPtr->tm_min, tmPtr->tm_sec, val.tv_usec);

        folderPath = string(SPARAM(IMAGEUTIL_SAVE_DIR)) + "/" + string(timeStr);
    } else {
        folderPath = string(SPARAM(IMAGEUTIL_SAVE_DIR)) + "/" + folderName;
    }
    
    FileMgmt::checkDir(folderPath.c_str());

    for (int i = 0; i < imageCount; i++) {
        char imageName[1024];
        sprintf(imageName, "%d.jpg", i);

        string filePath = folderPath + "/" + string(imageName);
        cv::Mat newImage = makeImage(data, i, channel, row, col);
        cv::imwrite(filePath, newImage);
    }
}

template <typename Dtype>
void ImageUtil<Dtype>::dispDatum(const Datum* datum, const string& windowName) {
	cv::Mat cv_temp = DecodeDatumToCVMat(datum, true, true);
	dispCVMat(cv_temp, windowName);
}

template <typename Dtype>
void ImageUtil<Dtype>::dispCVMat(const cv::Mat& cv_img, const string& windowName) {
	cv::imshow(windowName, cv_img);
	cv::waitKey(0);
	cv::destroyWindow(windowName);
}

template <typename Dtype>
void ImageUtil<Dtype>::nms(std::vector<std::vector<float>>& proposals, 
        std::vector<float>& scores, const float thresh, std::vector<uint32_t>& keep) {

    vector<pair<int, float>> vec;

    for (int i = 0; i < scores.size(); i++) {
        vec.push_back(make_pair(i, scores[i]));
    }

    struct scoreCompareStruct {
        bool operator()(const pair<int, float> &left, const pair<int, float> &right) {
            return left.second > right.second;
        }
    };

    sort(vec.begin(), vec.end(), scoreCompareStruct());

    int maxScoreIndex = vec[0].first;

    bool live[vec.size()];
    for (int i = 0; i < vec.size(); i++)
        live[i] = true;

    for (int i = 0; i < vec.size() - 1; i++) {
        if (live[i] == false)
            continue;

        float x1 = proposals[vec[i].first][0];
        float y1 = proposals[vec[i].first][1];
        float x2 = proposals[vec[i].first][2];
        float y2 = proposals[vec[i].first][3];
        float area = (x2 - x1) * (y2 - y1);
        if (area == 0.0f) {
            live[i] = false;
            continue;
        }

        for (int j = i + 1; j < vec.size(); j++) {
            float tx1 = proposals[vec[j].first][0];
            float ty1 = proposals[vec[j].first][1];
            float tx2 = proposals[vec[j].first][2];
            float ty2 = proposals[vec[j].first][3];
            float tarea = (tx2 - tx1) * (ty2 - ty1);
            if (tarea == 0.0f) {
                live[j] = false;
                continue;
            }
            
            float xx1 = max(x1, tx1);
            float yy1 = max(y1, ty1);
            float xx2 = min(x2, tx2);
            float yy2 = min(y2, ty2);

            float w = max(xx2 - xx1, 0.0f);
            float h = max(yy2 - yy1, 0.0f);
            float inter = w * h;
            float iou = inter / (area + tarea - inter);

            if (iou > thresh)
                live[j] = false;
        }
    }

    for (int i = 0; i < vec.size(); i++) {
        if (live[i])
            keep.push_back(vec[i].first);
    }
}

template class ImageUtil<float>;
