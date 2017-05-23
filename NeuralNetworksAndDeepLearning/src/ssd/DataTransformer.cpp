/*
 * DataTransformer.cpp
 *
 *  Created on: May 17, 2017
 *      Author: jkim
 */

#include "DataTransformer.h"
#include "SysLog.h"

using namespace std;

template <typename Dtype>
void transformInv(const int num, const int singleImageSize,
		const int imageHeight, const int imageWidth,
		const int height, const int width, const vector<Dtype>& pixelMeans,
		const Dtype* dataData, Data<Dtype>& temp) {

	Dtype* tempData = temp.mutable_host_data();
	for (int i = 0; i < num; i++) {
		temp.reshape({1, 3, uint32_t(imageHeight), uint32_t(imageWidth)});
		std::copy(dataData + i * singleImageSize, dataData + (i + 1) * singleImageSize, tempData);

		// transpose
		temp.transpose({0, 2, 3, 1});

		// pixel mean
		for (int j = 0; j < singleImageSize; j += 3) {
			tempData[j + 0] += pixelMeans[0];
			tempData[j + 1] += pixelMeans[1];
			tempData[j + 2] += pixelMeans[2];
		}

		cv::Mat im = cv::Mat(imageHeight, imageWidth, CV_32FC3, tempData);
		cv::resize(im, im, cv::Size(width, height), 0, 0, CV_INTER_LINEAR);

		im.convertTo(im, CV_8UC3);
		//cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
		//cv::imshow(windowName, im);
		//cv::waitKey(0);
		//cv::destroyAllWindows();
	}
}

template void transformInv(const int num, const int singleImageSize,
		const int imageHeight, const int imageWidth,
		const int height, const int width, const vector<float>& pixelMeans,
		const float* dataData, Data<float>& data);

