
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "IO.h"
#include "SysLog.h"

using namespace std;


bool ReadImageToDatum(const string& filename, const int label, const int height,
		const int width, const int min_dim, const int max_dim, const bool is_color,
		const string& encoding, Datum* datum) {
	cv::Mat cv_img = ReadImageToCVMat(filename, height, width, min_dim, max_dim, is_color);

	if (cv_img.data) {
		SASSERT0(!encoding.size());
		CVMatToDatum(cv_img, datum);
		datum->label = label;
		return true;
	} else {
		return false;
	}
}

cv::Mat ReadImageToCVMat(const string& filename, const int height, const int width,
		const int min_dim, const int max_dim, const bool is_color) {
	cv::Mat cv_img;
	int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
	if (!cv_img_origin.data) {
		cout << "Could not open or find file " << filename << endl;
		return cv_img_origin;
	}
	SASSERT0(min_dim == 0 && max_dim == 0);
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}

	//for (int i = 0; i < 30; i++) {
	//	printf("%d,", cv_img.data[i]);
	//}
	//cout << endl;





	return cv_img;
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
	SASSERT0(cv_img.depth() == CV_8U);
	datum->channels = cv_img.channels();
	datum->height = cv_img.rows;
	datum->width = cv_img.cols;
	//datum->clear_data ...
	datum->float_data.clear();
	datum->encoded = false;

	int datum_channels = datum->channels;
	int datum_height = datum->height;
	int datum_width = datum->width;
	int datum_size = datum_channels * datum_height * datum_width;
	string buffer(datum_size, ' ');

	// all B / all G / all R 구조로 channel을 나누어서 저장
	for (int h = 0; h < datum_height; h++) {
		const uchar* ptr = cv_img.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < datum_width; w++) {
			for (int c = 0; c < datum_channels; c++) {
				int datum_index = (c * datum_height + h) * datum_width + w;
				buffer[datum_index] = static_cast<char>(ptr[img_index++]);
			}
		}
	}

	/*
	// test로 channel분리하지 않고 opencv가 제공하는 포맷 그대로 저장
	for (int h = 0; h < datum_height; h++) {
		const uchar* ptr = cv_img.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < datum_width; w++) {
			for (int c = 0; c < datum_channels; c++) {
				int datum_index = h * datum_width * datum_channels + img_index;
				buffer[datum_index] = static_cast<char>(ptr[img_index++]);

				//if (h == 0 && w < 10) {
					//printf("%d,", (uchar)buffer[datum_index]);
				//}
			}
		}
		//cout << endl;
	}
	//cout << endl;
	 */

	datum->data = buffer;
}

cv::Mat DecodeDatumToCVMat(const Datum* datum, bool is_color) {
	SASSERT0(datum->channels == 1 || datum->channels == 3);

	int cv_type;
	if (datum->channels == 3) {
		cv_type = CV_8UC3;
	} else if (datum->channels == 1) {
		cv_type = CV_8U;
	}

	cv::Mat cv_img(datum->height, datum->width, cv_type, (uchar*)datum->data.c_str());

	if (!cv_img.data) {
		cout << "Could not decode datum." << endl;
	}
	return cv_img;
}







