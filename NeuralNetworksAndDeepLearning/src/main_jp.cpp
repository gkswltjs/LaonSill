

#include <stddef.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <map>
#include <cfloat>
#include <cassert>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Data.h"


#include "3rd_party/tinyxml2/tinyxml2.h"


using namespace std;
using namespace tinyxml2;
using namespace cv;


void opencv_test();
template <typename PtrType, typename PrtType>
void print_mat(cv::Mat& im);
void show_mat(cv::Mat& im);
void convert(cv::Mat& im, const vector<float>& pixelMeans);

int main_(void) {

	opencv_test();

	return 0;
}


void opencv_test() {
	cv::Mat image = cv::imread("/home/jkim/Downloads/sampleR32G64B128.png", CV_LOAD_IMAGE_COLOR);
	print_mat<unsigned char, unsigned int>(image);

	convert(image, {0.1, 0.2, 0.3});
	print_mat<float, float>(image);
}

template <typename PtrType, typename PrtType>
void print_mat(cv::Mat& im) {
	cout << "rows: " << im.rows << ", cols: " << im.cols <<
				", channels: " << im.channels() << endl;

	const size_t numImElems = im.rows*im.cols*im.channels();
	const int rowElemSize = im.cols*im.channels();
	const int colElemSize = im.channels();

	for (int i = 0; i < im.rows; i++) {
		for (int j = 0; j < im.cols; j++) {
			cout << "[";
			for (int k = 0; k < im.channels(); k++) {
				cout << (PrtType)((PtrType*)im.data)[i*rowElemSize+j*colElemSize+k] << ",";
			}
			cout << "],";
		}
		cout << endl;
	}
}

void show_mat(cv::Mat& im) {
	namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);
	imshow("MyWindow", im);

	cv::resize(im, im, cv::Size(), 1.5, 1.5, CV_INTER_LINEAR);

	namedWindow("resize", CV_WINDOW_AUTOSIZE);
	imshow("resize", im);

	waitKey(0);
	destroyAllWindows();
}


void convert(cv::Mat& im, const vector<float>& pixelMeans) {
	// Mean subtract and scale an image for use in a blob
	// cv::Mat, BGR이 cols만큼 반복, 다시 해당 row가 rows만큼 반복
	const uint32_t channels = im.channels();
	// XXX: 3채널 컬러 이미지로 강제
	assert(channels == 3);
	assert(channels == pixelMeans.size());

	Mat tempIm;
	im.convertTo(im, CV_32FC3, 1.0f/255.0f);
	im.copyTo(tempIm);

	float* imPtr = (float*)im.data;
	float* tempImPtr = (float*)tempIm.data;
	uint32_t rowUnit, colUnit;
	for (uint32_t i = 0; i < im.rows; i++) {
		rowUnit = i * im.cols * channels;
		for (uint32_t j = 0; j < im.cols; j++) {
			colUnit = j * channels;

			// imPtr: target, reordered as rgb
			// tempImPtr: source, ordered as bgr
			// pixelMeans: ordered as rgb
			imPtr[rowUnit + colUnit + 0] = tempImPtr[rowUnit + colUnit + 2] - pixelMeans[0];
			imPtr[rowUnit + colUnit + 1] = tempImPtr[rowUnit + colUnit + 1] - pixelMeans[1];
			imPtr[rowUnit + colUnit + 2] = tempImPtr[rowUnit + colUnit + 0] - pixelMeans[2];
		}
	}
}
































