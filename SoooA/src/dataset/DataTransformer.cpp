/*
 * DataTransformer.cpp
 *
 *  Created on: Jul 19, 2017
 *      Author: jkim
 */

#include "DataTransformer.h"
#include "SysLog.h"
#include "PropMgmt.h"

using namespace std;


template <typename Dtype>
DataTransformer<Dtype>::DataTransformer(DataTransformParam* param) {
	if (param == NULL) {
		this->param = DataTransformParam();
	} else {
		this->param = *param;
	}
	this->hasMean = false;
	this->hasCropSize = false;
	this->hasScale = false;
	this->hasMirror = false;

	if (this->param.mean.size() > 0) {
		this->hasMean = true;

		// fill mean value to 3 regardless of number of image channels.
		if (this->param.mean.size() == 1) {
			for (int i = 1; i < 3; i++) {
				this->param.mean.push_back(this->param.mean[0]);
			}
		}
	}

	if (this->param.cropSize != 0.0) {
		this->hasCropSize = true;
	}

	if (this->param.scale != 1.0) {
		this->hasScale = true;
	}

	if (this->param.mirror != false) {
		this->hasMirror = true;
	}

	srand((uint32_t)time(NULL));
}



template <typename Dtype>
DataTransformer<Dtype>::~DataTransformer() {

}

template <typename Dtype>
void DataTransformer<Dtype>::transformWithMeanScale(Datum* datum, const vector<float>& mean,
		const float scale, Dtype* dataPtr) {

	const bool hasMean = (mean.size() > 0);


	string decode;
	if (datum->encoded) {

	}


	const string& data = datum->data;



	const int datum_channels = datum->channels;
	const int datum_height = datum->height;
	const int datum_width = datum->width;
	int height = datum_height;
	int width = datum_width;
	int h_off = 0;
	int w_off = 0;

	Dtype datum_element;
	int top_index, data_index;
	for (int c = 0; c < datum_channels; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
				top_index = (c * height + h) * width + w;
				datum_element =
						static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));

				if (hasMean) {
					dataPtr[top_index] = (datum_element - mean[c]) * scale;
				} else {
					dataPtr[top_index] = datum_element * scale;
				}
			}
		}
	}
}


template <typename Dtype>
void DataTransformer<Dtype>::transform(Datum* datum, Dtype* dataPtr) {
	string decode;
	if (datum->encoded) {

	}

	const string& data = datum->data;
	const int datum_channels = datum->channels;
	const int datum_height = datum->height;
	const int datum_width = datum->width;
	int height = datum_height;
	int width = datum_width;
	int h_off = 0;
	int w_off = 0;

	Dtype datum_element;
	int top_index, data_index;
	for (int c = 0; c < datum_channels; c++) {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
				top_index = (c * height + h) * width + w;
				datum_element =
						static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));

				if (this->hasMean) {
					dataPtr[top_index] = (datum_element - this->param.mean[c]) *
							this->param.scale;
				} else {
					dataPtr[top_index] = datum_element * this->param.scale;
				}
			}
		}
	}
}

template <typename Dtype>
void DataTransformer<Dtype>::transform(cv::Mat& im, Data<Dtype>* data, int batchIdx) {

	const int cropSize = this->param.cropSize;
	const int imgChannels = im.channels();
	// height and width may change due to pad or cropping
	const int imgHeight = im.rows;
	const int imgWidth = im.cols;

	// Check dimensions
	const int channels = data->channels();
	const int height = data->height();
	const int width = data->width();
	const int batches = data->batches();

	SASSERT0(channels == imgChannels);
	SASSERT0(height <= imgHeight);
	SASSERT0(width <= imgWidth);
	SASSERT0(batches >= 1);

	SASSERT(im.depth() == CV_8U, "Image data type must be unsigned byte");

	const Dtype scale = Dtype(this->param.scale);
	const bool doMirror = this->hasMirror && rand(2);

	SASSERT0(imgChannels > 0);
	SASSERT0(imgHeight >= cropSize);
	SASSERT0(imgWidth >= cropSize);

	int hOff = 0;
	int wOff = 0;
	cv::Mat croppedIm = im;
	if (this->hasCropSize) {
		SASSERT0(cropSize == height);
		SASSERT0(cropSize == width);
		// We only do random crop when we do training.
		if (SNPROP(status) == NetworkStatus::Train) {
			hOff = rand(imgHeight - cropSize + 1);
			wOff = rand(imgWidth - cropSize + 1);
		} else {
			hOff = (imgHeight - cropSize) / 2;
			wOff = (imgWidth - cropSize) / 2;
		}
		cv::Rect roi(wOff, hOff, cropSize, cropSize);
		croppedIm = im(roi);
	} else {
		SASSERT0(imgHeight == height);
		SASSERT0(imgWidth == width);
	}

	SASSERT0(croppedIm.data);

	Dtype* dataPtr = data->mutable_host_data() + data->offset(batchIdx);
	int topIndex;
	for (int h = 0; h < height; h++) {
		const uchar* ptr = croppedIm.ptr<uchar>(h);
		int imgIndex = 0;
		for (int w = 0; w < width; w++) {
			for (int c = 0; c < imgChannels; c++) {
				if (doMirror) {
					topIndex = (c * height + h) * width + (width - 1 - w);
				} else {
					topIndex = (c * height + h) * width + w;
				}
				// int topIndex = (c * height + h) * width + w;
				Dtype pixel = static_cast<Dtype>(ptr[imgIndex++]);
				if (this->hasMean) {
					dataPtr[topIndex] = (pixel - this->param.mean[c]) * scale;
				} else {
					dataPtr[topIndex] = pixel * scale;
				}
			}
		}
	}
}


















template <typename Dtype>
int DataTransformer<Dtype>::rand(int n) {
	SASSERT0(n > 0);
	return std::rand() % n;
}

























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






template class DataTransformer<float>;


























