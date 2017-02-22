/*
 * RoITestInputLayer.cpp
 *
 *  Created on: Dec 16, 2016
 *      Author: jkim
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "RoITestInputLayer.h"
#include "PascalVOC.h"
#include "frcnn_common.h"
#include "RoIDBUtil.h"
#include "MockDataSet.h"


#define ROITESTINPUTLAYER_LOG 0

using namespace std;
using namespace cv;


template <typename Dtype>
RoITestInputLayer<Dtype>::RoITestInputLayer()
: InputLayer<Dtype>() {}

template <typename Dtype>
RoITestInputLayer<Dtype>::RoITestInputLayer(Builder* builder)
: InputLayer<Dtype>(builder) {
	this->numClasses = builder->_numClasses;
	this->pixelMeans = builder->_pixelMeans;

	initialize();
}

template <typename Dtype>
RoITestInputLayer<Dtype>::~RoITestInputLayer() {
	delete imdb;
}


template <typename Dtype>
void RoITestInputLayer<Dtype>::reshape() {
	// 입력 레이어의 경우 outputs만 사용자가 설정,
	// inputs에 대해 outputs와 동일하게 Data 참조하도록 강제한다.
	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < this->_outputs.size(); i++) {
			this->_inputs.push_back(this->_outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}

	Layer<Dtype>::_adjustInputShape();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		// "data"
		if (i == 0) {
			const vector<uint32_t> dataShape =
					{1, 3, vec_max(TEST_SCALES), TEST_MAX_SIZE};
			this->_inputData[0]->reshape(dataShape);
			this->_inputShape[0] = dataShape;

#if ROITESTINPUTLAYER_LOG
			printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(),
					dataShape[0], dataShape[1], dataShape[2], dataShape[3]);
#endif
		}
		// "im_info"
		else if (i == 1) {
			const vector<uint32_t> iminfoShape = {1, 1, 1, 3};
			this->_inputShape[1] = iminfoShape;
			this->_inputData[1]->reshape(iminfoShape);

#if ROITESTINPUTLAYER_LOG
			printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(),
					iminfoShape[0], iminfoShape[1], iminfoShape[2], iminfoShape[3]);
#endif
		}
	}
}



template <typename Dtype>
void RoITestInputLayer<Dtype>::feedforward() {
	reshape();
	getNextMiniBatch();
}

template <typename Dtype>
void RoITestInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();
	getNextMiniBatch();
}

template <typename Dtype>
void RoITestInputLayer<Dtype>::initialize() {
	imdb = combinedRoidb("voc_2007_trainval");

	const uint32_t numImages = imdb->imageIndex.size();
	this->_dataSet = new MockDataSet<Dtype>(1, 1, 1, numImages, 50, 1);

	this->perm.resize(numImages);
	iota(this->perm.begin(), this->perm.end(), 0);
	this->cur = 0;



	this->boxColors.push_back(cv::Scalar(10, 163, 240));
	this->boxColors.push_back(cv::Scalar(44, 90, 130));
	this->boxColors.push_back(cv::Scalar(239, 80, 0));
	this->boxColors.push_back(cv::Scalar(37, 0, 162));
	this->boxColors.push_back(cv::Scalar(226, 161, 27));

	this->boxColors.push_back(cv::Scalar(115, 0, 216));
	this->boxColors.push_back(cv::Scalar(0, 196, 164));
	this->boxColors.push_back(cv::Scalar(255, 0, 106));
	this->boxColors.push_back(cv::Scalar(23, 169, 96));
	this->boxColors.push_back(cv::Scalar(0, 138, 0));

	this->boxColors.push_back(cv::Scalar(138, 96, 118));
	this->boxColors.push_back(cv::Scalar(100, 135, 109));
	this->boxColors.push_back(cv::Scalar(0, 104, 250));
	this->boxColors.push_back(cv::Scalar(208, 114, 244));
	this->boxColors.push_back(cv::Scalar(0, 20, 229));

	this->boxColors.push_back(cv::Scalar(63, 59, 122));
	this->boxColors.push_back(cv::Scalar(135, 118, 100));
	this->boxColors.push_back(cv::Scalar(169, 171, 0));
	this->boxColors.push_back(cv::Scalar(255, 0, 170));
	this->boxColors.push_back(cv::Scalar(0, 193, 216));


}

template <typename Dtype>
IMDB* RoITestInputLayer<Dtype>::combinedRoidb(const string& imdb_name) {
	IMDB* imdb = getRoidb(imdb_name);
	return imdb;
}

template <typename Dtype>
IMDB* RoITestInputLayer<Dtype>::getRoidb(const string& imdb_name) {
	IMDB* imdb = getImdb(imdb_name);
	cout << "Loaded dataset " << imdb->name << " for testing ... " << endl;

	return imdb;
}

template <typename Dtype>
IMDB* RoITestInputLayer<Dtype>::getImdb(const string& imdb_name) {
	IMDB* imdb = new PascalVOC("test_sample", "2007",
			"/home/jkim/Dev/git/py-faster-rcnn/data/VOCdevkit2007",
			this->pixelMeans);
	imdb->loadGtRoidb();

	return imdb;
}




template <typename Dtype>
void RoITestInputLayer<Dtype>::getNextMiniBatch() {
	if (this->cur >= this->imdb->imageIndex.size())
		this->cur = 0;

	uint32_t index = this->perm[this->cur];

	const string imagePath = imdb->imagePathAt(index);
	Mat im = cv::imread(imagePath);
	//showImageMat(im, false);
	cout << "test image: " << imagePath <<
			" (" << im.rows << "x" << im.cols << ")" << endl;


	vector<string> boxLabelsText;
	for (uint32_t i = 0; i < imdb->roidb[index].boxes.size(); i++)
		boxLabelsText.push_back(imdb->convertIndToClass(imdb->roidb[index].gt_classes[i]));

	displayBoxesOnImage("TEST_GT", imagePath, 1, imdb->roidb[index].boxes,
			imdb->roidb[index].gt_classes, boxLabelsText, this->boxColors, 0, -1, false);


	imDetect(im);

	Util::imagePath = imagePath;

	this->cur++;
}

template <typename Dtype>
void RoITestInputLayer<Dtype>::imDetect(cv::Mat& im) {
	cv::Mat tempIm;

	// scale [0, 255] image to [0.0, 1.0] image
	im.convertTo(im, CV_32FC3, 1.0f/255.0f);
	im.copyTo(tempIm);
	float* tempImPtr = (float*)tempIm.data;
	float* imPtr = (float*)im.data;

	// BGR to RGB
	// bias image to mean 0
	const uint32_t channels = (uint32_t)im.channels();
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

	const vector<uint32_t> imShape = {(uint32_t)im.rows, (uint32_t)im.cols,
			channels};
	uint32_t imSizeMin = np_min(imShape, 0, 2);
	uint32_t imSizeMax = np_max(imShape, 0, 2);

	const uint32_t targetSize = TEST_SCALES[0];
	float imScale = float(targetSize) / float(imSizeMin);
	// Prevent the biggest axis from being more than MAX_SIZE
	if (np_round(imScale * imSizeMax) > TEST_MAX_SIZE)
		imScale = float(TEST_MAX_SIZE) / float(imSizeMax);

	cv::Mat imResized;
	cv::resize(im, imResized, cv::Size(), imScale, imScale, CV_INTER_LINEAR);
	imToBlob(imResized);

	//this->_inputData[1]->reshape({1, 1, 1, 3});
	Dtype* imInfo = this->_inputData[1]->mutable_host_data();
	imInfo[0] = this->_inputShape[0][2];
	imInfo[1] = this->_inputShape[0][3];
	imInfo[2] = imScale;
	//this->_inputShape[1] = {1, 1, 1, 3};
}

template <typename Dtype>
void RoITestInputLayer<Dtype>::imToBlob(Mat& im) {
	// Convert a list of images into a network input.
	// Assumes images are already prepared (means subtracted, BGR order, ...)

	vector<uint32_t> maxShape = {(uint32_t)im.rows, (uint32_t)im.cols,
			(uint32_t)im.channels()};

	const vector<uint32_t> inputShape = {1, maxShape[0], maxShape[1], 3};
	this->_inputData[0]->reshape(inputShape);
	this->_inputData[0]->set_host_data((Dtype*)im.data);

#if ROITESTINPUTLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[0]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif

	// Move channels (axis 3) to axis 1
	// Axis order will become: (batch elem, channel, height, width)
	const vector<uint32_t> channelSwap = {0, 3, 1, 2};
	this->_inputData[0]->transpose(channelSwap);
	this->_inputShape[0] = this->_inputData[0]->getShape();

	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			this->name.c_str(),
			this->_inputShape[0][0], this->_inputShape[0][1],
			this->_inputShape[0][2], this->_inputShape[0][3]);

#if ROITESTINPUTLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[0]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif
}

template<typename Dtype>
int RoITestInputLayer<Dtype>::getNumTrainData() {
    return this->_dataSet->getNumTrainData();
}

template<typename Dtype>
int RoITestInputLayer<Dtype>::getNumTestData() {
    return this->_dataSet->getNumTestData();
}

template<typename Dtype>
void RoITestInputLayer<Dtype>::shuffleTrainDataSet() {
    return this->_dataSet->shuffleTrainDataSet();
}

template class RoITestInputLayer<float>;
