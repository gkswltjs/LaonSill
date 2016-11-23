/*
 * RoIInputLayer.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: jkim
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "frcnn_common.h"
#include "Layer.h"
#include "RoIInputLayer.h"
#include "ImagePackDataSet.h"
#include "PascalVOC.h"
#include "RoIDBUtil.h"
#include "MockDataSet.h"

using namespace std;
using namespace cv;

template <typename Dtype>
RoIInputLayer<Dtype>::RoIInputLayer() {}

template <typename Dtype>
RoIInputLayer<Dtype>::RoIInputLayer(Builder* builder) : InputLayer<Dtype>(builder) {
	this->numClasses = builder->_numClasses;
	this->imsPerBatch = builder->_imsPerBatch;
	this->trainBatchSize = builder->_trainBatchSize;
	this->trainMaxSize = builder->_trainMaxSize;
	this->trainFgFraction = builder->_trainFgFraction;
	this->trainScales = builder->_trainScales;
	this->pixelMeans = builder->_pixelMeans;

	initialize();
}

template <typename Dtype>
RoIInputLayer<Dtype>::~RoIInputLayer() {
	// TODO Auto-generated destructor stub
	delete imdb;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::initialize() {
	//uint32_t rows, uint32_t cols, uint32_t channels, uint32_t numTrainData, uint32_t numTestData, uint32_t numLabels, uint32_t mode = 0
	this->_dataSet = new MockDataSet<Dtype>(1, 1, 1, 10, 1, 1);

	imdb = combinedRoidb("voc_2007_trainval");

	cout << imdb->roidb.size() << " roidb entries ... " << endl;

	// Train a Fast R-CNN network.
	filterRoidb(imdb->roidb);

	cout << "Computing bounding-box regression targets ... " << endl;

	RoIDBUtil::addBboxRegressionTargets(imdb->roidb, bboxMeans, bboxStds);
	cout << "done" << endl;


	shuffleRoidbInds();

}

template <typename Dtype>
void RoIInputLayer<Dtype>::reshape() {
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

					{this->imsPerBatch, 3, vec_max(this->trainScales), this->trainMaxSize};
			this->_inputShape[0] = dataShape;
			this->_inputData[0]->shape(dataShape);

			printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(),
					dataShape[0], dataShape[1], dataShape[2], dataShape[3]);
		}
		// "im_info"
		else if (i == 1) {
			const vector<uint32_t> iminfoShape = {1, 1, 1, 3};
			this->_inputShape[1] = iminfoShape;
			this->_inputData[1]->shape(iminfoShape);

			printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(),
					iminfoShape[0], iminfoShape[1], iminfoShape[2], iminfoShape[3]);
		}
		// "gt_boxes"
		else if (i == 2) {
			const vector<uint32_t> gtboxesShape = {1, 1, 1, 4};
			this->_inputShape[2] = gtboxesShape;
			this->_inputData[2]->shape(gtboxesShape);

			printf("<%s> layer' output-2 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(),
					gtboxesShape[0], gtboxesShape[1], gtboxesShape[2], gtboxesShape[3]);
		}
	}

}




template <typename Dtype>
void RoIInputLayer<Dtype>::feedforward() {
	reshape();
	getNextMiniBatch();
}

template <typename Dtype>
void RoIInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();
	getNextMiniBatch();
}





template <typename Dtype>
IMDB* RoIInputLayer<Dtype>::getImdb(const string& imdb_name) {
	IMDB* imdb = new PascalVOC("trainval_sample", "2007",
			"/home/jkim/Dev/git/py-faster-rcnn/data/VOCdevkit2007");
	imdb->loadGtRoidb();

	return imdb;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::getTrainingRoidb(IMDB* imdb) {
	cout << "Appending horizontally-flipped training examples ... " << endl;
	imdb->appendFlippedImages();
	cout << "done" << endl;

	cout << "Preparing training data ... " << endl;
	//rdl_roidb.prepare_roidb(imdb)
	cout << "done" << endl;
}

template <typename Dtype>
IMDB* RoIInputLayer<Dtype>::getRoidb(const string& imdb_name) {
	IMDB* imdb = getImdb(imdb_name);
	cout << "Loaded dataset " << imdb->name << " for training ... " << endl;
	getTrainingRoidb(imdb);

	return imdb;
}

template <typename Dtype>
IMDB* RoIInputLayer<Dtype>::combinedRoidb(const string& imdb_name) {
	IMDB* imdb = getRoidb(imdb_name);
	return imdb;
}


template <typename Dtype>
bool RoIInputLayer<Dtype>::isValidRoidb(RoIDB& roidb) {
	// Valid images have
	// 	(1) At least one foreground RoI OR
	// 	(2) At least one background RoI

	roidb.max_overlaps;
	vector<uint32_t> fg_inds, bg_inds;
	// find boxes with sufficient overlap
	roidb.print();
	np_where_s(roidb.max_overlaps, GE, TRAIN_FG_THRESH, fg_inds);
	// select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
	np_where(roidb.max_overlaps, {LT, LE}, {TRAIN_BG_THRESH_HI, TRAIN_BG_THRESH_LO}, bg_inds);

	// image is only valid if such boxes exist
	return (fg_inds.size() > 0 || bg_inds.size() > 0);
}

template <typename Dtype>
void RoIInputLayer<Dtype>::filterRoidb(vector<RoIDB>& roidb) {
	// Remove roidb entries that have no usable RoIs.

	const uint32_t numRoidb = roidb.size();
	for (int i = numRoidb-1; i >= 0; i--) {
		if (!isValidRoidb(roidb[i])) {
			roidb.erase(roidb.begin()+i);
		}
	}

	const uint32_t numAfter = roidb.size();
	cout << "Filtered " << numRoidb - numAfter << " roidb entries: " <<
			numRoidb << " -> " << numAfter << endl;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::shuffleRoidbInds() {
	// Randomly permute the training roidb
	// if cfg.TRAIN.ASPECT_GROUPING

	/*
	vector<uint32_t> horzInds;
	vector<uint32_t> vertInds;
	const vector<RoIDB>& roidb = imdb->roidb;
	const uint32_t numRoidbs = roidb.size();
	for (uint32_t i = 0; i < numRoidbs; i++) {
		// roidb 이미지에 대해 landscape, portrait 이미지 구분
		if (roidb[i].width >= roidb[i].height)
			horzInds.push_back(i);
		else
			vertInds.push_back(i);
	}

	// landscape, portrait 이미지 인덱스 셔플
	random_shuffle(horzInds.begin(), horzInds.end());
	random_shuffle(vertInds.begin(), vertInds.end());
	horzInds.insert(horzInds.end(), vertInds.begin(), vertInds.end());

	const uint32_t numRoidbsHalf = numRoidbs/2;
	vector<vector<uint32_t>> inds(numRoidbsHalf);
	for (uint32_t i = 0; i < numRoidbsHalf; i++) {
		inds[i].resize(2);
		inds[i][0] = horzInds[i*2];
		inds[i][1] = horzInds[i*2+1];
	}
	random_shuffle(inds.begin(), inds.end());

	this->perm.resize(numRoidbs);
	for (uint32_t i = 0; i < numRoidbsHalf; i++) {
		perm[i*2] = inds[i][0];
		perm[i*2+1] = inds[i][1];
	}
	*/

	const uint32_t numRoidb = imdb->roidb.size();
	this->perm.resize(numRoidb);
	for (uint32_t i = 0; i < numRoidb; i++)
		this->perm[i] = i;
	this->cur = 0;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::getNextMiniBatch() {
	// Return the blobs to be used for the next minibatch.
	vector<uint32_t> inds;
	getNextMiniBatchInds(inds);

	vector<RoIDB> minibatchDb;
	for (uint32_t i = 0; i < inds.size(); i++) {
		minibatchDb.push_back(this->imdb->roidb[inds[i]]);
	}

	getMiniBatch(minibatchDb);
}

template <typename Dtype>
void RoIInputLayer<Dtype>::getNextMiniBatchInds(vector<uint32_t>& inds) {
	// Return the roidb indices for the next minibatch.
	if (this->cur + this->imsPerBatch >= this->imdb->roidb.size())
		shuffleRoidbInds();

	inds.clear();
	inds.insert(inds.end(), this->perm.begin()+this->cur,
			this->perm.begin()+this->imsPerBatch);
	this->cur += this->imsPerBatch;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::getMiniBatch(const vector<RoIDB>& roidb) {
	// Given a roidb, construct a minibatch sampled from it.

	const uint32_t numImages = roidb.size();
	// Sample random scales to use for each image in this batch
	vector<uint32_t> randomScaleInds;
	npr_randint(0, this->trainScales.size(), numImages, randomScaleInds);

	assert(this->trainBatchSize % numImages == 0);

	uint32_t roisPerImage = this->trainBatchSize / numImages;
	uint32_t fgRoisPerImage = np_round(this->trainFgFraction * roisPerImage);

	// Get the input image blob
	vector<float> imScales;
	getImageBlob(roidb, randomScaleInds, imScales);

	// if cfg.TRAIN.HAS_RPN
	assert(imScales.size() == 1);	// Single batch only
	assert(roidb.size() == 1);		// Single batch only

	// gt boxes: (x1, y1, x2, y2, cls)
	vector<uint32_t> gtInds;
	np_where_s(roidb[0].gt_classes, NE, (uint32_t)0, gtInds);

	const uint32_t numGtInds = gtInds.size();
	vector<vector<float>> gt_boxes(numGtInds);
	for (uint32_t i = 0; i < numGtInds; i++) {
		gt_boxes[i].resize(5);
		gt_boxes[i][0] = roidb[0].boxes[gtInds[i]][0] * imScales[0];
		gt_boxes[i][1] = roidb[0].boxes[gtInds[i]][1] * imScales[0];;
		gt_boxes[i][2] = roidb[0].boxes[gtInds[i]][2] * imScales[0];;
		gt_boxes[i][3] = roidb[0].boxes[gtInds[i]][3] * imScales[0];;
		gt_boxes[i][4] = roidb[0].gt_classes[gtInds[i]];
	}

	// 벡터를 Data로 변환하는 유틸이 필요!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
	// im_info
	vector<vector<float>> imInfoData;
	imInfoData.push_back({this->_inputShape[0][2], this->_inputShape[0][3], imScales[0]});
	fillDataWith2dVec(imInfoData, this->_inputData[1]);
	//fillDataWith2dVec(inputShape1, this->_inputData[1]);
	this->_inputShape[1] = {1, 1, 1, 3};
	Data<Dtype>::printConfig = true;
	this->_inputData[1]->print_data();

	// gt_boxes
	fillDataWith2dVec(gt_boxes, this->_inputData[2]);
	this->_inputShape[2] = {1, 1, (uint32_t)gt_boxes.size(), (uint32_t)gt_boxes[0].size()};
	this->_inputData[2]->print_data();
	Data<Dtype>::printConfig = false;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::getImageBlob(const vector<RoIDB>& roidb,
		const vector<uint32_t>& scaleInds, vector<float>& imScales) {
	imScales.clear();

	vector<Mat> processedIms;
	// Builds an input blob from the images in the roidb at the specified scales.
	const uint32_t numImages = roidb.size();
	for (uint32_t i = 0; i < numImages; i++) {
		Mat im = cv::imread(roidb[i].image);
		im.convertTo(im, CV_32F);

		if (roidb[i].flipped)
			cv::flip(im, im, 1);

		uint32_t targetSize = this->trainScales[scaleInds[i]];
		float imScale = prepImForBlob(im, this->pixelMeans,
				targetSize, this->trainMaxSize);

		imScales.push_back(imScale);
		processedIms.push_back(im);
	}

	// create a blob to hold the input images
	imListToBlob(processedIms);
}


template <typename Dtype>
float RoIInputLayer<Dtype>::prepImForBlob(cv::Mat& im, const vector<float>& pixelMeans,
		const uint32_t targetSize, const uint32_t maxSize) {
	// Mean subtract and scale an image for use in a blob
	// cv::Mat, BGR이 cols만큼 반복, 다시 해당 row가 rows만큼 반복
	const uint32_t channels = im.channels();
	assert(channels == pixelMeans.size());

	float* imPtr = (float*)im.data;
	uint32_t rowUnit, colUnit;
	for (uint32_t i = 0; i < im.rows; i++) {
		rowUnit = i * im.cols * channels;
		for (uint32_t j = 0; j < im.cols; j++) {
			colUnit = j * channels;
			for (uint32_t k = 0; k < channels; k++) {
				// cv::Mat의 경우 RGB가 아닌 BGR로 데이터가 뒤집어져 있음.
				//imPtr[rowUnit + colUnit + k] -= pixelMeans[channels-k-1];
				// pixel means도 BGR로 뒤집어져 있음.
				imPtr[rowUnit + colUnit + k] -= pixelMeans[k];
			}
		}
	}

	const vector<uint32_t> imShape = {(uint32_t)im.cols, (uint32_t)im.rows,
			channels};
	uint32_t imSizeMin = np_min(imShape, 0, 2);
	uint32_t imSizeMax = np_max(imShape, 0, 2);
	float imScale = float(targetSize) / float(imSizeMin);
	// Prevent the biggest axis from being more than MAX_SIZE
	if (np_round(imScale * imSizeMax) > maxSize)
		imScale = float(maxSize) / float(imSizeMax);

	cv::resize(im, im, cv::Size(), imScale, imScale, CV_INTER_LINEAR);
	cout << "resized to [" << im.rows << ", " << im.cols << "]" << endl;

	return imScale;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::imListToBlob(vector<Mat>& ims) {
	// Convert a list of images into a network input.
	// Assumes images are already prepared (means subtracted, BGR order, ...)

	const uint32_t numImages = ims.size();

	vector<uint32_t> maxShape;
	vector<vector<uint32_t>> imShapes;
	for (uint32_t i = 0; i < numImages; i++)
		imShapes.push_back({(uint32_t)ims[i].rows, (uint32_t)ims[i].cols,
		(uint32_t)ims[i].channels()});
	np_array_max(imShapes, maxShape);

	const vector<uint32_t> inputShape = {numImages, 3, maxShape[0], maxShape[1]};
	this->_inputData[0]->reshape(inputShape);
	this->_inputShape[0] = inputShape;

	Dtype* dataPtr = this->_inputData[0]->mutable_host_data();

	const uint32_t batchSize = this->_inputData[0]->getCountByAxis(1);
	const uint32_t channelSize = this->_inputData[0]->getCountByAxis(2);
	const uint32_t rowSize = this->_inputData[0]->getCountByAxis(3);

	for (uint32_t b = 0; b < numImages; b++) {
		Mat& im = ims[b];
		float* ptr = (float*)im.data;

		for (uint32_t r = 0; r < im.rows; r++) {
			for (uint32_t c = 0; c < im.cols; c++) {
				// BGR to RGB
				dataPtr[b*batchSize + 0*channelSize + r*rowSize + c] = ptr[r*im.cols*3 + c*3 + 2];
				dataPtr[b*batchSize + 1*channelSize + r*rowSize + c] = ptr[r*im.cols*3 + c*3 + 1];
				dataPtr[b*batchSize + 2*channelSize + r*rowSize + c] = ptr[r*im.cols*3 + c*3 + 0];
			}
		}
	}

	// Move channels (axis 3) to axis 1
	// Axis order will become: (batch elem, channel, height, width)

}



template class RoIInputLayer<float>;
































