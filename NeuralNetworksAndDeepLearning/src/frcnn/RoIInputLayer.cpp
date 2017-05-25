/*
 * RoIInputLayer.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: jkim
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "frcnn_common.h"
#include "Layer.h"
#include "RoIInputLayer.h"
#include "ImagePackDataSet.h"
#include "PascalVOC.h"
#include "RoIDBUtil.h"
#include "MockDataSet.h"
#include "PropMgmt.h"

#define ROIINPUTLAYER_LOG 0

using namespace std;

template <typename Dtype>
RoIInputLayer<Dtype>::RoIInputLayer(const string& name)
: InputLayer<Dtype>(name) {
	initialize();
}

template <typename Dtype>
RoIInputLayer<Dtype>::RoIInputLayer(Builder* builder) : InputLayer<Dtype>(builder) {
	this->numClasses = builder->_numClasses;
	this->pixelMeans = builder->_pixelMeans;

	initialize();
}

template <typename Dtype>
RoIInputLayer<Dtype>::~RoIInputLayer() {
	delete this->imdb;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::initialize() {
	this->imdb = combinedRoidb("voc_2007_trainval");

	cout << this->imdb->roidb.size() << " roidb entries ... " << endl;
	this->_dataSet = new MockDataSet<Dtype>(1, 1, 1, this->imdb->roidb.size(), 50, 1);

	// Train a Fast R-CNN network.
	filterRoidb(this->imdb->roidb);

	cout << "Computing bounding-box regression targets ... " << endl;
	RoIDBUtil::addBboxRegressionTargets(this->imdb->roidb, this->bboxMeans, this->bboxStds);

	shuffleRoidbInds();





	if (!TRAIN_HAS_RPN) {
		const string path = "/home/jkim/Dev/SOOOA_HOME/network/proposal_target_layer.ptl";
		ifstream ifs(path, std::ios::in | std::ios::binary);

		uint32_t numData;
		ifs.read((char*)&numData, sizeof(uint32_t));

		cout << "numData: " << numData << endl;
		numData /= 5;

		proposalTargetDataList.resize(numData);

		for (uint32_t i = 0; i < numData; i++) {
			proposalTargetDataList[i].resize(5);
			for (uint32_t j = 0; j < 5; j++) {
				Data<float>* data = new Data<float>("", true);
				data->load(ifs);
				proposalTargetDataList[i][j] = data;
			}
		}
		ifs.close();
	}







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
void RoIInputLayer<Dtype>::reshape() {
	const vector<string>& outputs = SLPROP(Input, output);
	vector<string>& inputs = SLPROP(Input, input);
	if (inputs.size() < 1) {
		for (uint32_t i = 0; i < outputs.size(); i++) {
			inputs.push_back(outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}

	Layer<Dtype>::_adjustInputShape();

	const uint32_t numClasses = SLPROP(RoIInput, numClasses);
	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		if (TRAIN_HAS_RPN) {

			// "data"
			if (i == 0) {
				const vector<uint32_t> dataShape =
						{TRAIN_IMS_PER_BATCH, 3, vec_max(TRAIN_SCALES), TRAIN_MAX_SIZE};
				this->_inputData[0]->reshape(dataShape);
				this->_inputShape[0] = dataShape;

	#if ROIINPUTLAYER_LOG
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

	#if ROIINPUTLAYER_LOG
				printf("<%s> layer' output-1 has reshaped as: %dx%dx%dx%d\n",
						this->name.c_str(),
						iminfoShape[0], iminfoShape[1], iminfoShape[2], iminfoShape[3]);
	#endif
			}
			// "gt_boxes"
			else if (i == 2) {
				const vector<uint32_t> gtboxesShape = {1, 1, 1, 4};
				this->_inputShape[2] = gtboxesShape;
				this->_inputData[2]->reshape(gtboxesShape);

	#if ROIINPUTLAYER_LOG
				printf("<%s> layer' output-2 has reshaped as: %dx%dx%dx%d\n",
						this->name.c_str(),
						gtboxesShape[0], gtboxesShape[1], gtboxesShape[2], gtboxesShape[3]);
	#endif
			}
		} else {
			// "data"
			if (i == 0) {
				const vector<uint32_t> dataShape =
						{TRAIN_IMS_PER_BATCH, 3, vec_max(TRAIN_SCALES), TRAIN_MAX_SIZE};
				this->_inputData[0]->reshape(dataShape);
				this->_inputShape[0] = dataShape;

#if ROIINPUTLAYER_LOG
			printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
					this->name.c_str(),
					dataShape[0], dataShape[1], dataShape[2], dataShape[3]);
#endif
			}
			// "rois"
			else if (i == 1) {
				const vector<uint32_t> roisShape = {1, 1, 1, 5};
				this->_inputData[1]->reshape(roisShape);
				this->_inputShape[1] = roisShape;
			}
			// "labels"
			else if (i == 2) {
				const vector<uint32_t> labelsShape = {1, 1, 1, 1};
				this->_inputData[2]->reshape(labelsShape);
				this->_inputShape[2] = labelsShape;
			}
			// "bbox_targets"
			else if (i == 3) {
				const vector<uint32_t> bboxTargetsShape = {1, 1, 1, numClasses * 4};
				this->_inputData[3]->reshape(bboxTargetsShape);
				this->_inputShape[3] = bboxTargetsShape;
			}
			// "bbox_inside_weights"
			else if (i == 4) {
				const vector<uint32_t> bboxInsideWeights = {1, 1, 1, numClasses * 4};
				this->_inputData[4]->reshape(bboxInsideWeights);
				this->_inputShape[4] = bboxInsideWeights;
			}
			// "bbox_outside_weights"
			else if (i == 5) {
				const vector<uint32_t> bboxOutsideWeights = {1, 1, 1, numClasses * 4};
				this->_inputData[4]->reshape(bboxOutsideWeights);
				this->_inputShape[4] = bboxOutsideWeights;
			}
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
	const vector<float>& pixelMeans = SLPROP(RoIInput, pixelMeans);
	IMDB* imdb = new PascalVOC("trainval", "2007",
			"/home/jkim/Dev/git/py-faster-rcnn/data/VOCdevkit2007",
			pixelMeans);
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
	vector<uint32_t> fgInds, bgInds;
	// find boxes with sufficient overlap
#if ROIINPUTLAYER_LOG
	roidb.print();
#endif
	np_where_s(roidb.max_overlaps, GE, TRAIN_FG_THRESH, fgInds);
	// select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
	np_where(roidb.max_overlaps, {LT, LE}, {TRAIN_BG_THRESH_HI, TRAIN_BG_THRESH_LO}, bgInds);

	// image is only valid if such boxes exist
	return (fgInds.size() > 0 || bgInds.size() > 0);
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

#if TEST_MODE
	const uint32_t numRoidb = imdb->roidb.size();
	this->perm.resize(numRoidb);
	for (uint32_t i = 0; i < numRoidb; i++)
		this->perm[i] = i;
#else
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
#endif
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
		//cout << this->imdb->roidb[inds[i]].image << endl;
#if ROIINPUTLAYER_LOG
		this->imdb->roidb[inds[i]].print();
#endif
	}


	/*
	uint32_t index = inds[0];
	cout << "flipped: " << imdb->roidb[index].flipped << endl;
	vector<string> boxLabelsText;
	for (uint32_t i = 0; i < imdb->roidb[index].boxes.size(); i++)
		boxLabelsText.push_back(imdb->convertIndToClass(imdb->roidb[index].gt_classes[i]));

	displayBoxesOnImage("TRAIN_GT", imdb->roidb[index].getMat(), 1, imdb->roidb[index].boxes,
			imdb->roidb[index].gt_classes, boxLabelsText, this->boxColors, 0, -1, true);
			*/

	getMiniBatch(minibatchDb, inds);
}

template <typename Dtype>
void RoIInputLayer<Dtype>::getNextMiniBatchInds(vector<uint32_t>& inds) {
	// Return the roidb indices for the next minibatch.

	// 바운더리에서 마지막 아이템 처리하지 않고 처음 아이템으로 넘어가서 부등호 수정함.
	//if (this->cur + TRAIN_IMS_PER_BATCH >= this->imdb->roidb.size())
	if (this->cur + TRAIN_IMS_PER_BATCH > this->imdb->roidb.size())
		shuffleRoidbInds();

	inds.clear();
	inds.insert(inds.end(), this->perm.begin() + this->cur,
			this->perm.begin() + this->cur + TRAIN_IMS_PER_BATCH);

	cout << this->cur << ": ";
	this->cur += TRAIN_IMS_PER_BATCH;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::getMiniBatch(const vector<RoIDB>& roidb,
    const vector<uint32_t>& inds) {
	// Given a roidb, construct a minibatch sampled from it.

	const uint32_t numImages = roidb.size();
	// Sample random scales to use for each image in this batch
	vector<uint32_t> randomScaleInds;
	npr_randint(0, TRAIN_SCALES.size(), numImages, randomScaleInds);

	assert(TRAIN_BATCH_SIZE % numImages == 0);

	uint32_t roisPerImage = TRAIN_BATCH_SIZE / numImages;
	uint32_t fgRoisPerImage = np_round(TRAIN_FG_FRACTION * roisPerImage);

	// Get the input image blob
	vector<float> imScales;
	vector<cv::Mat> processedIms = getImageBlob(roidb, randomScaleInds, imScales);

	// if cfg.TRAIN.HAS_RPN
	assert(imScales.size() == 1);	// Single batch only
	assert(roidb.size() == 1);		// Single batch only




	if (TRAIN_HAS_RPN) {
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

		// gt_boxes
		fillDataWith2dVec(gt_boxes, this->_inputData[2]);
		this->_inputShape[2] =
            {1, 1, (uint32_t)gt_boxes.size(), (uint32_t)gt_boxes[0].size()};



		/*
		// 최종 scale된 input image, bounding box를 display
		vector<string> boxLabelsText;
		for (uint32_t j = 0; j < roidb[0].boxes.size(); j++)
			boxLabelsText.push_back(imdb->convertIndToClass(roidb[0].gt_classes[j]));

		displayBoxesOnImage("INPUT DATA", processedIms[0], 1, gt_boxes,
				roidb[0].gt_classes, boxLabelsText, this->boxColors, 0, -1, true);
				*/


#if ROIINPUTLAYER_LOG
		Data<Dtype>::printConfig = true;
		this->_inputData[1]->print_data();
		this->_inputData[2]->print_data();
		Data<Dtype>::printConfig = false;
#endif
	} else {

		const uint32_t ind = inds[0];

		const vector<uint32_t> roisShape = proposalTargetDataList[ind][0]->getShape();
		this->_inputData[1]->reshape(roisShape);
		this->_inputData[1]->set_host_data(proposalTargetDataList[ind][0]);
		this->_inputShape[1] = roisShape;

		const vector<uint32_t> labelsShape = proposalTargetDataList[ind][1]->getShape();
		this->_inputData[2]->reshape(labelsShape);
		this->_inputData[2]->set_host_data(proposalTargetDataList[ind][1]);
		this->_inputShape[2] = labelsShape;

		const vector<uint32_t> bboxTargetsShape = proposalTargetDataList[ind][2]->getShape();
		this->_inputData[3]->reshape(bboxTargetsShape);
		this->_inputData[3]->set_host_data(proposalTargetDataList[ind][2]);
		this->_inputShape[3] = bboxTargetsShape;

		const vector<uint32_t> bboxInsideWeightsShape =
            proposalTargetDataList[ind][3]->getShape();
		this->_inputData[4]->reshape(bboxInsideWeightsShape);
		this->_inputData[4]->set_host_data(proposalTargetDataList[ind][3]);
		this->_inputShape[4] = bboxInsideWeightsShape;

		const vector<uint32_t> bboxOutsideWeightsShape =
            proposalTargetDataList[ind][4]->getShape();
		this->_inputData[5]->reshape(bboxOutsideWeightsShape);
		this->_inputData[5]->set_host_data(proposalTargetDataList[ind][4]);
		this->_inputShape[5] = bboxOutsideWeightsShape;
	}
}

template <typename Dtype>
vector<cv::Mat> RoIInputLayer<Dtype>::getImageBlob(const vector<RoIDB>& roidb,
		const vector<uint32_t>& scaleInds, vector<float>& imScales) {
	imScales.clear();

	const vector<float>& pixelMeans = SLPROP(RoIInput, pixelMeans);
	vector<cv::Mat> processedIms;
	// Builds an input blob from the images in the roidb at the specified scales.
	const uint32_t numImages = roidb.size();
	for (uint32_t i = 0; i < numImages; i++) {
		cv::Mat im = roidb[i].getMat();
#if !ROIINPUTLAYER_LOG
		cout << "image: " << roidb[i].image.substr(roidb[i].image.length()-10) <<
				" (" << im.rows << "x" << im.cols << ")" << ", flip: " << roidb[i].flipped;
		//cout << "original: " << ((float*)im.data) << endl;
#endif
		// XXX: for test
		Util::imagePath = roidb[i].image;
		//Mat im = cv::imread("/home/jkim/Downloads/sampleR32G64B128.png");

		// 수정: IMDB.appendFlippedImages()에서 flip 처리
		//if (roidb[i].flipped)
		//	cv::flip(im, im, 1);

		/*
		cout << "flipped: " << roidb[i].flipped << endl;
		vector<string> boxLabelsText;
		for (uint32_t j = 0; j < roidb[i].boxes.size(); j++)
			boxLabelsText.push_back(imdb->convertIndToClass(roidb[i].gt_classes[j]));

		displayBoxesOnImage("TRAIN_GT", im, 1, roidb[i].boxes,
				roidb[i].gt_classes, boxLabelsText, this->boxColors, 0, -1, true);
				*/

		/*
		string windowName1 = "im purity test original";
		cv::namedWindow(windowName1, CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName1, im);
		//cv::waitKey(0);
		//cv::destroyAllWindows();
		 */
		//cout << "before: " << ((float*)im.data) << endl;

		uint32_t targetSize = TRAIN_SCALES[scaleInds[i]];
		cv::Mat imResized;
		float imScale = prepImForBlob(im, imResized, pixelMeans, targetSize,
				TRAIN_MAX_SIZE);

		cout << " -> <" << targetSize << ", " << imScale << "> (" <<
				imResized.rows << "x" << imResized.cols << ")" << endl;
		//cout << "after: " << ((float*)im.data) << ", " << ((float*)imResized.data) << endl;
		/*
		string windowName2 = "im purity test result";
		cv::namedWindow(windowName2, CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName2, imResized);
		cv::waitKey(0);
		cv::destroyAllWindows();
		*/
		imScales.push_back(imScale);
		processedIms.push_back(imResized);
	}

	// create a blob to hold the input images
	imListToBlob(processedIms);

	return processedIms;
}


template <typename Dtype>
float RoIInputLayer<Dtype>::prepImForBlob(cv::Mat& im, cv::Mat& imResized,
		const vector<float>& pixelMeans, const uint32_t targetSize, const uint32_t maxSize) {
	// Mean subtract and scale an image for use in a blob
	// cv::Mat, BGR이 cols만큼 반복, 다시 해당 row가 rows만큼 반복
	const uint32_t channels = im.channels();
	// XXX: 3채널 컬러 이미지로 강제
	assert(channels == 3);
	assert(channels == pixelMeans.size());



	/*
#if TEST_MODE
	im.convertTo(im, CV_32F);
#else
	Mat tempIm;
	im.convertTo(im, CV_32FC3, 1.0f/255.0f);
	im.copyTo(tempIm);
	float* tempImPtr = (float*)tempIm.data;
#endif
	float* imPtr = (float*)im.data;


	uint32_t rowUnit, colUnit;
	for (uint32_t i = 0; i < im.rows; i++) {
		rowUnit = i * im.cols * channels;
		for (uint32_t j = 0; j < im.cols; j++) {
			colUnit = j * channels;

#if TEST_MODE
			for (uint32_t k = 0; k < channels; k++) {
				// cv::Mat의 경우 RGB가 아닌 BGR로 데이터가 뒤집어져 있음.
				//imPtr[rowUnit + colUnit + k] -= pixelMeans[channels-k-1];
				// pixel means도 BGR로 뒤집어져 있음.
				imPtr[rowUnit + colUnit + k] -= pixelMeans[k];
			}
#else
			// imPtr: target, reordered as rgb
			// tempImPtr: source, ordered as bgr
			// pixelMeans: ordered as rgb
			imPtr[rowUnit + colUnit + 0] = tempImPtr[rowUnit + colUnit + 2] - pixelMeans[0];
			imPtr[rowUnit + colUnit + 1] = tempImPtr[rowUnit + colUnit + 1] - pixelMeans[1];
			imPtr[rowUnit + colUnit + 2] = tempImPtr[rowUnit + colUnit + 0] - pixelMeans[2];
#endif
		}
	}
	*/



	const vector<uint32_t> imShape = {(uint32_t)im.cols, (uint32_t)im.rows,
			channels};
	uint32_t imSizeMin = np_min(imShape, 0, 2);
	uint32_t imSizeMax = np_max(imShape, 0, 2);
	float imScale = float(targetSize) / float(imSizeMin);
	// Prevent the biggest axis from being more than MAX_SIZE
	if (np_round(imScale * imSizeMax) > maxSize)
		imScale = float(maxSize) / float(imSizeMax);

	cv::resize(im, imResized, cv::Size(), imScale, imScale, CV_INTER_LINEAR);
#if ROIINPUTLAYER_LOG
	cout << "resized to [" << im.rows << ", " << im.cols << ", " << imScale << "]" << endl;
#endif

	return imScale;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::imListToBlob(vector<cv::Mat>& ims) {
	// Convert a list of images into a network input.
	// Assumes images are already prepared (means subtracted, BGR order, ...)

	const uint32_t numImages = ims.size();
	assert(numImages == 1);

	vector<uint32_t> maxShape;
	vector<vector<uint32_t>> imShapes;
	for (uint32_t i = 0; i < numImages; i++)
		imShapes.push_back({(uint32_t)ims[i].rows, (uint32_t)ims[i].cols,
		(uint32_t)ims[i].channels()});
	np_array_max(imShapes, maxShape);

	const vector<uint32_t> inputShape = {numImages, maxShape[0], maxShape[1], 3};
	this->_inputData[0]->reshape(inputShape);
	this->_inputData[0]->set_host_data((Dtype*)ims[0].data);

#if ROIINPUTLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[0]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif

	// Move channels (axis 3) to axis 1
	// Axis order will become: (batch elem, channel, height, width)
	const vector<uint32_t> channelSwap = {0, 3, 1, 2};
	this->_inputData[0]->transpose(channelSwap);
	this->_inputShape[0] = this->_inputData[0]->getShape();

#if ROIINPUTLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[0]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif
}

template<typename Dtype>
int RoIInputLayer<Dtype>::getNumTrainData() {
    return this->_dataSet->getNumTrainData();
}

template<typename Dtype>
int RoIInputLayer<Dtype>::getNumTestData() {
    return this->_dataSet->getNumTestData();
}

template<typename Dtype>
void RoIInputLayer<Dtype>::shuffleTrainDataSet() {
    return this->_dataSet->shuffleTrainDataSet();
}






/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* RoIInputLayer<Dtype>::initLayer() {
    RoIInputLayer* layer = new RoIInputLayer<Dtype>(SLPROP_BASE(name));
    return (void*)layer;
}

template<typename Dtype>
void RoIInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    RoIInputLayer<Dtype>* layer = (RoIInputLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void RoIInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    SASSERT0(index == 0);

    RoIInputLayer<Dtype>* layer = (RoIInputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == 0);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool RoIInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    RoIInputLayer<Dtype>* layer = (RoIInputLayer<Dtype>*)instancePtr;
    //layer->reshape();
    return true;
}

template<typename Dtype>
void RoIInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    cout << "RoIInputLayer.. forward(). miniBatchIndex : " << miniBatchIdx << endl;
}

template<typename Dtype>
void RoIInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    cout << "RoIInputLayer.. backward()" << endl;
}

template<typename Dtype>
void RoIInputLayer<Dtype>::learnTensor(void* instancePtr) {
    cout << "RoIInputLayer.. learn()" << endl;
}

template void* RoIInputLayer<float>::initLayer();
template void RoIInputLayer<float>::destroyLayer(void* instancePtr);
template void RoIInputLayer<float>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index);
template bool RoIInputLayer<float>::allocLayerTensors(void* instancePtr);
template void RoIInputLayer<float>::forwardTensor(void* instancePtr, int miniBatchIdx);
template void RoIInputLayer<float>::backwardTensor(void* instancePtr);
template void RoIInputLayer<float>::learnTensor(void* instancePtr);

template class RoIInputLayer<float>;
