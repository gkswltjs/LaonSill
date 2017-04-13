/*
 * FrcnnTestOutputLayer.cpp
 *
 *  Created on: Dec 16, 2016
 *      Author: jkim
 */

#include <vector>
#include <array>


#include "FrcnnTestOutputLayer.h"
#include "BboxTransformUtil.h"

#define FRCNNTESTOUTPUTLAYER_LOG 0

using namespace std;


template <typename Dtype>
FrcnnTestOutputLayer<Dtype>::FrcnnTestOutputLayer(Builder* builder)
: Layer<Dtype>(builder) {
	this->maxPerImage = builder->_maxPerImage;
	this->thresh = builder->_thresh;
	this->vis = builder->_vis;
	this->savePath = builder->_savePath;

	initialize();
}

template <typename Dtype>
FrcnnTestOutputLayer<Dtype>::~FrcnnTestOutputLayer() {

}


template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;
	}
}





template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::feedforward() {
	reshape();

	vector<vector<Dtype>> scores;
	vector<vector<Dtype>> predBoxes;

	imDetect(scores, predBoxes);
	testNet(scores, predBoxes);
}


template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::imDetect(vector<vector<Dtype>>& scores, vector<vector<Dtype>>& predBoxes) {

#if FRCNNTESTOUTPUTLAYER_LOG
	this->_printOn();
	this->_inputData[0]->print_data({}, false, -1);		// rois
	this->_inputData[1]->print_data({}, false, -1);		// im_info
	this->_inputData[2]->print_data({}, false, -1);		// cls_prob
	this->_inputData[3]->print_data({}, false, -1);		// bbox_pred
	this->_printOff();
#endif
	// im_info (1, 1, 1, 3-[height, width, scale])
	const Dtype imHeight = this->_inputData[1]->host_data()[0];
	const Dtype imWidth = this->_inputData[1]->host_data()[1];
	const Dtype imScale = this->_inputData[1]->host_data()[2];


	// rois (1, 1, #rois, 5-[batch index, x1, y1, x2, y2])
	const uint32_t numRois = this->_inputData[0]->getShape(2);
	vector<vector<Dtype>> boxes(numRois);
	const Dtype* rois = this->_inputData[0]->host_data();

	for (uint32_t i = 0; i < numRois; i++) {
		boxes[i].resize(4);
		// unscale back to raw image space
		boxes[i][0] = rois[5 * i + 1] / imScale;
		boxes[i][1] = rois[5 * i + 2] / imScale;
		boxes[i][2] = rois[5 * i + 3] / imScale;
		boxes[i][3] = rois[5 * i + 4] / imScale;

		/*
		boxes[i][0] = rois[5 * i + 1];
		boxes[i][1] = rois[5 * i + 2];
		boxes[i][2] = rois[5 * i + 3];
		boxes[i][3] = rois[5 * i + 4];
		*/
	}

#if FRCNNTESTOUTPUTLAYER_LOG
	print2dArray("boxes", boxes);
	this->_printOn();
	this->_inputData[3]->print_data({}, false);
	this->_printOff();
#endif
	fill2dVecWithData(this->_inputData[2], scores);

	/*
	this->_printOn();
	this->_inputData[2]->print_data({}, false);
	print2dArray("scores", scores);
	this->_printOff();
	*/

	// bbox_pred (#rois, 4 * num classes)
	BboxTransformUtil::bboxTransformInv(boxes, this->_inputData[3], predBoxes);
	BboxTransformUtil::clipBoxes(predBoxes,
			{round(imHeight/imScale), round(imWidth/imScale)});
	//BboxTransformUtil::clipBoxes(predBoxes,
	//		{round(imHeight), round(imWidth)});

#if FRCNNTESTOUTPUTLAYER_LOG
	print2dArray("boxes", boxes);
	//print2dArray("scores", scores);
	print2dArray("predBoxes", predBoxes);
#endif
}

template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::testNet(vector<vector<Dtype>>& scores,
		vector<vector<Dtype>>& boxes) {


	/*
	cout << "rois shape: " << endl;
	print2dArray("rois", proposals);

	const string windowName = "rois";
	uint32_t numBoxes = proposals.size();

	Dtype scale = this->_inputData[2]->host_data()[2];
	int boxOffset = 1;
	cout << "scale: " << scale << endl;
	const int onceSize = 5;

	for (int j = 0; j < (numBoxes / onceSize); j++) {
		cv::Mat im = cv::imread(Util::imagePath, CV_LOAD_IMAGE_COLOR);
		cv::resize(im, im, cv::Size(), scale, scale, CV_INTER_LINEAR);

		for (uint32_t i = j*onceSize; i < (j+1)*onceSize; i++) {
			cv::rectangle(im, cv::Point(proposals[i][boxOffset+0], proposals[i][boxOffset+1]),
				cv::Point(proposals[i][boxOffset+2], proposals[i][boxOffset+3]),
				cv::Scalar(0, 0, 255), 2);
		}

		cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName, im);

		if (pause) {
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	}
	*/





	const Dtype confThresh = Dtype(0.7);
	const Dtype nmsThresh = Dtype(0.3);

	vector<vector<float>> result;

	vector<uint32_t> keep;
	vector<Dtype> clsScores;
	vector<vector<Dtype>> clsBoxes;
	vector<uint32_t> inds;

	for (int clsInd = 1; clsInd < this->classes.size(); clsInd++) {
		const string& cls = this->classes[clsInd];

		fillClsScores(scores, clsInd, clsScores);
		fillClsBoxes(boxes, clsInd, clsBoxes);

		cout << cls << "\t\tboxes before nms: " << scores.size();
		nms(clsBoxes, clsScores, nmsThresh, keep);
		cout << " , after nms: " << keep.size() << endl;

		clsBoxes = vec_keep_by_index(clsBoxes, keep);
		clsScores = vec_keep_by_index(clsScores, keep);

		// score 중 confThresh 이상인 것에 대해
		np_where_s(clsScores, GE, confThresh, inds);

		if (inds.size() == 0)
			continue;


		cout << "num of " << cls << ": " << inds.size() << endl;

		int offset = result.size();

		for (int i = 0; i < inds.size(); i++) {
			vector<float> temp(6);
			temp[0] = float(clsInd);
			temp[1] = clsBoxes[inds[i]][0];
			temp[2] = clsBoxes[inds[i]][1];
			temp[3] = clsBoxes[inds[i]][2];
			temp[4] = clsBoxes[inds[i]][3];
			temp[5] = clsScores[inds[i]];
			cout << "\tscore:" << temp[5] << endl;
			result.push_back(temp);

			//printf("%f, %f, %f, %f", temp[1], temp[2], temp[3], temp[4]);
		}
	}

	if (Util::imagePath.size() == 0)
		Util::imagePath = "/home/jkim/Dev/git/py-faster-rcnn-v/data/demo/000010.jpg";



	//const Dtype imScale = this->_inputData[1]->host_data()[2];
	cv::Mat im = cv::imread(Util::imagePath, CV_LOAD_IMAGE_COLOR);
	//cv::resize(im, im, cv::Size(), imScale, imScale, CV_INTER_LINEAR);
	uint32_t numBoxes = result.size();

	for (uint32_t i = 0; i < numBoxes; i++) {
		int clsInd = round(result[i][0]);

		cv::rectangle(im, cv::Point(result[i][1], result[i][2]),
			cv::Point(result[i][3], result[i][4]),
			boxColors[clsInd-1], 2);

		cv::putText(im, this->classes[clsInd] , cv::Point(result[i][1],
				result[i][2]+15.0f), 2, 0.5f, boxColors[clsInd-1]);
	}

	if (this->savePath == "") {
		const string windowName = "result";
		cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName, im);

		if (true) {
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	} else {
		cv::imwrite(this->savePath + "/" + Util::imagePath.substr(Util::imagePath.length()-10), im);
	}

	/*
	displayBoxesOnImage("TEST_RESULT", Util::imagePath, 1, restoredBoxes, boxLabels, {},
			boxColors);

	cout << "end object detection result ... " << endl;
	*/
}

template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::fillClsScores(vector<vector<Dtype>>& scores, int clsInd,
		vector<Dtype>& clsScores) {

	const int size = scores.size();
	clsScores.resize(size);
	for (int i = 0; i < size; i++) {
		clsScores[i] = scores[i][clsInd];
	}
}

template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::fillClsBoxes(vector<vector<Dtype>>& boxes, int clsInd,
		vector<vector<Dtype>>& clsBoxes) {
	const int size = boxes.size();
	clsBoxes.resize(size);

	for (int i = 0; i < size; i++) {
		clsBoxes[i].resize(4);

		int base = clsInd * 4;
		clsBoxes[i][0] = boxes[i][base + 0];
		clsBoxes[i][1] = boxes[i][base + 1];
		clsBoxes[i][2] = boxes[i][base + 2];
		clsBoxes[i][3] = boxes[i][base + 3];
	}
}






template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::initialize() {

	this->classes = {"__background__", "aeroplane", "bicycle", "bird", "boat", "bottle",
			"bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
			"person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};


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




template class FrcnnTestOutputLayer<float>;






























