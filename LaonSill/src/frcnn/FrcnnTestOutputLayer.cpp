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
#include "PropMgmt.h"
#include "StdOutLog.h"

#define FRCNNTESTOUTPUTLAYER_LOG 0

using namespace std;


template <typename Dtype>
FrcnnTestOutputLayer<Dtype>::FrcnnTestOutputLayer()
: Layer<Dtype>(),
  labelMap(SLPROP(FrcnnTestOutput, labelMapPath)) {
	this->type = Layer<Dtype>::FrcnnTestOutput;

	SASSERT(SNPROP(status) == NetworkStatus::Test,
			"FrcnnTestOutputLayer can be run only in Test Status");
	/*
	this->classes = {"__background__", "aeroplane", "bicycle", "bird", "boat", "bottle",
			"bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
			"person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
			*/

	this->labelMap.build();

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
FrcnnTestOutputLayer<Dtype>::~FrcnnTestOutputLayer() {}


template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		this->_outputData[0]->reshape({1, 1, 1, 7});
	}

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

	const Dtype confThresh = Dtype(SLPROP(FrcnnTestOutput, confThresh));
	const Dtype nmsThresh = Dtype(SLPROP(FrcnnTestOutput, nmsThresh));

	vector<vector<float>> result;

	vector<uint32_t> keep;
	vector<Dtype> clsScores;
	vector<vector<Dtype>> clsBoxes;
	vector<uint32_t> inds;
	vector<vector<Dtype>> detectionOut;

	for (int clsInd = 1; clsInd < this->labelMap.getCount(); clsInd++) {
		const string& cls = this->labelMap.convertIndToLabel(clsInd);

		fillClsScores(scores, clsInd, clsScores);
		fillClsBoxes(boxes, clsInd, clsBoxes);

		//cout << cls << "\t\tboxes before nms: " << scores.size();
		nms(clsBoxes, clsScores, nmsThresh, keep);
		//cout << " , after nms: " << keep.size() << endl;

		clsBoxes = vec_keep_by_index(clsBoxes, keep);
		clsScores = vec_keep_by_index(clsScores, keep);

		// score 중 confThresh 이상인 것에 대해
		np_where_s(clsScores, GE, confThresh, inds);
		//np_where_s(clsScores, GE, 0.01, inds);

		if (inds.size() == 0)
			continue;


		//cout << "num of " << cls << ": " << inds.size() << endl;

		int offset = result.size();

		for (int i = 0; i < inds.size(); i++) {
			vector<float> temp(7);
			temp[0] = 0.f;
			temp[1] = float(clsInd);
			temp[2] = clsScores[inds[i]];
			temp[3] = clsBoxes[inds[i]][0];
			temp[4] = clsBoxes[inds[i]][1];
			temp[5] = clsBoxes[inds[i]][2];
			temp[6] = clsBoxes[inds[i]][3];

			//cout << "\tscore:" << temp[2] << endl;
			result.push_back(temp);
		}
	}

	//exit(1);
	if (Util::imagePath.size() == 0)
		Util::imagePath = "/home/jkim/Dev/git/py-faster-rcnn-v/data/demo/000010.jpg";

	//const Dtype imScale = this->_inputData[1]->host_data()[2];
	cv::Mat im = cv::imread(Util::imagePath, CV_LOAD_IMAGE_COLOR);
	//cv::resize(im, im, cv::Size(), imScale, imScale, CV_INTER_LINEAR);
	uint32_t numBoxes = result.size();

	//cout << "rows: " << im.rows << ", cols: " << im.cols << endl;
	//cout << result[0][3] << "," << result[0][4] << "," << result[0][5] << "," << result[0][6] << endl;

	for (uint32_t i = 0; i < numBoxes; i++) {
		int clsInd = round(result[i][1]);

		cv::rectangle(im, cv::Point(result[i][3], result[i][4]),
			cv::Point(result[i][5], result[i][6]),
			boxColors[clsInd-1], 2);

		cv::putText(im, this->labelMap.convertIndToLabel(clsInd) , cv::Point(result[i][3],
				result[i][4]+15.0f), 2, 0.5f, boxColors[clsInd-1]);
	}

	if (SLPROP(FrcnnTestOutput, savePath) == "") {
		const string windowName = "result";
		cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName, im);

		if (true) {
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	} else {
		cv::imwrite(SLPROP(FrcnnTestOutput, savePath) + "/" + Util::imagePath.substr(Util::imagePath.length()-10), im);
	}

	if (result.size() > 0) {
		fillDataWith2dVec(result, this->_outputData[0]);
	} else {
		this->_outputData[0]->reshape({1, 1, 1, 7});
		this->_outputData[0]->mutable_host_data()[1] = -1;
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



/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* FrcnnTestOutputLayer<Dtype>::initLayer() {
    FrcnnTestOutputLayer* layer = new FrcnnTestOutputLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::destroyLayer(void* instancePtr) {
    FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
	if (isInput) {
		SASSERT0(index < 4);
	} else {
		SASSERT0(index < 1);
	}

    FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(layer->_outputData.size() == index);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool FrcnnTestOutputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
	FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;
	layer->feedforward();
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::backwardTensor(void* instancePtr) {
	FrcnnTestOutputLayer<Dtype>* layer = (FrcnnTestOutputLayer<Dtype>*)instancePtr;
	layer->backpropagation();
}

template<typename Dtype>
void FrcnnTestOutputLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}


template class FrcnnTestOutputLayer<float>;






























