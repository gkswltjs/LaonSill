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
FrcnnTestOutputLayer<Dtype>::FrcnnTestOutputLayer()
: Layer<Dtype>() {

}

template <typename Dtype>
FrcnnTestOutputLayer<Dtype>::FrcnnTestOutputLayer(Builder* builder)
: Layer<Dtype>(builder) {
	this->maxPerImage = builder->_maxPerImage;
	this->thresh = builder->_thresh;
	this->vis = builder->_vis;

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

		// "rois"
		if (i == 0) {}
		// "im_info"
		else if (i == 1) {}
		// "cls_prob"
		else if (i == 2) {}
		// "bbox_pred"
		else if (i == 3) {}
#if FRCNNTESTOUTPUTLAYER_LOG
		cout << this->_inputs[i] << ": (" <<
				this->_inputData[i]->getShape()[0] << ", " <<
				this->_inputData[i]->getShape()[1] << ", " <<
				this->_inputData[i]->getShape()[2] << ", " <<
				this->_inputData[i]->getShape()[3] << ")" << endl;
#endif
	}
}





/*
template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::feedforward() {
	reshape();

#if FRCNNTESTOUTPUTLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[0]->print_data({}, false);
	this->_inputData[1]->print_data({}, false);
	this->_inputData[2]->print_data({}, false);
	this->_inputData[3]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif
	// rois (1, 1, #rois, 5-[batch index, x1, y1, x2, y2])
	const uint32_t numRois = this->_inputData[0]->getShape(2);
	vector<vector<Dtype>> boxes(numRois);
	const Dtype* rois = this->_inputData[0]->host_data();

	// im_info (1, 1, 1, 3-[height, width, scale])
	const Dtype* imInfo = this->_inputData[1]->host_data();
	float imScale = imInfo[2];
	//float imHeight = roundf(imInfo[0] / imScale);
	//float imWidth = roundf(imInfo[1] / imScale);

	for (uint32_t i = 0; i < numRois; i++) {
		boxes[i].resize(4);
		// unscale back to raw image space
		//boxes[i][0] = rois[5 * i + 1] / imScale;
		//boxes[i][1] = rois[5 * i + 2] / imScale;
		//boxes[i][2] = rois[5 * i + 3] / imScale;
		//boxes[i][3] = rois[5 * i + 4] / imScale;

		boxes[i][0] = rois[5 * i + 1];
		boxes[i][1] = rois[5 * i + 2];
		boxes[i][2] = rois[5 * i + 3];
		boxes[i][3] = rois[5 * i + 4];
	}

#if FRCNNTESTOUTPUTLAYER_LOG
	print2dArray("boxes", boxes);
	Data<Dtype>::printConfig = true;
	this->_inputData[3]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif

	// bbox_pred (#rois, 4 * num classes)
	vector<vector<Dtype>> predBoxes;
	BboxTransformUtil::bboxTransformInv(boxes, this->_inputData[3], predBoxes);
	BboxTransformUtil::clipBoxes(predBoxes, {imInfo[0], imInfo[1]});

#if FRCNNTESTOUTPUTLAYER_LOG
	print2dArray("predBoxes", predBoxes);

	Data<Dtype>::printConfig = true;
	this->_inputData[2]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif

	const uint32_t numClasses = this->_inputData[2]->getShape(3);
	// cls_prob (#rois, num classes)
	//const Dtype* scores = this->_inputData[2]->host_data();
	vector<vector<Dtype>> scores;
	fill2dVecWithData(this->_inputData[2], scores);



	vector<vector<float>> all_boxes;
	vector<float> all_scores;
	vector<int> all_classes;


	// 각 rois에 대해 background가 아니며 최고 score가 기준치 이상인
	// rois를 최종 대상으로 선택한다.
	int maxScoreIndex;
	float maxScore;
	for (uint32_t i = 0; i < numRois; i++) {
		maxScoreIndex = -1;
		maxScore = -1.0f;
		for (uint32_t j = 0; j < numClasses; j++) {
			if (scores[i][j] > maxScore) {
				maxScoreIndex = j;
				maxScore = scores[i][j];
			}
		}

		if (maxScoreIndex > 0) {
			cout << "for " << i << "th rois, maxScore->" << maxScore <<
					", maxScoreIndex: " << maxScoreIndex << endl;
		}

		if (maxScoreIndex > 0 && maxScore >= this->thresh) {
			vector<float> box(4);
			box[0] = predBoxes[i][4*maxScoreIndex+0];
			box[1] = predBoxes[i][4*maxScoreIndex+1];
			box[2] = predBoxes[i][4*maxScoreIndex+2];
			box[3] = predBoxes[i][4*maxScoreIndex+3];

			all_boxes.push_back(box);
			all_scores.push_back(maxScore);
			all_classes.push_back(maxScoreIndex);
		}
	}






	vector<uint32_t> keep;
	nms(all_boxes, all_scores, TEST_NMS, keep);

	all_boxes = vec_keep_by_index(all_boxes, keep);
	all_scores = vec_keep_by_index(all_scores, keep);
	all_classes = vec_keep_by_index(all_classes, keep);


	cout << "object detection result: " << endl;

	for (uint32_t i = 0; i < keep.size(); i++) {
		cout << "for class " << all_classes[i] << endl;

		cout << "\t" << i << ": (" << all_boxes[i][0] << "," <<
			all_boxes[i][1] << "," <<
			all_boxes[i][2] << "," <<
			all_boxes[i][3] << ") -> (" <<
			all_boxes[i][0] / imScale << "," <<
			all_boxes[i][1] / imScale << "," <<
			all_boxes[i][2] / imScale << "," <<
			all_boxes[i][3] / imScale << ")" << endl;
	}
	cout << "end object detection result ... " << endl;

	vector<vector<float>> restoredBoxes(keep.size());
	for (uint32_t i = 0; i < keep.size(); i++) {
		restoredBoxes[i].resize(4);
		restoredBoxes[i][0] = all_boxes[i][0] / imScale;
		restoredBoxes[i][1] = all_boxes[i][1] / imScale;
		restoredBoxes[i][2] = all_boxes[i][2] / imScale;
		restoredBoxes[i][3] = all_boxes[i][3] / imScale;
	}
	displayBoxesOnImage(Util::imagePath, 1, restoredBoxes);

}
*/


template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::feedforward() {
	reshape();

#if FRCNNTESTOUTPUTLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[0]->print_data({}, false);
	this->_inputData[1]->print_data({}, false);
	this->_inputData[2]->print_data({}, false);
	this->_inputData[3]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif
	// rois (1, 1, #rois, 5-[batch index, x1, y1, x2, y2])
	const uint32_t numRois = this->_inputData[0]->getShape(2);
	vector<vector<Dtype>> boxes(numRois);
	const Dtype* rois = this->_inputData[0]->host_data();

	// im_info (1, 1, 1, 3-[height, width, scale])
	const Dtype* imInfo = this->_inputData[1]->host_data();
	float imScale = imInfo[2];
	//float imHeight = roundf(imInfo[0] / imScale);
	//float imWidth = roundf(imInfo[1] / imScale);

	for (uint32_t i = 0; i < numRois; i++) {
		boxes[i].resize(4);
		// unscale back to raw image space
		//boxes[i][0] = rois[5 * i + 1] / imScale;
		//boxes[i][1] = rois[5 * i + 2] / imScale;
		//boxes[i][2] = rois[5 * i + 3] / imScale;
		//boxes[i][3] = rois[5 * i + 4] / imScale;

		boxes[i][0] = rois[5 * i + 1];
		boxes[i][1] = rois[5 * i + 2];
		boxes[i][2] = rois[5 * i + 3];
		boxes[i][3] = rois[5 * i + 4];
	}

#if FRCNNTESTOUTPUTLAYER_LOG
	print2dArray("boxes", boxes);
	Data<Dtype>::printConfig = true;
	this->_inputData[3]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif

	// bbox_pred (#rois, 4 * num classes)
	vector<vector<Dtype>> predBoxes;
	BboxTransformUtil::bboxTransformInv(boxes, this->_inputData[3], predBoxes);
	BboxTransformUtil::clipBoxes(predBoxes, {imInfo[0], imInfo[1]});

#if FRCNNTESTOUTPUTLAYER_LOG
	print2dArray("predBoxes", predBoxes);

	Data<Dtype>::printConfig = true;
	this->_inputData[2]->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif

	const uint32_t numClasses = this->_inputData[2]->getShape(3);
	// cls_prob (#rois, num classes)
	//const Dtype* scores = this->_inputData[2]->host_data();
	vector<vector<Dtype>> scores;
	fill2dVecWithData(this->_inputData[2], scores);

	vector<vector<array<float, 5>>> all_boxes(numClasses);
	vector<float> all_scores;

	vector<Dtype> score;
	vector<vector<Dtype>> predBox;

	// XXX:
	// skip j = 0, because it's the background class
	for (uint32_t i = 1; i < numClasses; i++) {
		// 현재 class에서 score가 thresh를 넘는 roi index 찾기
		vector<uint32_t> inds;
		for (uint32_t j = 0; j < numRois; j++) {
			if (scores[j][i] > this->thresh)
				inds.push_back(j);
		}
		if (inds.size() < 1)
			continue;

#if FRCNNTESTOUTPUTLAYER_LOG
		cout << "for class " << i << endl;
		printArray("inds", inds);
#endif

		vector<uint32_t> keep;
		predBox.resize(inds.size());
		score.resize(inds.size());

		uint32_t ind;
		for (uint32_t j = 0; j < inds.size(); j++) {
			predBox[j].resize(4);

			ind = inds[j];
			predBox[j][0] = predBoxes[ind][4*i+0];
			predBox[j][1] = predBoxes[ind][4*i+1];
			predBox[j][2] = predBoxes[ind][4*i+2];
			predBox[j][3] = predBoxes[ind][4*i+3];

			score[j] = scores[ind][i];
		}
		nms(predBox, score, TEST_NMS, keep);

#if FRCNNTESTOUTPUTLAYER_LOG
		//printArray("keep", keep);
		for (uint32_t j = 0; j < keep.size(); j++) {
			cout << j << " (" << keep[j] << "): " << inds[keep[j]] << endl;
		}
#endif

		all_boxes[i].resize(keep.size());
		for (uint32_t j = 0; j < keep.size(); j++) {
			all_boxes[i][j][0] = predBox[keep[j]][0];
			all_boxes[i][j][1] = predBox[keep[j]][1];
			all_boxes[i][j][2] = predBox[keep[j]][2];
			all_boxes[i][j][3] = predBox[keep[j]][3];
			all_boxes[i][j][4] = score[keep[j]];

			all_scores.push_back(score[keep[j]]);
		}

#if FRCNNTESTOUTPUTLAYER_LOG

		cout << "all boxes for class " << i << endl;
		for (uint32_t j = 0; j < all_boxes[i].size(); j++) {
			cout << j << "\t: " <<
					all_boxes[i][j][0] << "," <<
					all_boxes[i][j][1] << "," <<
					all_boxes[i][j][2] << "," <<
					all_boxes[i][j][3] << "," <<
					all_boxes[i][j][4] << endl;
		}
		cout << "--------------------" << endl;

#endif
	}

	// Limit to maxPerImage detections *over all classes*
	if (this->maxPerImage > 0) {
		cout << "maxPerImage: " << this->maxPerImage << endl;

		if (all_scores.size() > this->maxPerImage) {
			sort(all_scores.begin(), all_scores.end());

#if FRCNNTESTOUTPUTLAYER_LOG
			printArray("all_scores", all_scores);
#endif

			// XXX: 테스트를 위해 imageThresh를 강제로 0.00f로 고정
			float imageThresh = all_scores[all_scores.size()-this->maxPerImage];
			//float imageThresh = 0.00f;

			for (uint32_t i = 1; i < numClasses; i++) {
				typename vector<array<float, 5>>::iterator it;
				for(it = all_boxes[i].begin(); it != all_boxes[i].end();) {
					if ((*it)[4] < imageThresh) {
						all_boxes[i].erase(it);
					} else {
						it++;
					}
				}
			}
		}
	}

	cout << "object detection result: " << endl;
	for (uint32_t i = 1; i < numClasses; i++) {
		if (all_boxes[i].size() > 0) {
			cout << "for class " << i << endl;
			for (uint32_t j = 0; j < all_boxes[i].size(); j++) {
				cout << "\t" << j << ": (" << all_boxes[i][j][0] << "," <<
						all_boxes[i][j][1] << "," <<
						all_boxes[i][j][2] << "," <<
						all_boxes[i][j][3] << ") -> (" <<

						all_boxes[i][j][0] / imScale << "," <<
						all_boxes[i][j][1] / imScale << "," <<
						all_boxes[i][j][2] / imScale << "," <<
						all_boxes[i][j][3] / imScale << ")" << endl;
			}

			vector<vector<float>> restoredBoxes(all_boxes[i].size());
			for (uint32_t j = 0; j < all_boxes[i].size(); j++) {
				restoredBoxes[j].resize(all_boxes[i].size());
				restoredBoxes[j][0] = all_boxes[i][j][0] / imScale;
				restoredBoxes[j][1] = all_boxes[i][j][1] / imScale;
				restoredBoxes[j][2] = all_boxes[i][j][2] / imScale;
				restoredBoxes[j][3] = all_boxes[i][j][3] / imScale;
			}
			displayBoxesOnImage(Util::imagePath, 1, restoredBoxes);
		}
	}
	cout << "end object detection result ... " << endl;
}


template <typename Dtype>
void FrcnnTestOutputLayer<Dtype>::initialize() {

}




template class FrcnnTestOutputLayer<float>;






























