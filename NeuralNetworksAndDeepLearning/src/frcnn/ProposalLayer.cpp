/*
 * ProposalLayer.cpp
 *
 *  Created on: Nov 29, 2016
 *      Author: jkim
 */

#include <vector>


#include "ProposalLayer.h"
#include "GenerateAnchorsUtil.h"
#include "BboxTransformUtil.h"
#include "frcnn_common.h"
#include "NetworkConfig.h"


#define PROPOSALLAYER_LOG 0


using namespace std;

template <typename Dtype>
ProposalLayer<Dtype>::ProposalLayer()
	: HiddenLayer<Dtype>() {
	initialize();
}

template <typename Dtype>
ProposalLayer<Dtype>::ProposalLayer(Builder* builder)
	: HiddenLayer<Dtype>(builder) {
	this->featStride = builder->_featStride;
	this->scales = builder->_scales;

	initialize();
}

template <typename Dtype>
ProposalLayer<Dtype>::~ProposalLayer() {

}

template <typename Dtype>
void ProposalLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		// rois blob: holds R regions of interest, each is a 5-tuple
		// (n, x1, y1, x2, y2) specifying an image batch index n and a
		// rectangle (x1, y1, x2, y2)
		this->_outputData[0]->reshape({1, 1, 1, 5});

		// scores data: holds scores for R regions of interest
		if (this->_outputData.size() > 1) {
			this->_outputData[1]->reshape({1, 1, 1, 1});
		}
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
void ProposalLayer<Dtype>::feedforward() {
	reshape();

	// Algorithm:
	//
	// for each (H, W) location i
	//		generate A anchor boxes centered on cell i
	//		apply predicted bbox deltas at cell i to each of the A 
    //		ancscoreData->getShape()hors
	// clip predicted boxes to image
	// remove predicted boxes with either height or width < threshold
	// sort all (proposal, score) pairs by score from hightest to lowest
	// take top pre_nms_topN proposals before NMS
	// apply NMS with threshold 0.7 to remaining proposals
	// take after_nms_topN proposals after NMS
	// return the top proposals (-> RoIs top, scores top)

	assert(this->_inputData[0]->getShape(0) == 1 &&
			"Only single item batches are supported");


	uint32_t preNmsTopN;
	uint32_t postNmsTopN;
	float nmsThresh;
	uint32_t minSize;

	if (this->networkConfig->_phase == NetworkPhase::TrainPhase) {
		preNmsTopN 	= TRAIN_RPN_PRE_NMS_TOP_N;
		postNmsTopN	= TRAIN_RPN_POST_NMS_TOP_N;
		nmsThresh 	= TRAIN_RPN_NMS_THRESH;
		minSize 	= TRAIN_RPN_MIN_SIZE;
	} else if (this->networkConfig->_phase == NetworkPhase::TestPhase) {
		preNmsTopN 	= TEST_RPN_PRE_NMS_TOP_N;
		postNmsTopN = TEST_RPN_POST_NMS_TOP_N;
		nmsThresh 	= TEST_RPN_NMS_THRESH;
		minSize 	= TEST_RPN_MIN_SIZE;
	}


	// the first set of numAnchors channels are bg probs
	// the second set are the fg probs, which we want
	Data<Dtype>* scoresData = this->_inputData[0]->range({0, (int)this->numAnchors, 0, 0},
			{-1, -1, -1, -1});


#if PROPOSALLAYER_LOG
	Data<Dtype>::printConfig = true;
	this->_inputData[1]->print_data({}, false);
	scoresData->print_data({}, false);
	Data<Dtype>::printConfig = false;

#endif

	Data<Dtype>* bboxDeltas = new Data<Dtype>("bboxDeltas");
	bboxDeltas->reshapeLike(this->_inputData[1]);
	bboxDeltas->set_host_data(this->_inputData[1]);
	Data<Dtype>* imInfo = this->_inputData[2]->range({0, 0, 0, 0}, {-1, -1, 1, -1});

#if PROPOSALLAYER_LOG
	cout << "im_size: (" << imInfo->host_data()[0] << ", " <<
			imInfo->host_data()[1] << ")" << endl;
	cout << "scale: " << imInfo->host_data()[2] << endl;
	Data<Dtype>::printConfig = true;
	bboxDeltas->print_data({}, false);
	imInfo->print_data({}, false);
	Data<Dtype>::printConfig = false;
#endif

	// 1. Generate propsals from bbox deltas and shifted anchors
	const uint32_t height = scoresData->getShape(2);
	const uint32_t width = scoresData->getShape(3);

#if PROPOSALLAYER_LOG
	cout << "score map size: " << scoresData->getShape(0) << ", " <<
        scoresData->getShape(1) << ", " << scoresData->getShape(2) << ", " <<
        scoresData->getShape(3) << endl;
#endif

	// Enumerate all shifts
	const uint32_t numShifts = height * width;
	vector<vector<uint32_t>> shifts(numShifts);

	for (uint32_t i = 0; i < numShifts; i++)
		shifts[i].resize(4);

	for (uint32_t i = 0; i < height; i++) {
		for (uint32_t j = 0; j < width; j++) {
			vector<uint32_t>& shift = shifts[i*width+j];
			shift[0] = j*this->featStride;
			shift[2] = j*this->featStride;
			shift[1] = i*this->featStride;
			shift[3] = i*this->featStride;
		}
	}
#if PROPOSALLAYER_LOG
	print2dArray("shifts", shifts);
#endif

	// Enumerate all shifted anchors:
	//
	// add A anchors (1, A, 4) to
	// cell K shifts (K, 1, 4) to get
	// shift anchors (K, A, 4)
	// reshape to (K*A, 4) shifted anchors
	const uint32_t A = this->numAnchors;
	const uint32_t K = shifts.size();
	const uint32_t totalAnchors = K * A;

	vector<vector<float>> anchors(totalAnchors);
	uint32_t anchorIndex;
	for (uint32_t i = 0; i < K; i++) {
		for (uint32_t j = 0; j < A; j++) {
			anchorIndex = i * A + j;
			vector<float>& anchor = anchors[anchorIndex];
			anchor.resize(4);
			anchor[0] = this->anchors[j][0] + shifts[i][0];
			anchor[1] = this->anchors[j][1] + shifts[i][1];
			anchor[2] = this->anchors[j][2] + shifts[i][2];
			anchor[3] = this->anchors[j][3] + shifts[i][3];
		}
	}
#if PROPOSALLAYER_LOG
	print2dArray("anchors", anchors);
#endif

	// Transpose and reshape predicted bbox transformations to get them
	// into the same order as the anchors:
	//
	// bbox deltas will be (1, 4 * A, H, W) format
	// transpose to (1, H, W, 4 * A)
	// reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
	// in slowest to fastest order

	bboxDeltas->transpose({0, 2, 3, 1});
	bboxDeltas->reshapeInfer({1, 1, -1, 4});

	// Same stroy for the scores:
	//
	// scores are (1, A, H, W) format
	// transpose to (1, H, W, A)
	// reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
	scoresData->transpose({0, 2, 3, 1});
	//scoresData->reshapeInfer({1, 1, -1, 1});
	// XXX: 정렬에 용이할 것 같아서 바꿔 봄.
	scoresData->reshapeInfer({1, 1, 1, -1});
	vector<float> scores;
	fill1dVecWithData(scoresData, scores);
	delete scoresData;

	// Convert anchors into proposals via bbox transformations
	vector<vector<Dtype>> proposals;
	BboxTransformUtil::bboxTransformInv(anchors, bboxDeltas, proposals);
	delete bboxDeltas;

#if PROPOSALLAYER_LOG
	printArray("scores", scores);
	print2dArray("proposals", proposals);
#endif

	// 2. clip predicted boxes to image
	BboxTransformUtil::clipBoxes(proposals,
			{imInfo->host_data()[0], imInfo->host_data()[1]});
#if PROPOSALLAYER_LOG
	cout << imInfo->host_data()[0] << "x" << imInfo->host_data()[1] << endl;
	print2dArray("proposals", proposals);
#endif

	// 3. remove predicted boxes with either height or width < threshold
	// (NOTE: convert minSize to input image scale stored in imInfo[2])
	vector<uint32_t> keep;
	_filterBoxes(proposals, minSize * imInfo->host_data()[2], keep);
	proposals = vec_keep_by_index(proposals, keep);
	scores = vec_keep_by_index(scores, keep);
#if PROPOSALLAYER_LOG
	printArray("keep", keep);
	print2dArray("proposals", proposals);
	printArray("scores", scores);
#endif

	// 4. sort all (proposal, score) pairs by score from highest to lowest
	// 5. take preNmsTopN (e.g. 6000)
	vector<uint32_t> order(scores.size());
#if TEST_MODE
	for (uint32_t i = 0; i < order.size(); i++)
		order[i] = i;
#else
	iota(order.begin(), order.end(), 0);
	vec_argsort(scores, order);
#endif

#if PROPOSALLAYER_LOG
	for (uint32_t i = 0; i < order.size(); i++) {
		cout << order[i] << "\t: " << scores[order[i]] << endl;
	}
#endif

	if (preNmsTopN > 0 && preNmsTopN < order.size())
		order.erase(order.begin() + preNmsTopN, order.end());
	proposals = vec_keep_by_index(proposals, order);
	scores = vec_keep_by_index(scores, order);

#if PROPOSALLAYER_LOG
	printArray("order", order);
	print2dArray("proposals", proposals);
	printArray("scores", scores);
#endif

	// 6. apply nms (e.g. threshold = 0.7)
	// 7. take postNmsTopN (e.g. 300)
	// 8. return the top proposals (->RoIs top)
#if TEST_MODE
	keep.resize(postNmsTopN);
	iota(keep.begin(), keep.end(), 0);
#else

	nms(proposals, scores, nmsThresh, keep);
	/*
	const uint32_t numDets = proposals.size();
	float* dets = new float[numDets*5];
	for (uint32_t i = 0; i < numDets; i++) {
		dets[i*5+0] = proposals[i][0];
		dets[i*5+1] = proposals[i][1];
		dets[i*5+2] = proposals[i][2];
		dets[i*5+3] = proposals[i][3];
		dets[i*5+4] = scores[i];
	}
	nms(dets, numDets, nmsThresh, keep);
	delete [] dets;
	*/

#endif
	if (postNmsTopN > 0 && postNmsTopN < keep.size())
		keep.erase(keep.begin() + postNmsTopN, keep.end());
	proposals = vec_keep_by_index(proposals, keep);

	//printArray("scores", scores);
	scores = vec_keep_by_index(scores, keep);
	//printArray("scores", scores);

	// Output rois data
	// Our RPN implementation only supports a single input image, so all
	// batch inds are 0
	vec_2d_pad(1, proposals);
	this->_outputData[0]->reshape({1, 1, (uint32_t)proposals.size(), 5});
	this->_outputData[0]->fill_host_with_2d_vec(proposals, {0, 1, 2, 3});

#if PROPOSALLAYER_LOG
	cout << "# of proposals: " << proposals.size() << endl;
	print2dArray("proposals", proposals);
	Data<Dtype>::printConfig = true;
	this->_outputData[0]->print_data({}, false);
	Data<Dtype>::printConfig = false;
	printArray("scores", scores);
#endif

	// XXX:
	// [Optional] output scores data
	if (this->_outputData.size() > 1) {
		assert(this->_outputData.size() == 1);
	}
	delete imInfo;
}

template <typename Dtype>
void ProposalLayer<Dtype>::backpropagation() {

}

template <typename Dtype>
void ProposalLayer<Dtype>::initialize() {
	GenerateAnchorsUtil::generateAnchors(this->anchors, this->scales);
	this->numAnchors = this->anchors.size();

#if PROPOSALLAYER_LOG
	cout << "featStride: " << this->featStride << endl;
	print2dArray("anchors", this->anchors);
#endif
}

template <typename Dtype>
void ProposalLayer<Dtype>::_filterBoxes(std::vector<std::vector<float>>& boxes,
		const float minSize, std::vector<uint32_t>& keep) {
	// Remove all boxes with any side smaller than minSize

	keep.clear();
	float ws, hs;
	const uint32_t numBoxes = boxes.size();
	for (uint32_t i = 0; i < numBoxes; i++) {
		std::vector<float>& box = boxes[i];
		ws = box[2] - box[0] + 1;
		hs = box[3] - box[1] + 1;

		if (ws >= minSize && hs >= minSize)
			keep.push_back(i);
	}
}

template class ProposalLayer<float>;
