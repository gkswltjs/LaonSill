/*
 * AnchorTargetLayer.cpp
 *
 *  Created on: Nov 18, 2016
 *      Author: jkim
 */

#include "common.h"
#include "AnchorTargetLayer.h"
#include "frcnn_common.h"
#include "GenerateAnchorsUtil.h"
#include "RoIDBUtil.h"

#define ANCHORTARGETLAYER_LOG 0

using namespace std;


template <typename Dtype>
AnchorTargetLayer<Dtype>::AnchorTargetLayer() {
	// TODO Auto-generated constructor stub
}

template <typename Dtype>
AnchorTargetLayer<Dtype>::AnchorTargetLayer(Builder* builder)
	: HiddenLayer<Dtype>(builder) {

	this->featStride = builder->_featStride;
	this->allowedBorder = builder->_allowedBorder;
	this->scales = builder->_scales;

	initialize();
}

template <typename Dtype>
AnchorTargetLayer<Dtype>::~AnchorTargetLayer() {
	// TODO Auto-generated destructor stub
}



template <typename Dtype>
void AnchorTargetLayer<Dtype>::reshape() {
	// Reshaping happens during the call to forward.
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (!adjusted)
		return;

	for (uint32_t i = 0; i < this->_inputData.size(); i++) {
		this->_inputShape[i] = this->_inputData[i]->getShape();
	}

	// allow boxes to sit over the dege by a small amount
	const uint32_t height = this->_inputData[0]->getShape(2);
	const uint32_t width = this->_inputData[0]->getShape(3);

#if ANCHORTARGETLAYER_LOG
	cout << "AnchorTargetLayer: height-" << height << ", width-" << width << endl;
#endif

	// labels
	this->_outputData[0]->reshape({1, 1, this->numAnchors*height, width});
	// bbox_targets
	this->_outputData[1]->reshape({1, this->numAnchors*4, height, width});
	// bbox_inside_weights
	this->_outputData[2]->reshape({1, this->numAnchors*4, height, width});
	// bbox_outside_weights
	this->_outputData[3]->reshape({1, this->numAnchors*4, height, width});
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::feedforward() {
	reshape();

	// Algorithm:
	//
	// for each (H, W) location i
	// 		generate 9 anchor boxes centered on cell i
	// 		apply predicted bbox deltas at cell i to each of the 9 anchors
	// filter out-of-image anchors
	// measture GT overlap

	// Only single item batches are supported.
	assert(this->_inputData[0]->getShape(0) == 1);

	// map of shape (..., H, W)
	// XXX: 디버깅 때문에 Height에 -1 한 것!!! 반드시 원복해야 함 !!!
	const uint32_t height = this->_inputData[0]->getShape(2);
	//const uint32_t height = this->_inputData[0]->getShape(2)-1;
	const uint32_t width = this->_inputData[0]->getShape(3);
	// GT boxes (x1, y1, x2, y2, label)
	vector<vector<float>> gtBoxes;
	fill2dVecWithData(this->_inputData[1], gtBoxes);





#if ANCHORTARGETLAYER_LOG
	print2dArray("gtBoxes", gtBoxes);
#endif

	// im_info
	vector<float> imInfo;
	fill1dVecWithData(this->_inputData[2], imInfo);
#if ANCHORTARGETLAYER_LOG
	printArray("imInfo", imInfo);
#endif

	//displayBoxesOnImage(Util::imagePath, imInfo[2], gtBoxes);



	// DEBUG INFO
#if ANCHORTARGETLAYER_LOG
	cout << "im_size: (" << imInfo[0] << ", " << imInfo[1] << ")" << endl;
	cout << "scale: " << imInfo[2] << endl;
	cout << "height, width: (" << height << ", " << width << ")" << endl;
#endif

	// 1. Generate proposals from bbox deltas and shifted anchors
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
#if ANCHORTARGETLAYER_LOG
	print2dArray("shifts", shifts);
#endif

	// add A anchors (1, A, 4) to
	// cell K shifts (K, 1, 4) to get
	// shift anchors (K, A, 4)
	// reshape to (K*A, 4) shifted anchors
	const uint32_t A = this->numAnchors;
	const uint32_t K = shifts.size();
	const uint32_t totalAnchors = K * A;

	vector<vector<float>> allAnchors(totalAnchors);
	uint32_t anchorIndex;
	for (uint32_t i = 0; i < K; i++) {
		for (uint32_t j = 0; j < A; j++) {
			anchorIndex = i * A + j;
			vector<float>& anchor = allAnchors[anchorIndex];
			anchor.resize(4);
			anchor[0] = this->anchors[j][0] + shifts[i][0];
			anchor[1] = this->anchors[j][1] + shifts[i][1];
			anchor[2] = this->anchors[j][2] + shifts[i][2];
			anchor[3] = this->anchors[j][3] + shifts[i][3];
		}
	}
#if ANCHORTARGETLAYER_LOG
	print2dArray("allAnchors", allAnchors);
#endif

	// only keep anchors inside the image
	vector<uint32_t> indsInside;
	vector<vector<float>> anchors;

	const uint32_t orgHeight = imInfo[0];
	const uint32_t orgWidth = imInfo[1];
	for (uint32_t i = 0; i < totalAnchors; i++) {
		vector<float>& tempAnchor = allAnchors[i];
		if ((tempAnchor[0] >= -this->allowedBorder) &&
				(tempAnchor[1] >= -this->allowedBorder) &&
				(tempAnchor[2] < orgWidth+this->allowedBorder) &&
				(tempAnchor[3] < orgHeight+this->allowedBorder)) {
			anchors.push_back(tempAnchor);
			indsInside.push_back(i);
		}
	}
#if ANCHORTARGETLAYER_LOG
	print2dArray("anchors", anchors);
	printArray("indsInside", indsInside);
	cout << "total_anchors: " << totalAnchors << endl;
	cout << "inds_inside: " << anchors.size() << endl;
#endif

	// label: 1 is positive, 0 is negative, -1 is don't care
	vector<int> labels;
	labels.assign(indsInside.size(), -1);

	// overlaps between the anchors and the gt boxes
	// overlaps (ex, gt)
	// 각 anchor와 gtBox간의 IoU값
	vector<vector<float>> overlaps;
	RoIDBUtil::bboxOverlaps(anchors, 0, gtBoxes, 0, overlaps);
#if ANCHORTARGETLAYER_LOG
	print2dArray("overlaps", overlaps);
#endif

	// 각 anchor별 최대 IoU인 gtBox index
	vector<uint32_t> argmaxOverlaps;
	np_argmax(overlaps, 1, argmaxOverlaps);
#if ANCHORTARGETLAYER_LOG
	printArray("argmaxOverlaps", argmaxOverlaps);
#endif

	// 각 anchor별 최대 IoU값
	vector<float> maxOverlaps;
	np_array_value_by_index_array(overlaps, 1, argmaxOverlaps, maxOverlaps);
#if ANCHORTARGETLAYER_LOG
	printArray("maxOverlaps", maxOverlaps);
#endif

	// 각 gtBox별 최대 IoU인 anchor index
	vector<uint32_t> gtArgmaxOverlaps;
	np_argmax(overlaps, 0, gtArgmaxOverlaps);
#if ANCHORTARGETLAYER_LOG
	printArray("gtArgmaxOverlaps", gtArgmaxOverlaps);
#endif

	// 각 gtBox별 최대 IoU값
	vector<float> gtMaxOverlaps;
	np_array_value_by_index_array(overlaps, 0, gtArgmaxOverlaps, gtMaxOverlaps);
#if ANCHORTARGETLAYER_LOG
	printArray("gtMaxOverlaps", gtMaxOverlaps);
#endif

	np_where_s(overlaps, gtMaxOverlaps, gtArgmaxOverlaps);
#if ANCHORTARGETLAYER_LOG
	printArray("gtArgmaxOverlaps", gtArgmaxOverlaps);
#endif

	if (!TRAIN_RPN_CLOBBER_POSITIVES) {
		// assign bg labels first so that positive labels can clobber them

#if ANCHORTARGETLAYER_LOG
		cout << "RPN_CLOBBER_NEGATIVE" << endl;
		uint32_t clobberCount = 0;
#endif
		for (uint32_t i = 0; i < labels.size(); i++) {
			if (maxOverlaps[i] < TRAIN_RPN_NEGATIVE_OVERLAP) {
				labels[i] = 0;
#if ANCHORTARGETLAYER_LOG
				cout << clobberCount++ << "\t: " << i << endl;
#endif
			}
		}
	}

	// fg label: for each gt, anchor with highest overlap
	for (uint32_t i = 0; i < gtArgmaxOverlaps.size(); i++) {
		labels[gtArgmaxOverlaps[i]] = 1;
	}

	// fg label: above threshold IOU
#if ANCHORTARGETLAYER_LOG
	cout << "FG LABEL: ABOVE THRESHOLD IOU" << endl;
	uint32_t fgCount = 0;
#endif
	for (uint32_t i = 0; i < labels.size(); i++) {
		if (maxOverlaps[i] >= TRAIN_RPN_POSITIVE_OVERLAP) {
			labels[i] = 1;
#if ANCHORTARGETLAYER_LOG
			cout << fgCount++ << "\t: " << i << endl;
#endif
		}
	}

	if (TRAIN_RPN_CLOBBER_POSITIVES) {
		// assign bg labels last so that negative labels can clobber positives
		for (uint32_t i = 0; i < labels.size(); i++) {
			if (maxOverlaps[i] < TRAIN_RPN_NEGATIVE_OVERLAP)
				labels[i] = 0;
		}
	}
#if ANCHORTARGETLAYER_LOG
	printArray("labels", labels);
#endif

	// subsample positive labels if we have too many
	const uint32_t numFg = uint32_t(TRAIN_RPN_FG_FRACTION * TRAIN_RPN_BATCHSIZE);
	vector<uint32_t> fgInds;
	np_where_s(labels, EQ, 1, fgInds);
	uint32_t finalNumFg = fgInds.size();
	if (fgInds.size() > numFg) {
#if TEST_MODE
		for (uint32_t i = numFg; i < fgInds.size(); i++) {
			labels[fgInds[i]] = -1;
		}
		finalNumFg = numFg;
#else
		vector<uint32_t> disableInds;
		npr_choice(fgInds, fgInds.size()-numFg, disableInds);
		for (uint32_t i = 0; i < disableInds.size(); i++)
			labels[disableInds[i]] = -1;
		finalNumFg -= disableInds.size();
#endif
	}

	// subsample negative labels if we have too many
	const uint32_t numBg = TRAIN_RPN_BATCHSIZE - finalNumFg;
	vector<uint32_t> bgInds;
	np_where_s(labels, EQ, 0, bgInds);
	uint32_t finalNumBg = bgInds.size();
	if (bgInds.size() > numBg) {
		vector<uint32_t> disableInds;
#if TEST_MODE
		// XXX: 디버깅 문제로 임시 -> 원복함 -------------------------------------------------
		for (uint32_t i = numBg; i < bgInds.size(); i++) {
			labels[bgInds[i]] = -1;
		}
		finalNumBg = numBg;
#else
		npr_choice(bgInds, bgInds.size()-numBg, disableInds);
		for (uint32_t i = 0; i < disableInds.size(); i++)
			labels[disableInds[i]] = -1;
		finalNumBg -= disableInds.size();
#endif
		// ---------------------------------------------------------
	}

	vector<vector<float>> bboxTargets;
	vector<vector<float>> gtBoxesTemp(argmaxOverlaps.size());
	for (uint32_t i = 0; i < argmaxOverlaps.size(); i++) {
		gtBoxesTemp[i] = gtBoxes[argmaxOverlaps[i]];
	}
	_computeTargets(anchors, gtBoxesTemp, bboxTargets);

#if ANCHORTARGETLAYER_LOG
	print2dArray("anchors", anchors);
	print2dArray("gtBoxes", gtBoxesTemp);
	print2dArray("bboxTargets", bboxTargets);
#endif



	//Data<Dtype>* data = new Data<Dtype>("bboxTargets");
	//data->reshape({1, 1, bboxTargets.size(), bboxTargets[0].size()});
	//data->fill_host_with_2d_vec(bboxTargets);
	//data->save("/home/jkim/Documents/bboxTargets.data");

	vector<vector<float>> bboxInsideWeights(labels.size());
	for (uint32_t i = 0; i < labels.size(); i++) {
		bboxInsideWeights[i].resize(4);
		if (labels[i] == 1)
			bboxInsideWeights[i] = TRAIN_RPN_BBOX_INSIDE_WEIGHTS;
	}
#if ANCHORTARGETLAYER_LOG
	print2dArray("bboxInsideWeights", bboxInsideWeights);
#endif

	vector<vector<float>> bboxOutsideWeights(labels.size());
	// XXX: supports only negative trainRpnPositiveWeight so far...
	assert(TRAIN_RPN_POSITIVE_WEIGHT < 0);
	const float uniformWeight = 1.0f / (finalNumFg + finalNumBg);
	const vector<float> positiveWeights = {uniformWeight, uniformWeight,
			uniformWeight, uniformWeight};
	const vector<float> negativeWeights = {uniformWeight, uniformWeight,
			uniformWeight, uniformWeight};

	for (uint32_t i = 0; i < labels.size(); i++) {
		if (labels[i] == 1)
			bboxOutsideWeights[i] = positiveWeights;
		else if (labels[i] == 0)
			bboxOutsideWeights[i] = negativeWeights;
		else
			bboxOutsideWeights[i].resize(4);
	}
#if ANCHORTARGETLAYER_LOG
	print2dArray("bboxOutsideWeights", bboxOutsideWeights);
#endif

	// map up to original set of anchors
	vector<int> finalLabels;
	vector<vector<float>> finalBboxTargets;
	vector<vector<float>> finalBboxInsideWeights;
	vector<vector<float>> finalBboxOutsideWeights;

	_unmap(labels, totalAnchors, indsInside, -1, finalLabels);
	_unmap(bboxTargets, totalAnchors, indsInside, finalBboxTargets);
	_unmap(bboxInsideWeights, totalAnchors, indsInside, finalBboxInsideWeights);
	_unmap(bboxOutsideWeights, totalAnchors, indsInside, finalBboxOutsideWeights);

#if ANCHORTARGETLAYER_LOG
	printArray("finalLabels", finalLabels);
	print2dArray("finalBboxTargets", finalBboxTargets);
	print2dArray("finalBboxInsideWeights", finalBboxInsideWeights);
	print2dArray("finalBboxOutsideWeights", finalBboxOutsideWeights);
#endif

	//Data<Dtype>::printConfig = true;
	// labels
	this->_outputData[0]->reshape({1, height, width, A});
	//fillDataWith1dVec(finalLabels, {0, 3, 1, 2}, this->_outputData[0]);
	this->_outputData[0]->fill_host_with_1d_vec(finalLabels, {0, 3, 1, 2});
	this->_outputData[0]->reshape({1, 1, A * height, width});
	this->_outputData[0]->print_data("labels", {}, false);

	// bbox_targets
	this->_outputData[1]->reshape({1, height, width, A * 4});
	//fillDataWith2dVec(finalBboxTargets, {0, 3, 1, 2}, this->_outputData[1]);
	this->_outputData[1]->fill_host_with_2d_vec(finalBboxTargets, {0, 3, 1, 2});
	this->_outputData[1]->reshape({1, A * 4, height, width});
	this->_outputData[1]->print_data("bbox_targets", {}, false);

	// bbox_inside_weights
	this->_outputData[2]->reshape({1, height, width, A * 4});
	//fillDataWith2dVec(finalBboxInsideWeights, {0, 3, 1, 2}, this->_outputData[2]);
	this->_outputData[2]->fill_host_with_2d_vec(finalBboxInsideWeights, {0, 3, 1, 2});
	this->_outputData[2]->reshape({1, A * 4, height, width});

	/*
	print2dArray("finalBboxInsideWeights", finalBboxInsideWeights);
	Data<Dtype>::printConfig = true;
	this->_outputData[2]->print_data("bbox_inside_weights", {}, false);
	Data<Dtype>::printConfig = false;
	exit(1);
	*/



	// bbox_outside_weights
	this->_outputData[3]->reshape({1, height, width, A * 4});
	//fillDataWith2dVec(finalBboxOutsideWeights, {0, 3, 1, 2}, this->_outputData[3]);
	this->_outputData[3]->fill_host_with_2d_vec(finalBboxOutsideWeights, {0, 3, 1, 2});
	this->_outputData[3]->reshape({1, A * 4, height, width});
	this->_outputData[3]->print_data("bbox_outside_weights", {}, false);
	//Data<Dtype>::printConfig = false;
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::backpropagation() {
	// This layer does not propagate gradients.
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::initialize() {

	GenerateAnchorsUtil::generateAnchors(this->anchors, this->scales);
	this->numAnchors = this->anchors.size();

	print2dArray("anchors", this->anchors);
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::_computeTargets(
		const vector<vector<float>>& exRois,
		const vector<vector<float>>& gtRois,
		vector<vector<float>>& bboxTargets) {

	assert(exRois.size() == gtRois.size());
	//assert(exRois[0].size() == 4);
	//assert(gtRois[0].size() == 5);

	bboxTargets.resize(exRois.size());
	for (uint32_t i = 0; i < exRois.size(); i++) {
		bboxTargets[i].resize(4);
	}

	BboxTransformUtil::bboxTransform(exRois, 0, gtRois, 0, bboxTargets);
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::_unmap(const vector<int>& data,
		const uint32_t count, const vector<uint32_t>& indsInside,
		const int fill, vector<int>& result) {
	// Unmap a subset of item (data) back to the original set of items (of size count)
	result.resize(count);
	std::fill(result.begin(), result.end(), fill);

	for (uint32_t i = 0; i < indsInside.size(); i++) {
		result[indsInside[i]] = data[i];
	}
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::_unmap(const vector<vector<float>>& data,
		const uint32_t count, const vector<uint32_t>& indsInside,
		vector<vector<float>>& result) {
	// Unmap a subset of item (data) back to the original set of items (of size count)
	result.resize(count);
	const uint32_t dim2 = data[0].size();
	for (uint32_t i = 0; i < count; i++) {
		result[i].resize(dim2);
	}

	for (uint32_t i = 0; i < indsInside.size(); i++) {
		result[indsInside[i]] = data[i];
	}
}



template class AnchorTargetLayer<float>;



















