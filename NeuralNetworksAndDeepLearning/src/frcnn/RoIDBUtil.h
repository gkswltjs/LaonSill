/*
 * RoIDBUtil.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef ROIDBUTIL_H_
#define ROIDBUTIL_H_

#include "frcnn_common.h"
#include "BboxTransformUtil.h"
#include "RoIDB.h"
#include <algorithm>
#include <iostream>
#include <ostream>
#include <cassert>
#include <cstdint>
#include <vector>


class RoIDBUtil {
public:
	static void addBboxRegressionTargets(std::vector<RoIDB>& roidb,
			std::vector<std::vector<float>>& means, std::vector<std::vector<float>>& stds) {
		// Add information needed to train bounding-box regressors.
		assert(roidb.size() > 0);

		const uint32_t numImages = roidb.size();
		// Infer numfer of classes from the number of columns in gt_overlaps
		const uint32_t numClasses = roidb[0].gt_overlaps[0].size();

		for (uint32_t i = 0; i < numImages; i++) {
			RoIDBUtil::computeTargets(roidb[i]);
		}

		//std::vector<std::vector<float>> means;
		np_tile({0.0f, 0.0f, 0.0f, 0.0f}, numClasses, means);
		print2dArray("bbox target means", means);

		//std::vector<std::vector<float>> stds;
		np_tile({0.1f, 0.1f, 0.2f, 0.2f}, numClasses, stds);
		print2dArray("bbox target stdeves", stds);

		// Normalize targets
		std::cout << "Normalizing targets" << std::endl;
		for (uint32_t i = 0; i < numImages; i++) {
			std::vector<std::vector<float>>& targets = roidb[i].bbox_targets;
			std::vector<uint32_t> clsInds;
			for (uint32_t j = 1; j < numClasses; j++) {
				np_where_s(targets, static_cast<float>(j), (uint32_t)0, clsInds);

				for (uint32_t k = 0; k < clsInds.size(); k++) {
					targets[k][1] = (targets[k][1] - means[j][0]) / stds[j][0];
					targets[k][2] = (targets[k][2] - means[j][1]) / stds[j][1];
					targets[k][3] = (targets[k][3] - means[j][2]) / stds[j][2];
					targets[k][4] = (targets[k][4] - means[j][3]) / stds[j][3];
				}
			}
			print2dArray("bbox_targets", targets);
		}
	}

	static void computeTargets(RoIDB& roidb) {
		roidb.print();

		// Compute bounding-box regression targets for an image.
		// Indices of ground-truth ROIs
		std::vector<uint32_t> gt_inds;
		// XXX: 1.0f float compare check
		np_where_s(roidb.max_overlaps, EQ, 1.0f, gt_inds);
		if (gt_inds.size() < 1) {
			// Bail if the image has no ground-truth ROIs
		}
		// Indices of examples for which we try to make predictions
		std::vector<uint32_t> ex_inds;
		np_where_s(roidb.max_overlaps, GE, TRAIN_BBOX_THRESH, ex_inds);

		// Get IoU overlap between each ex ROI and gt ROI
		std::vector<std::vector<float>> ex_gt_overlaps;
		bboxOverlaps(roidb.boxes, gt_inds, ex_inds, ex_gt_overlaps);
		print2dArray("ex_gt_overlaps", ex_gt_overlaps);

		// Find which gt ROI each ex ROI has max overlap with:
		// this will be the ex ROI's gt target
		std::vector<uint32_t> gt_assignment;
		np_argmax(ex_gt_overlaps, 1, gt_assignment);
		std::vector<uint32_t> gt_rois_inds;
		std::vector<std::vector<uint32_t>> gt_rois;
		py_arrayElemsWithArrayInds(gt_inds, gt_assignment, gt_rois_inds);
		py_arrayElemsWithArrayInds(roidb.boxes, gt_rois_inds, gt_rois);
		print2dArray("gt_rois", gt_rois);
		std::vector<std::vector<uint32_t>> ex_rois;
		py_arrayElemsWithArrayInds(roidb.boxes, ex_inds, ex_rois);
		print2dArray("ex_rois", ex_rois);

		const uint32_t numRois = roidb.boxes.size();
		const uint32_t numEx = ex_inds.size();
		std::vector<std::vector<float>>& targets = roidb.bbox_targets;
		targets.resize(numRois);
		for (uint32_t i = 0; i < numRois; i++) {
			targets[i].resize(5);
			// XXX: init to zero ... ?
		}
		print2dArray("targets", targets);

		for (uint32_t i = 0; i < numEx; i++) {
			targets[i][0] = roidb.max_classes[i];
		}
		print2dArray("targets", targets);
		BboxTransformUtil::bboxTransform(ex_rois, gt_rois, targets, 1);
		print2dArray("targets", targets);

		roidb.print();
	}

	static void bboxOverlaps(const std::vector<std::vector<uint32_t>>& rois,
			const std::vector<uint32_t>& gt_inds, const std::vector<uint32_t>& ex_inds,
			std::vector<std::vector<float>>& result) {

		const uint32_t numEx = ex_inds.size();
		const uint32_t numGt = gt_inds.size();

		result.resize(numEx);
		for (uint32_t i = 0; i < numEx; i++) {
			result[i].resize(numGt);
			for (uint32_t j = 0; j < numGt; j++) {
				result[i][j] = iou(rois[ex_inds[i]], rois[gt_inds[j]]);
			}
		}
	}

	static void bboxOverlaps(const std::vector<std::vector<float>>& ex,
			const std::vector<std::vector<float>>& gt,
			std::vector<std::vector<float>>& result) {

		const uint32_t numEx = ex.size();
		const uint32_t numGt = gt.size();

		result.resize(numEx);
		for (uint32_t i = 0; i < numEx; i++) {
			result[i].resize(numGt);
			for (uint32_t j = 0; j < numGt; j++) {
				result[i][j] = iou(ex[i], gt[j]);
			}
		}
	}

	template <typename Dtype>
	static float iou(const std::vector<Dtype>& box1, const std::vector<Dtype>& box2) {
		float iou = 0.0f;
		Dtype left, right, top, bottom;
		left = std::max(box1[0], box2[0]);
		right = std::min(box1[2], box2[2]);
		top = std::max(box1[1], box2[1]);
		bottom = std::min(box1[3], box2[3]);

		if(left < right &&
				top < bottom) {
			float i = float((right-left)*(bottom-top));
			float u = float((box1[2]-box1[0])*(box1[3]-box1[1]) +
					(box2[2]-box2[0])*(box2[3]-box2[1]) - i);
			iou = i/u;
		}
		return iou;
	}
};



#endif /* ROIDBUTIL_H_ */
