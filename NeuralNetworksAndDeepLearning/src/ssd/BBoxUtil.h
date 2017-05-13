/*
 * BBoxUtil.h
 *
 *  Created on: Apr 28, 2017
 *      Author: jkim
 */

#ifndef BBOXUTIL_H_
#define BBOXUTIL_H_

#include <map>
#include <vector>

#include "ssd_common.h"
#include "Data.h"

typedef std::map<int, std::vector<NormalizedBBox>> LabelBBox;


template <typename Dtype>
void GetGroundTruth(const Dtype* gtData, const int numGt, const int backgroundLabelId,
		const bool useDifficultGt, std::map<int, std::vector<NormalizedBBox>>* allGtBboxes);

template <typename Dtype>
void GetPriorBBoxes(const Dtype* priorData, const int numPriors,
		std::vector<NormalizedBBox>* priorBBoxes,
		std::vector<std::vector<float>>* priorVariances);

template <typename Dtype>
void GetLocPredictions(const Dtype* locData, const int num, const int numPredsPerClass,
		const int numLocClasses, const bool shareLocation, std::vector<LabelBBox>* locPreds);

void FindMatches(const std::vector<LabelBBox>& allLocPreds,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const int numClasses, const bool shareLocation, const std::string& matchType,
		const float overlapThreshold, const bool usePriorForMatching,
		const int backgroundLabelId, const std::string& codeType,
		const bool encodeVarianceInTarget, const bool ignoreCrossBoundaryBBox,
		std::vector<std::map<int, std::vector<float>>>* allMatchOverlaps,
		std::vector<std::map<int, std::vector<int>>>* allMatchIndices);

void DecodeBBoxes(const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const std::string& codeType, const bool varianceEncodedInTarget,
		const bool clipBBox, const std::vector<NormalizedBBox>& bboxes,
		std::vector<NormalizedBBox>* decodeBBoxes);

void DecodeBBox(const NormalizedBBox& priorBBox, const std::vector<float>& priorVariances,
		const std::string& codeType, const bool varianceEncodedInTarget,
		const bool clipBBox, const NormalizedBBox& bbox, NormalizedBBox* decodeBBox);

void MatchBBox(const std::vector<NormalizedBBox>& gtBBoxes,
		const std::vector<NormalizedBBox>& predBBoxes, const int label,
		const std::string& matchType, const float overlapThreshold,
		const bool ignoreCrossBoundaryBBox, std::vector<int>* matchIndices,
		std::vector<float>* matchOverlaps);

float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);

void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clipBBox);

bool IsCrossBoundaryBBox(const NormalizedBBox& bbox);

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		NormalizedBBox* intersectBBox);

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		const bool normalized = true);

template <typename Dtype>
void EncodeLocPrediction(const std::vector<LabelBBox>& allLocPreds,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const std::string& codeType, const bool encodeVarianceInTarget,
		const bool bpInside, const bool usePriorForMatching,
		Dtype* locPredData, Dtype* locGtData);

void EncodeBBox(const NormalizedBBox& priorBBox, const std::vector<float>& priorVariance,
		const std::string& codeType, const bool encodeVarianceInTarget,
		const NormalizedBBox& bbox, NormalizedBBox* encodeBBox);

template <typename Dtype>
void MineHardExamples(Data<Dtype>& confData,
		const std::vector<LabelBBox>& allLocPreds,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		const std::vector<NormalizedBBox>& priorBBoxes,
		const std::vector<std::vector<float>>& priorVariances,
		const std::vector<std::map<int, std::vector<float>>>& allMatchOverlaps,
		const int numClasses, const int backgroundLabelId, const bool usePriorForNms,
		const std::string& confLossType, const std::string& miningType,
		const std::string& locLossType, const float negPosRatio, const float negOverlap,
		const std::string& codeType, const bool encodeVarianceInTarget, const float nmsThresh,
		const int topK, const int sampleSize, const bool bpInside,
		const bool usePriorForMatching, int* numMatches, int* numNegs,
		std::vector<std::map<int, std::vector<int>>>* allMatchIndices,
		std::vector<std::vector<int>>* allNegIndices);

template <typename Dtype>
void EncodeConfPrediction(const Dtype* confData, const int num, const int numPriors,
		const int numClasses, const int backgroundLabelId, const bool mapObjectToAgnostic,
		const std::string& miningType, const std::string& confLossType,
		const std::vector<std::map<int, std::vector<int>>>& allMatchIndices,
		const std::vector<std::vector<int>>& allNegIndices,
		const std::map<int, std::vector<NormalizedBBox>>& allGtBBoxes,
		Dtype* confPredData, Dtype* confGtData);


#endif /* BBOXUTIL_H_ */
