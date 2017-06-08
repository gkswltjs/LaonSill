/*
 * MultiBoxLossLayer.cpp
 *
 *  Created on: Apr 27, 2017
 *      Author: jkim
 */

#include "MultiBoxLossLayer.h"
#include "ssd_common.h"
#include "SmoothL1LossLayer.h"
#include "SoftmaxWithLossLayer.h"
#include "SysLog.h"
#include "BBoxUtil.h"
#include "MathFunctions.h"

#define MULTIBOXLOSSLAYER_LOG 0

using namespace std;

template <typename Dtype>
MultiBoxLossLayer<Dtype>::MultiBoxLossLayer(Builder* builder)
: LossLayer<Dtype>(builder),
  locPred("locPred"),
  locGt("locGt"),
  locLoss("locLoss"),
  confPred("confPred"),
  confGt("confGt"),
  confLoss("confLoss") {
	initialize(builder);
}

template <typename Dtype>
MultiBoxLossLayer<Dtype>::~MultiBoxLossLayer() {

}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();
	bool inputShapeChanged = false;
	for (int i = 0; i < this->_inputData.size(); i++) {
		if (Layer<Dtype>::_isInputShapeChanged(0)) {
			inputShapeChanged = true;
			break;
		}
	}
	if (!inputShapeChanged) return;


	// XXX: JUST FOR TEST !!! /////////////////////////////
	//this->_inputData[0]->reshapeInfer({0, 1, -1, 1});
	//this->_inputData[1]->reshapeInfer({0, 1, -1, 1});
	//this->_inputData[2]->reshapeInfer({this->_inputData[2]->getShape(2), 1, -1, 1});
	///////////////////////////////////////////////////////


	this->num = this->_inputData[0]->batches();
	this->numPriors = this->_inputData[2]->channels() / 4;
	this->numGt = this->_inputData[3]->height();

	SASSERT0(this->_inputData[0]->batches() == this->_inputData[1]->batches());
	SASSERT(this->numPriors * this->locClasses * 4 == this->_inputData[0]->channels(),
			"Number of priors must match number of location predictions.");
	SASSERT(this->numPriors * this->numClasses == this->_inputData[1]->channels(),
			"Number of priors must match number of confidence predictions.");

	this->_outputData[0]->reshape({1, 1, 1, 1});
	this->_outputData[0]->mutable_host_grad()[0] = this->lossWeight;
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* locData = this->_inputData[0]->host_data();
	const Dtype* confData = this->_inputData[1]->host_data();
	const Dtype* priorData = this->_inputData[2]->host_data();
	const Dtype* gtData = this->_inputData[3]->host_data();

	// Retrieve all ground truth
	// key: item_id (index in batch), value: gt bbox list belongs to item_id
	map<int, vector<NormalizedBBox>> allGtBBoxes;
	GetGroundTruth(gtData, this->numGt, this->backgroundLabelId, this->useDifficultGt,
			&allGtBBoxes);

#if MULTIBOXLOSSLAYER_LOG
	for (auto it = allGtBBoxes.begin(); it != allGtBBoxes.end(); it++) {
		cout << "for itemId: " << it->first << endl;
		for (int i = 0; i < it->second.size(); i++) {
			cout << "\t" << i << endl;
			it->second[i].print();
		}
	}
#endif

	// Retrieve all prior bboxes. It is same withing a batch since we assume all
	// images in a batch are of same dimension.
	// all prior bboxes
	vector<NormalizedBBox> priorBBoxes;
	// all prior variances
	vector<vector<float>> priorVariances;
	GetPriorBBoxes(priorData, this->numPriors, &priorBBoxes, &priorVariances);

#if MULTIBOXLOSSLAYER_LOG
	cout << "priorBBoxes ... " << endl;
	for (int i = 0; i < 10; i++) {
		priorBBoxes[i].print();
		cout << "-----" << endl;
	}

	cout << "priorVariances ... " << endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 4; j++) {
			cout << priorVariances[i][j] << ",";
			cout << "-----" << endl;
		}
		cout << endl;
	}
#endif

	// Retrieve all predictions.
	// allLocPreds[0]: 첫번째 이미지의 prediction ...
	// allLocPreds[0][-1]: shareLocation==true인 경우 label은 오직 -1뿐,
	// 					   -1 key에 전체 prediction box list를 value로 contain.
	vector<LabelBBox> allLocPreds;
	GetLocPredictions(locData, this->num, this->numPriors, this->locClasses,
			this->shareLocation, &allLocPreds);

#if MULTIBOXLOSSLAYER_LOG
	//this->_printOn();
	//this->_inputData[0]->print_data({}, false);
	//this->_printOff();

	for (int i = 0; i < allLocPreds.size(); i++) {
		LabelBBox& labelBBox = allLocPreds[i];

		for (LabelBBox::iterator it = labelBBox.begin(); it != labelBBox.end(); it++) {
			cout << it->first << endl;
			for (int j = 0; j < 10; j++) {
				it->second[j].print();
				cout << "-------" << endl;
			}
		}
	}
#endif

	// Find matches between source bboxes and ground truth bboxes.
	// for each image in batch, (label : overlaps for each prior bbox)
	vector<map<int, vector<float>>> allMatchOverlaps;
	// allMatchOverlaps: batch내 각 이미지에 대한 (label=-1:prior bbox overlap)맵 리스트
	// allMatchIndices: batch내 각 이미지에 대한 최대 match gt index 맵 리스트 
	FindMatches(allLocPreds, allGtBBoxes, priorBBoxes, priorVariances,
			this->numClasses, this->shareLocation, this->matchType,
			this->overlapThreshold, this->usePriorForMatching,
			this->backgroundLabelId, this->codeType,
			this->encodeVarianceInTarget, this->ignoreCrossBoundaryBbox,
			&allMatchOverlaps, &this->allMatchIndices);

#if MULTIBOXLOSSLAYER_LOG
	priorBBoxes[8333].print();

	cout << "allMatchOverlaps" << endl;
	for (int i = 0; i < allMatchOverlaps.size(); i++) {
		map<int, vector<float>>& matchOverlaps = allMatchOverlaps[i];
		for (map<int, vector<float>>::iterator it = matchOverlaps.begin();
				it != matchOverlaps.end(); it++) {
			cout << it->first << endl;
			for (int j = 0; j < it->second.size(); j++) {
				if (it->second[j] > 1e-6) {
					//cout << j << "\t\t" << it->second[j] << endl;
				}
			}
		}
	}

	cout << "allMatchIndices" << endl;
	for (int i = 0; i < this->allMatchIndices.size(); i++) {
		int matchCount = 0;
		map<int, vector<int>>& matchIndices = this->allMatchIndices[i];
		for (map<int, vector<int>>::iterator it = matchIndices.begin();
				it != matchIndices.end(); it++) {
			cout << it->first << endl;
			for (int j = 0; j < it->second.size(); j++) {
				if (it->second[j] > -1) {
					//cout << j << "\t\t" << it->second[j] << endl;
					matchCount++;
				}
			}
		}
		cout << "match count: " << matchCount << endl;
	}
#endif

	this->numMatches = 0;
	int numNegs = 0;
	// Sample hard negative (and positive) examples based on mining type.
	// allNegInidices: batch내 이미지별 negative sample 리스트.
	MineHardExamples(*this->_inputData[1], allLocPreds, allGtBBoxes, priorBBoxes,
			priorVariances, allMatchOverlaps,
			this->numClasses, this->backgroundLabelId, this->usePriorForNMS,
			this->confLossType, this->miningType, this->locLossType, this->negPosRatio,
			this->negOverlap, this->codeType, this->encodeVarianceInTarget, this->nmsThresh,
			this->topK, this->sampleSize, this->bpInside, this->usePriorForMatching,
			&this->numMatches, &numNegs, &this->allMatchIndices, &this->allNegIndices);

#if MULTIBOXLOSSLAYER_LOG
	// std::vector<std::vector<int>> allNegIndices;
	for (int i = 0; i < this->allNegIndices.size(); i++) {
		cout << i << "-----" << endl;
		for (int j = 0; j < this->allNegIndices[i].size(); j++) {
			cout << j << "\t\t" << this->allNegIndices[i][j] << endl;
		}
	}
	cout << "numNegs: " << numNegs << endl;
#endif

	// 
	if (this->numMatches >= 1) {
		// Form data to pass on to locLossLayer
		vector<uint32_t> locShape(4, 1);
		locShape[3] = this->numMatches * 4;
		this->locPred.reshape(locShape);
		this->locGt.reshape(locShape);
		Dtype* locPredData = this->locPred.mutable_host_data();
		Dtype* locGtData = this->locGt.mutable_host_data();
		EncodeLocPrediction(allLocPreds, allGtBBoxes, this->allMatchIndices, priorBBoxes,
				priorVariances, this->codeType, this->encodeVarianceInTarget, this->bpInside,
				this->usePriorForMatching, locPredData, locGtData);

#if MULTIBOXLOSSLAYER_LOG
		this->_printOn();
		this->locPred.print_data({}, false, -1);
		this->locGt.print_data({}, false, -1);
		this->_printOff();
#endif

		//this->locLossLayer->reshape();
		this->locLossLayer->feedforward();
	} else {
		this->locLoss.mutable_host_data()[0] = 0;
	}

	// Form data to pass on to confLossLayer.
	if (this->doNegMining) {
		this->numConf = this->numMatches + numNegs;
	} else {
		this->numConf = this->num * this->numPriors;
	}

	if (this->numConf >= 1) {
		// Reshape the confidence data.
		vector<uint32_t> confShape(4, 1);
		if (this->confLossType == "SOFTMAX") {
			confShape[3] = this->numConf;
			this->confGt.reshape(confShape);
			confShape[2] = this->numConf;
			confShape[3] = this->numClasses;
			this->confPred.reshape(confShape);
		} else if (this->confLossType == "LOGISTIC") {
			confShape[2] = this->numConf;
			confShape[3] = this->numClasses;
			this->confGt.reshape(confShape);
			this->confPred.reshape(confShape);
		} else {
			SASSERT(false, "Unknown confidence loss type.");
		}

		if (!this->doNegMining) {
			// Consider all scores.
			// Share data and grad with inputData[1]
			SASSERT0(this->confPred.getCount() == this->_inputData[1]->getCount());
			this->confPred.share_data(this->_inputData[1]);
		}
		Dtype* confPredData = this->confPred.mutable_host_data();
		Dtype* confGtData = this->confGt.mutable_host_data();
		soooa_set(this->confGt.getCount(), Dtype(this->backgroundLabelId), confGtData);
		EncodeConfPrediction(confData, this->num, this->numPriors, this->numClasses,
				this->backgroundLabelId, this->mapObjectToAgnostic, this->miningType,
				this->confLossType, this->allMatchIndices, this->allNegIndices, allGtBBoxes,
				confPredData, confGtData);

#if MULTIBOXLOSSLAYER_LOG
		this->_printOn();
		this->confGt.print_data({}, false, -1);
		this->confPred.print_data({}, false, -1);
		this->_printOff();
#endif

		//this->confLossLayer->reshape();
		this->confLossLayer->feedforward();
	} else {
		this->confLoss.mutable_host_data()[0] = 0;
	}

	this->_outputData[0]->mutable_host_data()[0] = 0;
	if (this->_propDown[0]) {
		Dtype normalizer = LossLayer<Dtype>::getNormalizer(this->num, this->numPriors,
				this->numMatches);
		this->_outputData[0]->mutable_host_data()[0] +=
				this->locWeight * this->locLoss.host_data()[0] / normalizer;
	}
	if (this->_propDown[1]) {
		Dtype normalizer = LossLayer<Dtype>::getNormalizer(this->num, this->numPriors,
				this->numMatches);
		this->_outputData[0]->mutable_host_data()[0] +=
				this->confLoss.host_data()[0] / normalizer;
	}
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::backpropagation() {
	SASSERT(!this->_propDown[2], "MultiBoxLossLayer cannot backpropagate to prior inputs.");
	SASSERT(!this->_propDown[3], "MultiBoxLossLayer cannot backpropagate to label inputs.");

	//this->_printOn();
	//this->_inputData[0]->print_data({}, false, -1);

	// Back propagate on location prediction.
	if (this->_propDown[0]) {
		Dtype* locInputGrad = this->_inputData[0]->mutable_host_grad();
		soooa_set(this->_inputData[0]->getCount(), Dtype(0), locInputGrad);
		//this->_inputData[0]->print_grad({}, false, -1);

		if (this->numMatches >= 1) {
			vector<bool> locPropDown;
			// Only back propagate on prediction, not ground truth.
			locPropDown.push_back(true);
			locPropDown.push_back(false);
			this->locLossLayer->_propDown = locPropDown;

			//this->locLoss.print_grad({}, false, -1);
			this->locLossLayer->backpropagation();
			//this->locPred.print_grad({}, false, -1);

			// Scale gradient.
			Dtype normalizer = LossLayer<Dtype>::getNormalizer(this->num, this->numPriors,
					this->numMatches);
			Dtype lossWeight = this->_outputData[0]->host_grad()[0] / normalizer;
			soooa_gpu_scal(this->locPred.getCount(), lossWeight,
					this->locPred.mutable_device_grad());
			//this->locPred.print_grad({}, false, -1);

			// Copy gradient back to inputData[0]
			const Dtype* locPredGrad = this->locPred.host_grad();
			int count = 0;
			for (int i = 0; i < this->num; i++) {
				for (map<int, vector<int>>::iterator it = this->allMatchIndices[i].begin();
						it != this->allMatchIndices[i].end(); it++) {
					const int label = this->shareLocation ? 0 : it->first;
					const vector<int>& matchIndex = it->second;
					for (int j = 0; j < matchIndex.size(); j++) {
						if (matchIndex[j] <= -1)  {
							continue;
						}
						// Copy the grad to the right place.
						int startIdx = this->locClasses * 4 * j + label * 4;
						soooa_copy<Dtype>(4, locPredGrad + count * 4, locInputGrad + startIdx);
						count++;
					}
				}
				locInputGrad += this->_inputData[0]->offset(1);
			}
			//this->_inputData[0]->print_grad({}, false, -1);
		}
	}
	//this->_printOff();

	// Back propagate on confidence prediction.
	if (this->_propDown[1]) {
		Dtype* confInputGrad = this->_inputData[1]->mutable_host_grad();
		soooa_set(this->_inputData[1]->getCount(), Dtype(0), confInputGrad);
		if (this->numConf >= 1) {
			vector<bool> confPropDown;
			// Only back propagate on prediction, not ground truth.
			confPropDown.push_back(true);
			confPropDown.push_back(false);
			this->confLossLayer->_propDown = confPropDown;
			this->confLossLayer->backpropagation();
			// Scale gradient.
			Dtype normalizer = LossLayer<Dtype>::getNormalizer(this->num, this->numPriors,
					this->numMatches);
			Dtype lossWeight = this->_outputData[0]->host_grad()[0] / normalizer;
			soooa_gpu_scal(this->confPred.getCount(), lossWeight,
					this->confPred.mutable_device_grad());
			// Copy gradient back to inputData[1]
			const Dtype* confPredGrad = this->confPred.host_grad();
			if (this->doNegMining) {
				int count = 0;
				for (int i = 0; i < this->num; i++) {
					// Copy matched (positive) bboxes scores' grad.
					const map<int, vector<int>>& matchIndices = this->allMatchIndices[i];
					for (map<int, vector<int>>::const_iterator it = matchIndices.begin();
							it != matchIndices.end(); it++) {
						const vector<int>& matchIndex = it->second;
						SASSERT0(matchIndex.size() == this->numPriors);
						for (int j = 0; j < this->numPriors; j++) {
							if (matchIndex[j] <= -1) {
								continue;
							}
							// Copy the grad to the right place.
							soooa_copy<Dtype>(this->numClasses,
									confPredGrad + count * this->numClasses,
									confInputGrad + j * numClasses);
							count++;
						}
					}
					// Copy negative bboxes scores' grad
					for (int n = 0; n < this->allNegIndices[i].size(); n++) {
						int j = this->allNegIndices[i][n];
						SASSERT0(j < this->numPriors);
						soooa_copy<Dtype>(this->numClasses,
								confPredGrad + count * this->numClasses,
								confInputGrad + j * this->numClasses);
						count++;
					}
					confInputGrad += this->_inputData[1]->offset(1);
				}
			} else {
				// The grad is already computed and stored.
				this->_inputData[1]->share_grad(&this->confPred);
			}
		}
	}

	// After backward, remove match statistics.
	this->allMatchIndices.clear();
	this->allNegIndices.clear();
}

template <typename Dtype>
Dtype MultiBoxLossLayer<Dtype>::cost() {
	return this->_outputData[0]->host_data()[0];
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::initialize(Builder* builder) {
	if (builder->_propDown.size() == 0) {
		this->_propDown.push_back(true);
		this->_propDown.push_back(true);
		this->_propDown.push_back(false);
		this->_propDown.push_back(false);
	}

	// Get other parameters.
	SASSERT(builder->_numClasses >= 0, "Must provide numClasses > 0");
	this->numClasses = builder->_numClasses;
	this->shareLocation = builder->_shareLocation;
	this->locClasses = this->shareLocation ? 1 : this->numClasses;
	this->backgroundLabelId = builder->_backgroundLabelId;
	this->useDifficultGt = builder->_useDifficultGt;
	this->miningType = builder->_miningType;
	this->doNegMining = (this->miningType != "NONE") ? true : false;
	this->encodeVarianceInTarget = builder->_encodeVarianceInTarget;

	if (this->doNegMining) {
		SASSERT(this->shareLocation,
				"Currently only support negative mining if shareLocation is true.");
	}

	vector<uint32_t> lossShape(4, 1);
	// Set up localization loss layer
	this->locWeight = builder->_locWeight;
	this->locLossType = builder->_locLossType;
	this->bpInside = builder->_bpInside;
	this->usePriorForNMS = builder->_usePriorForNMS;
	this->sampleSize = builder->_sampleSize;
	this->nmsThresh = builder->_nmsThresh;
	this->topK = builder->_topK;
	this->eta = builder->_eta;
	this->mapObjectToAgnostic = builder->_mapObjectToAgnostic;
	this->codeType = builder->_codeType;
	this->matchType = builder->_matchType;
	this->usePriorForMatching = builder->_usePriorForMatching;
	this->overlapThreshold = builder->_overlapThreshold;
	this->ignoreCrossBoundaryBbox = builder->_ignoreCrossBoundaryBbox;
	this->negPosRatio = builder->_negPosRatio;
	this->negOverlap = builder->_negOverlap;

	// fake shape
	vector<uint32_t> locShape(4, 1);
	locShape[3] = 4;
	this->locPred.reshape(locShape);
	this->locGt.reshape(locShape);
	this->locInputVec.push_back(&this->locPred);
	this->locInputVec.push_back(&this->locGt);

	this->locLoss.reshape(lossShape);
	this->locOutputVec.push_back(&this->locLoss);

	if (this->locLossType == "SMOOTH_L1") {
		SmoothL1LossLayer<float>::Builder* smoothL1LossLayerBuilder =
				new typename SmoothL1LossLayer<float>::Builder();
		smoothL1LossLayerBuilder
			->id(0)
			->name(this->name + "_smooth_L1_loc")
			->lossWeight(this->locWeight);
		this->locLossLayer = smoothL1LossLayerBuilder->build();
	} else {
		SASSERT(false, "Unknown localization loss type.");
	}

	setLayerData(this->locLossLayer, "input", this->locInputVec);
	setLayerData(this->locLossLayer, "output", this->locOutputVec);

	// Set up confidence loss layer.
	this->confLossType = builder->_confLossType;
	this->confInputVec.push_back(&this->confPred);
	this->confInputVec.push_back(&this->confGt);
	this->confLoss.reshape(lossShape);
	this->confOutputVec.push_back(&this->confLoss);

	if (this->confLossType == "SOFTMAX") {
		SASSERT(this->backgroundLabelId >= 0,
				"backgroundLabelId should be within [0, numClasses) for Softmax.");
		SASSERT(this->backgroundLabelId < this->numClasses,
				"backgroundLabelId should be within [0, numClasses) for Softmax.");

		SoftmaxWithLossLayer<float>::Builder* softmaxWithLossLayerBuilder =
				new typename SoftmaxWithLossLayer<float>::Builder();
		softmaxWithLossLayerBuilder
			->id(0)
			->name(this->name + "_softmax_conf")
			->lossWeight(Dtype(1.))
			->normalization(LossLayer<float>::NormalizationMode::NoNormalization)
			->softmaxAxis(3);
		this->confLossLayer = softmaxWithLossLayerBuilder->build();
	} else {
		SASSERT(false, "Unknown confidence loss type.");
	}
	setLayerData(this->confLossLayer, "input", this->confInputVec);
	setLayerData(this->confLossLayer, "output", this->confOutputVec);
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::setLayerData(Layer<Dtype>* layer, const std::string& type,
		std::vector<Data<Dtype>*>& dataVec) {

	if (type == "input") {
		for (int i = 0; i < dataVec.size(); i++) {
			layer->_inputs.push_back(dataVec[i]->_name);
			layer->_inputData.push_back(dataVec[i]);
		}
	} else if (type == "output") {
		for (int i = 0; i < dataVec.size(); i++) {
			layer->_outputs.push_back(dataVec[i]->_name);
			layer->_outputData.push_back(dataVec[i]);
		}
	} else {
		SASSERT(false, "invalid layer data type.");
	}
}

template class MultiBoxLossLayer<float>;




































