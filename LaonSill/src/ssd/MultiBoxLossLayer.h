/*
 * MultiBoxLossLayer.h
 *
 *  Created on: Apr 27, 2017
 *      Author: jkim
 */

#ifndef MULTIBOXLOSSLAYER_H_
#define MULTIBOXLOSSLAYER_H_

#include "common.h"
#include "LossLayer.h"

template <typename Dtype>
class MultiBoxLossLayer : public LossLayer<Dtype> {
public:
	/*
	class Builder : public LossLayer<Dtype>::Builder {
	public:
		std::string _locLossType;
		std::string _confLossType;
		Dtype _locWeight;
		int _numClasses;
		bool _shareLocation;
		std::string _matchType;
		Dtype _overlapThreshold;
		bool _usePriorForMatching;
		int _backgroundLabelId;
		bool _useDifficultGt;
		Dtype _negPosRatio;
		Dtype _negOverlap;
		std::string _codeType;
		bool _ignoreCrossBoundaryBbox;
		std::string _miningType;
		bool _encodeVarianceInTarget;
		bool _bpInside;
		bool _usePriorForNMS;
		int _sampleSize;

		// NonMaximumSuppressionParameter
		Dtype _nmsThresh;
		int _topK;
		Dtype _eta;

		bool _mapObjectToAgnostic;

		Builder() {
			this->type = Layer<Dtype>::MultiBoxLoss;
			this->_locLossType = "SMOOTH_L1";
			this->_confLossType = "SOFTMAX";
			this->_locWeight = Dtype(1);
			this->_numClasses = -1;
			this->_shareLocation = true;
			this->_matchType = "PER_PREDICTION";
			this->_overlapThreshold = Dtype(0.5);
			this->_usePriorForMatching = true;
			this->_backgroundLabelId = 0;
			this->_useDifficultGt = true;
			this->_negPosRatio = Dtype(3);
			this->_negOverlap = Dtype(0.5);
			this->_codeType = "CORNER";
			this->_ignoreCrossBoundaryBbox = false;
			this->_miningType = "MAX_NEGATIVE";
			this->_encodeVarianceInTarget = false;
			this->_bpInside = false;
			this->_usePriorForNMS = false;
			this->_sampleSize = 64;

			this->_nmsThresh = 0.0f;
			this->_topK = -1;
			this->_eta = Dtype(1);


			this->_mapObjectToAgnostic = false;
		}
		virtual Builder* name(const std::string name) {
			LossLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			LossLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			LossLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			LossLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			LossLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		virtual Builder* normalization(const typename LossLayer<Dtype>::NormalizationMode normalization) {
			LossLayer<Dtype>::Builder::normalization(normalization);
			return this;
		}
		virtual Builder* locLossType(const std::string& locLossType) {
			this->_locLossType = locLossType;
			return this;
		}
		virtual Builder* confLossType(const std::string& confLossType) {
			this->_confLossType = confLossType;
			return this;
		}
		virtual Builder* locWeight(const Dtype locWeight) {
			this->_locWeight = locWeight;
			return this;
		}
		virtual Builder* numClasses(const int numClasses) {
			this->_numClasses = numClasses;
			return this;
		}
		virtual Builder* shareLocation(const bool shareLocation) {
			this->_shareLocation = shareLocation;
			return this;
		}
		virtual Builder* matchType(const std::string& matchType) {
			this->_matchType = matchType;
			return this;
		}
		virtual Builder* overlapThreshold(const Dtype overlapThreshold) {
			this->_overlapThreshold = overlapThreshold;
			return this;
		}
		virtual Builder* usePriorForMatching(const bool usePriorForMatching) {
			this->_usePriorForMatching = usePriorForMatching;
			return this;
		}
		virtual Builder* backgroundLabelId(const int backgroundLabelId) {
			this->_backgroundLabelId = backgroundLabelId;
			return this;
		}
		virtual Builder* useDifficultGt(const bool useDifficultGt) {
			this->_useDifficultGt = useDifficultGt;
			return this;
		}
		virtual Builder* negPosRatio(const Dtype negPosRatio) {
			this->_negPosRatio = negPosRatio;
			return this;
		}
		virtual Builder* negOverlap(const Dtype negOverlap) {
			this->_negOverlap = negOverlap;
			return this;
		}
		virtual Builder* codeType(const std::string& codeType) {
			this->_codeType = codeType;
			return this;
		}
		virtual Builder* ignoreCrossBoundaryBbox(const bool ignoreCrossBoundaryBbox) {
			this->_ignoreCrossBoundaryBbox = ignoreCrossBoundaryBbox;
			return this;
		}
		virtual Builder* miningType(const std::string& miningType) {
			this->_miningType = miningType;
			return this;
		}
		virtual Builder* encodeVarianceInTarget(const bool encodeVarianceInTarget) {
			this->_encodeVarianceInTarget = encodeVarianceInTarget;
			return this;
		}
		virtual Builder* bpInside(const bool bpInside) {
			this->_bpInside = bpInside;
			return this;
		}
		virtual Builder* usePriorForNMS(const bool usePriorForNMS) {
			this->_usePriorForNMS = usePriorForNMS;
			return this;
		}
		virtual Builder* sampleSize(const int sampleSize) {
			this->_sampleSize = sampleSize;
			return this;
		}
		virtual Builder* nmsThresh(const Dtype nmsThresh) {
			this->_nmsThresh = nmsThresh;
			return this;
		}
		virtual Builder* topK(const int topK) {
			this->_topK = topK;
			return this;
		}
		virtual Builder* eta(const Dtype eta) {
			this->_eta = eta;
			return this;
		}
		virtual Builder* mapObjectToAgnostic(const bool mapObjectToAgnostic) {
			this->_mapObjectToAgnostic = mapObjectToAgnostic;
			return this;
		}
		Layer<Dtype>* build() {
			return new MultiBoxLossLayer(this);
		}

	};
	*/

	MultiBoxLossLayer();
	virtual ~MultiBoxLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

private:
	void setLayerData(Layer<Dtype>* layer, const std::string& type,
			std::vector<Data<Dtype>*>& dataVec);
	Layer<Dtype>* buildLocLossLayer(const LocLossType locLossType);
	Layer<Dtype>* buildConfLossLayer(const ConfLossType confLossType);

private:
	/*
	std::string locLossType;
	std::string confLossType;
	Dtype locWeight;
	int numClasses;
	bool shareLocation;
	std::string matchType;
	Dtype overlapThreshold;
	bool usePriorForMatching;
	int backgroundLabelId;
	bool useDifficultGt;
	Dtype negPosRatio;
	Dtype negOverlap;
	std::string codeType;
	bool ignoreCrossBoundaryBbox;
	std::string miningType;
	bool doNegMining;
	bool encodeVarianceInTarget;
	bool bpInside;
	bool usePriorForNMS;
	Dtype nmsThresh;
	int topK;
	Dtype eta;
	int sampleSize;
	bool mapObjectToAgnostic;
	*/


	int locClasses;
	int numGt;
	int num;
	int numPriors;

	int numMatches;
	int numConf;
	std::vector<std::map<int, std::vector<int>>> allMatchIndices;
	std::vector<std::vector<int>> allNegIndices;

	// How to normalize the loss


	Layer<Dtype>* locLossLayer;
	//std::vector<Data<Dtype>*> locInputVec;
	//std::vector<Data<Dtype>*> locOutputVec;
	Data<Dtype> locPred;
	Data<Dtype> locGt;
	Data<Dtype> locLoss;

	Layer<Dtype>* confLossLayer;
	//std::vector<Data<Dtype>*> confInputVec;
	//std::vector<Data<Dtype>*> confOutputVec;
	Data<Dtype> confPred;
	Data<Dtype> confGt;
	Data<Dtype> confLoss;


public:
    /****************************************************************************
     * layer callback functions
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);


};

#endif /* MULTIBOXLOSSLAYER_H_ */
