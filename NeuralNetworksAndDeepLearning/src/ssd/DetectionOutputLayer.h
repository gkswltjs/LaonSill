/*
 * DetectionOutputLayer.h
 *
 *  Created on: May 15, 2017
 *      Author: jkim
 */

#ifndef DETECTIONOUTPUTLAYER_H_
#define DETECTIONOUTPUTLAYER_H_

#include <boost/property_tree/ptree.hpp>

#include "common.h"
#include "Layer.h"
#include "ssd_common.h"

/*
 * @brief Generate the detection output based on location and confidence
 *        predictions by doing non maximum suppression.
 */
template <typename Dtype>
class DetectionOutputLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		int _numClasses;
		bool _shareLocation;
		int _backgroundLabelId;

		// NonMaximumSuppressionParameter
		float _nmsThreshold;
		int _topK;
		float _eta;
		// SaveOutputParameter
		std::string _outputDirectory;
		std::string _outputNamePrefix;
		std::string _outputFormat;
		std::string _labelMapFile;
		std::string _nameSizeFile;
		int _numTestImage;
		//
		std::string _codeType;

		bool _varianceEncodedInTarget;
		int _keepTopK;
		float _confidenceThreshold;
		bool _visualize;
		float _visualizeThresh;
		std::string _saveFile;

		Builder() {
			this->type = Layer<Dtype>::DetectionOutput;
			this->_numClasses = -1;
			this->_shareLocation = true;
			this->_backgroundLabelId = 0;

			this->_nmsThreshold = -FLT_MAX;
			this->_topK = -1;
			this->_eta = 1.f;

			this->_numTestImage = -1;

			this->_codeType = "CORNER";
			this->_varianceEncodedInTarget = false;
			this->_keepTopK = -1;
			this->_confidenceThreshold = -FLT_MAX;
			this->_visualize = false;
			this->_visualizeThresh = 0.6f;
		}
		virtual Builder* name(const std::string name) {
			Layer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			Layer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			Layer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			Layer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			Layer<Dtype>::Builder::propDown(propDown);
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
		virtual Builder* backgroundLabelId(const int backgroundLabelId) {
			this->_backgroundLabelId = backgroundLabelId;
			return this;
		}
		virtual Builder* nmsThreshold(const float nmsThreshold) {
			this->_nmsThreshold = nmsThreshold;
			return this;
		}
		virtual Builder* topK(const int topK) {
			this->_topK = topK;
			return this;
		}
		virtual Builder* eta(const float eta) {
			this->_eta = eta;
			return this;
		}
		virtual Builder* outputDirectory(const std::string& outputDirectory) {
			this->_outputDirectory = outputDirectory;
			return this;
		}
		virtual Builder* outputNamePrefix(const std::string& outputNamePrefix) {
			this->_outputNamePrefix = outputNamePrefix;
			return this;
		}
		virtual Builder* outputFormat(const std::string& outputFormat) {
			this->_outputFormat = outputFormat;
			return this;
		}
		virtual Builder* labelMapFile(const std::string& labelMapFile) {
			this->_labelMapFile = labelMapFile;
			return this;
		}
		virtual Builder* nameSizeFile(const std::string& nameSizeFile) {
			this->_nameSizeFile = nameSizeFile;
			return this;
		}
		virtual Builder* numTestImage(const int numTestImage) {
			this->_numTestImage = numTestImage;
			return this;
		}
		virtual Builder* codeType(const std::string& codeType) {
			this->_codeType = codeType;
			return this;
		}
		virtual Builder* varianceEncodedInTarget(const bool varianceEncodedInTarget) {
			this->_varianceEncodedInTarget = varianceEncodedInTarget;
			return this;
		}
		virtual Builder* keepTopK(const int keepTopK) {
			this->_keepTopK = keepTopK;
			return this;
		}
		virtual Builder* confidenceThreshold(const float confidenceThreshold) {
			this->_confidenceThreshold = confidenceThreshold;
			return this;
		}
		virtual Builder* visualize(const bool visualize) {
			this->_visualize = visualize;
			return this;
		}
		virtual Builder* visualizeThresh(const float visualizeThresh) {
			this->_visualizeThresh = visualizeThresh;
			return this;
		}
		virtual Builder* saveFile(const std::string& saveFile) {
			this->_saveFile = saveFile;
			return this;
		}
		Layer<Dtype>* build() {
			return new DetectionOutputLayer(this);
		}
	};

	DetectionOutputLayer(Builder* builder);
	virtual ~DetectionOutputLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void initialize(Builder* builder);

private:
	int numClasses;
	bool shareLocation;
	int backgroundLabelId;

	// NonMaximumSuppressionParameter
	float nmsThreshold;
	int topK;
	float eta;

	std::string outputDirectory;
	std::string outputNamePrefix;
	std::string outputFormat;
	std::string labelMapFile;
	int numTestImage;

	std::string codeType;

	bool varianceEncodedInTarget;
	int keepTopK;
	float confidenceThreshold;
	bool visualize;
	float visualizeThresh;
	std::string saveFile;



	int numLocClasses;
	bool needSave;
	int numPriors;
	int nameCount;

	Data<Dtype> bboxPreds;						// mbox_loc과 동일 shape
	Data<Dtype> bboxPermute;					// !shareLocation인 경우 사용
	Data<Dtype> confPermute;					// mbox_conf_flatten과 동일 shape



	LabelMap<Dtype> labelMap;
	std::vector<std::string> names;				// test 이미지 이름 목록
	std::vector<std::pair<int, int>> sizes;		// test 이미지 height, width 목록

	std::map<int, std::string> labelToName;
	std::map<int, std::string> labelToDisplayName;

	boost::property_tree::ptree detections;


	Data<Dtype> temp;

};

#endif /* DETECTIONOUTPUTLAYER_H_ */
