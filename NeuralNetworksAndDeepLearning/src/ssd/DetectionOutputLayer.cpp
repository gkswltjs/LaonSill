/*
 * DetectionOutputLayer.cpp
 *
 *  Created on: May 15, 2017
 *      Author: jkim
 */

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include "DetectionOutputLayer.h"
#include "BBoxUtil.h"
#include "DataTransformer.h"
#include "MathFunctions.h"
#include "StdOutLog.h"
#include "SysLog.h"

using namespace std;
using namespace boost::property_tree;

template <typename Dtype>
DetectionOutputLayer<Dtype>::DetectionOutputLayer(Builder* builder)
: Layer<Dtype>(builder),
  bboxPreds("bboxPreds"),
  bboxPermute("bboxPermute"),
  confPermute("confPermute"),
  temp("temp") {
	initialize(builder);
}

template <typename Dtype>
DetectionOutputLayer<Dtype>::~DetectionOutputLayer() {

}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();
	bool inputShapeChanged = false;
	for (int i = 0; i < this->_inputData.size(); i++) {
		if (Layer<Dtype>::_isInputShapeChanged(0)) {
			inputShapeChanged = true;
			break;
		}
	}
	if (!inputShapeChanged) return;

	this->bboxPreds.reshapeLike(this->_inputData[0]);
	if (!this->shareLocation) {
		this->bboxPermute.reshapeLike(this->_inputData[0]);
	}
	this->confPermute.reshapeLike(this->_inputData[1]);


	if (this->needSave) {
		SASSERT0(this->nameCount <= this->names.size());
		if (this->nameCount % this->numTestImage == 0) {
			// Clean all outputs.
			if (this->outputFormat == "VOC") {
				boost::filesystem::path outputDirectory(this->outputDirectory);
				for (map<int, string>::iterator it = this->labelToName.begin();
						it != this->labelToName.end(); it++) {
					if (it->first == this->backgroundLabelId) {
						continue;
					}
					std::ofstream outfile;
					boost::filesystem::path file(this->outputNamePrefix + it->second + ".txt");
					boost::filesystem::path _outfile = outputDirectory / file;
					outfile.open(_outfile.string().c_str(), std::ofstream::out);
				}
			}
		}
	}

	SASSERT0(this->_inputData[0]->batches() == this->_inputData[1]->batches());
	if (this->bboxPreds.batches() != this->_inputData[0]->batches() ||
			this->bboxPreds.getCountByAxis(1) != this->_inputData[0]->getCountByAxis(1)) {
		this->bboxPreds.reshapeLike(this->_inputData[0]);
	}
	if (!this->shareLocation && (this->bboxPermute.batches() != this->_inputData[0]->batches()
			|| this->bboxPermute.getCountByAxis(1) != this->_inputData[0]->getCountByAxis(1))) {
		this->bboxPermute.reshapeLike(this->_inputData[0]);
	}
	if (this->confPermute.batches() != this->_inputData[1]->batches() ||
			this->confPermute.getCountByAxis(1) != this->_inputData[1]->getCountByAxis(1)) {
		this->confPermute.reshapeLike(this->_inputData[1]);
	}

	this->numPriors = this->_inputData[2]->channels() / 4;

	//cout << "numPriors: " << this->numPriors << ", numLocClasses: " << this->numLocClasses << endl;
	//this->_inputData[0]->print_shape();
	SASSERT(this->numPriors * this->numLocClasses * 4 == this->_inputData[0]->channels(),
			"Number of priors must match number of location predictions.");

	//cout << "numClasses: " << this->numClasses << endl;
	//this->_inputData[1]->print_shape();
	SASSERT(this->numPriors * this->numClasses == this->_inputData[1]->channels(),
			"Number of priors must match number of confidence predictions.");
	// num() and channels() are 1.
	vector<uint32_t> outputShape(4, 1);
	// Since the number of bboxes to be kept is unknown before nms, we manually
	// set it to (fake) 1.
	outputShape[2] = 1;
	// Each orw is a 7 dimension vector, which stores
	// [image_id, label, confidence, xmin, ymin, xmax, ymax]
	outputShape[3] = 7;
	this->_outputData[0]->reshape(outputShape);
}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* locData = this->_inputData[0]->device_data();
	const Dtype* priorData = this->_inputData[2]->device_data();
	const int num = this->_inputData[0]->batches();

	// Decode predictions.
	Dtype* bboxData = this->bboxPreds.mutable_device_data();
	const int locCount = this->bboxPreds.getCount();
	const bool clipBBox = false;
	DecodeBBoxesGPU<Dtype>(locCount, locData, priorData, this->codeType,
			this->varianceEncodedInTarget, this->numPriors, this->shareLocation,
			this->numLocClasses, this->backgroundLabelId, clipBBox, bboxData);

	// Retrieve all decoded location predictions.
	const Dtype* bboxHostData;
	if (!this->shareLocation) {
		Dtype* bboxPermuteData = this->bboxPermute.mutable_device_data();
		PermuteDataGPU<Dtype>(locCount, bboxData, this->numLocClasses, this->numPriors, 4,
				bboxPermuteData);
		bboxHostData = this->bboxPermute.host_data();
	} else {
		bboxHostData = this->bboxPreds.host_data();
	}

	// Retrieve all confidences.
	Dtype* confPermuteData = this->confPermute.mutable_device_data();
	PermuteDataGPU<Dtype>(this->_inputData[1]->getCount(), this->_inputData[1]->device_data(),
			this->numClasses, this->numPriors, 1, confPermuteData);
	const Dtype* confHostData = this->confPermute.host_data();

	int numKept = 0;
	vector<map<int, vector<int>>> allIndices;
	for (int i = 0; i < num; i++) {
		map<int, vector<int>> indices;
		int numDet = 0;
		const int confIdx = i * this->numClasses * this->numPriors;
		int bboxIdx;
		if (this->shareLocation) {
			bboxIdx = i * this->numPriors * 4;
		} else {
			bboxIdx = confIdx * 4;
		}
		for (int c = 0; c < this->numClasses; c++) {
			if (c == this->backgroundLabelId) {
				// Ignore background class.
				continue;
			}
			const Dtype* curConfData = confHostData + confIdx + c * this->numPriors;
			const Dtype* curBBoxData = bboxHostData + bboxIdx;
			if (!this->shareLocation) {
				curBBoxData += c * this->numPriors * 4;
			}
			ApplyNMSFast(curBBoxData, curConfData, this->numPriors, this->confidenceThreshold,
					this->nmsThreshold, this->eta, this->topK, &(indices[c]));
			numDet += indices[c].size();
		}
		if (this->keepTopK > -1 && numDet > this->keepTopK) {
			vector<pair<float, pair<int, int>>> scoreIndexPairs;
			for (map<int, vector<int>>::iterator it = indices.begin();
					it != indices.end(); it++) {
				int label = it->first;
				const vector<int>& labelIndices = it->second;
				for (int j = 0; j < labelIndices.size(); j++) {
					int idx = labelIndices[j];
					float score = confHostData[confIdx + label * this->numPriors + idx];
					scoreIndexPairs.push_back(
							std::make_pair(score, std::make_pair(label, idx)));
				}
			}

			// Keep top k results per image.
			std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(),
					SortScorePairDescend<pair<int, int>>);
			scoreIndexPairs.resize(this->keepTopK);
			// Store the new indices.
			map<int, vector<int>> newIndices;
			for (int j = 0; j < scoreIndexPairs.size(); j++) {
				int label = scoreIndexPairs[j].second.first;
				int idx = scoreIndexPairs[j].second.second;
				newIndices[label].push_back(idx);
			}
			allIndices.push_back(newIndices);
			numKept += keepTopK;
		} else {
			allIndices.push_back(indices);
			numKept += numDet;
		}
	}

	vector<uint32_t> outputShape(4, 1);
	outputShape[2] = numKept;
	outputShape[3] = 7;
	Dtype* outputData;
	if (numKept == 0) {
		STDOUT_LOG("Couldn't find any detections.");
		outputShape[2] = num;
		this->_outputData[0]->reshape(outputShape);
		outputData = this->_outputData[0]->mutable_host_data();
		soooa_set<Dtype>(this->_outputData[0]->getCount(), -1, outputData);
		// Generate fake results per image.
		for (int i = 0; i < num; i++) {
			outputData[0] = i;
			outputData += 7;
		}
	} else {
		this->_outputData[0]->reshape(outputShape);
		outputData = this->_outputData[0]->mutable_host_data();
	}

	int count = 0;
	boost::filesystem::path outputDirectory(this->outputDirectory);
	for (int i = 0; i < num; i++) {
		const int confIdx = i * this->numClasses * this->numPriors;
		int bboxIdx;
		if (this->shareLocation) {
			bboxIdx = i * this->numPriors * 4;
		} else {
			bboxIdx = confIdx * 4;
		}
		for (map<int, vector<int>>::iterator it = allIndices[i].begin();
				it != allIndices[i].end(); it++) {
			int label = it->first;
			vector<int>& indices = it->second;
			if (this->needSave) {
				SASSERT(this->labelToName.find(label) != this->labelToName.end(),
						"Cannot find label: %d in the label map.", label);
				SASSERT0(this->nameCount < this->names.size());
			}
			const Dtype* curConfData = confHostData + confIdx + label * this->numPriors;
			const Dtype* curBBoxData = bboxHostData + bboxIdx;
			if (!this->shareLocation) {
				curBBoxData += label * this->numPriors * 4;
			}
			for (int j = 0; j < indices.size(); j++) {
				int idx = indices[j];
				outputData[count * 7] = i;
				outputData[count * 7 + 1] = label;
				outputData[count * 7 + 2] = curConfData[idx];
				for (int k = 0; k < 4; k++) {
					outputData[count * 7 + 3 + k] = curBBoxData[idx * 4 + k];
				}
				if (this->needSave) {
					// Generate output bbox.
					NormalizedBBox bbox;
					bbox.xmin = outputData[count * 7 + 3];
					bbox.ymin = outputData[count * 7 + 4];
					bbox.xmax = outputData[count * 7 + 5];
					bbox.ymax = outputData[count * 7 + 6];
					NormalizedBBox outBBox;
					OutputBBox(bbox, this->sizes[this->nameCount], false, &outBBox);
					float score = outputData[count * 7 + 2];
					float xmin = outBBox.xmin;
					float ymin = outBBox.ymin;
					float xmax = outBBox.xmax;
					float ymax = outBBox.ymax;

					ptree ptXmin;
					ptree ptYmin;
					ptree ptWidth;
					ptree ptHeight;
					ptXmin.put<float>("", round(xmin * 100) / 100.);
					ptYmin.put<float>("", round(ymin * 100) / 100.);
					ptWidth.put<float>("", round((xmax - xmin) * 100) / 100.);
					ptHeight.put<float>("", round((ymax - ymin) * 100) / 100.);

					ptree curBBox;
					curBBox.push_back(std::make_pair("", ptXmin));
					curBBox.push_back(std::make_pair("", ptYmin));
					curBBox.push_back(std::make_pair("", ptWidth));
					curBBox.push_back(std::make_pair("", ptHeight));

					ptree curDet;
					curDet.put("image_id", this->names[this->nameCount]);
					if (this->outputFormat == "ILSVRC") {
						curDet.put<int>("category_id", label);
					} else {
						curDet.put("category_id", this->labelToName[label].c_str());
					}
					curDet.add_child("bbox", curBBox);
					curDet.put<float>("score", score);

					this->detections.push_back(std::make_pair("", curDet));
				}
				count++;
			}
		}

		if (this->needSave) {
			this->nameCount++;
			cout << "nameCount: " << this->nameCount << ", numTestImage: " << this->numTestImage << endl;
			if (this->nameCount % this->numTestImage == 0) {
				cout << "meet the condition!" << endl;
				if (this->outputFormat == "VOC") {
					map<string, std::ofstream*> outfiles;
					for (int c = 0; c < this->numClasses; c++) {
						if (c == this->backgroundLabelId) {
							continue;
						}
						string labelName = this->labelToName[c];
						boost::filesystem::path file(
								this->outputNamePrefix + labelName + ".txt");
						boost::filesystem::path outfile = outputDirectory / file;
						outfiles[labelName] = new std::ofstream(outfile.string().c_str(),
								std::ofstream::out);
					}
					BOOST_FOREACH(ptree::value_type& det, this->detections.get_child("")) {
						ptree pt = det.second;
						string labelName = pt.get<string>("category_id");
						if (outfiles.find(labelName) == outfiles.end()) {
							std::cout << "Cannot find " << labelName << endl;
							continue;
						}
						string imageName = pt.get<string>("image_id");
						float score = pt.get<float>("score");
						vector<int> bbox;
						BOOST_FOREACH(ptree::value_type& elem, pt.get_child("bbox")) {
							bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
						}
						*(outfiles[labelName]) << imageName;
						*(outfiles[labelName]) << " " << score;
						*(outfiles[labelName]) << " " << bbox[0] << " " << bbox[1];
						*(outfiles[labelName]) << " " << bbox[0] + bbox[2];
						*(outfiles[labelName]) << " " << bbox[1] + bbox[3];
						*(outfiles[labelName]) << endl;
					}
					for (int c = 0; c < this->numClasses; c++) {
						if (c == this->backgroundLabelId) {
							continue;
						}
						string labelName = this->labelToName[c];
						outfiles[labelName]->flush();
						outfiles[labelName]->close();
						delete outfiles[labelName];
					}
				} else if (this->outputFormat == "COCO") {
					SASSERT(false, "COCO is not supported");
				} else if (this->outputFormat == "ILSVRC") {
					SASSERT(false, "ILSVRC is not supported");
				}
				this->nameCount = 0;
				this->detections.clear();
			}
		}
	}
	if (this->visualize) {
#if 0
		vector<cv::Mat> cvImgs;
		const int singleImageSize = this->_inputData[3]->getCountByAxis(1);
		const int imageHeight = 300;
		const int imageWidth = 300;
		const int height = 0;
		const int width = 0;
		const vector<Dtype> pixelMeans = {};
		const Dtype* dataData = this->_inputData[3]->host_data();
		transformInv(num, singleImageSize,
				imageHeight, imageWidth,
				height, width, pixelMeans,
				dataData, this->temp);
		vector<cv::Scalar> colors = GetColors(this->labelToDisplayName.size());
		VisualizeBBox(cvImgs, this->_outputData[0], this->visualizeThresh, colors,
				this->labelToDisplayName, this->saveFile);
#endif
	}
}

/*
template <typename Dtype>
void DetectionOutputLayer<Dtype>::feedforward() {
	reshape();

	const Dtype* locData = this->_inputData[0]->host_data();
	const Dtype* confData = this->_inputData[1]->host_data();
	const Dtype* priorData = this->_inputData[2]->host_data();
	const int num = this->_inputData[0]->batches();

	// Retrieve all location predictions.
	vector<LabelBBox> allLocPreds;
	GetLocPredictions(locData, num, this->numPriors, this->numLocClasses, this->shareLocation,
			&allLocPreds);

	// Retrieve all confidences.
	vector<map<int, vector<float>>> allConfScores;
	GetConfidenceScores(confData, num, this->numPriors, this->numClasses, &allConfScores);

	// Retrieve all prioro bboxes. It is same within a batch since we assume all
	// images in a batch are of same dimension.
	vector<NormalizedBBox> priorBBoxes;
	vector<vector<float>> priorVariances;
	GetPriorBBoxes(priorData, this->numPriors, &priorBBoxes, &priorVariances);

	// Decode all loc predictions to bboxes.
	vector<LabelBBox> allDecodeBBoxes;
	const bool clipBBox = false;
	DecodeBBoxesAll(allLocPreds, priorBBoxes, priorVariances, num,
			this->shareLocation, this->numLocClasses, this->backgroundLabelId,
			this->codeType, this->varianceEncodedInTarget,
			clipBBox, &allDecodeBBoxes);

	int numKept = 0;
	vector<map<int, vector<int>>> allIndices;
	for (int i = 0; i < num; i++) {
		const LabelBBox& decodeBBoxes = allDecodeBBoxes[i];
		const map<int, vector<float>>& confScores = allConfScores[i];
		map<int, vector<int>> indices;
		int numDet = 0;
		for (int c = 0; c < this->numClasses; c++) {
			if (c == this->backgroundLabelId) {
				// Ignore background class.
				continue;
			}
			if (confScores.find(c) == confScores.end()) {
				// Something bad happend if there are no predictions for current label.
				STDOUT_LOG("Could not find confidence predictions for label %d.", c);
			}
			const vector<float>& scores = confScores.find(c)->second;
			int label = this->shareLocation ? -1 : c;
			if (decodeBBoxes.find(label) == decodeBBoxes.end()) {
				// Something bad happend if there are no predictions for current label.
				STDOUT_LOG("Could not find location predictions for label %d.", c);
				continue;
			}
			const vector<NormalizedBBox>& bboxes = decodeBBoxes.find(label)->second;
			ApplyNMSFast(bboxes, scores, this->confidenceThreshold, this->nmsThreshold,
					this->eta, this->topK, &(indices[c]));
			numDet += indices[c].size();
		}
		if (this->keepTopK > -1 && numDet > this->keepTopK) {
			vector<pair<float, pair<int, int>>> scoreIndexPairs;
			for (map<int, vector<int>>::iterator it = indices.begin();
					it != indices.end(); it++) {
				int label = it->first;
				const vector<int>& labelIndices = it->second;
				if (confScores.find(label) == confScores.end()) {
					// Something bad happend for current label.
					STDOUT_LOG("Could not find location predictions for %d.", label);
					continue;
				}
				const vector<float>& scores = confScores.find(label)->second;
				for (int j = 0; j < labelIndices.size(); j++) {
					int idx = labelIndices[j];
					SASSERT0(idx < scores.size());
					scoreIndexPairs.push_back(
							std::make_pair(scores[idx], std::make_pair(label, idx)));
				}
			}
			// Keep top k results per image.
			std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(),
					SortScorePairDescend<pair<int, int>>);
			scoreIndexPairs.resize(this->keepTopK);
			// Store the new indices.
			map<int, vector<int>> newIndices;
			for (int j = 0; j < scoreIndexPairs.size(); j++) {
				int label = scoreIndexPairs[j].second.first;
				int idx = scoreIndexPairs[j].second.second;
				newIndices[label].push_back(idx);
			}
			allIndices.push_back(newIndices);
			numKept += keepTopK;
		} else {
			allIndices.push_back(indices);
			numKept += numDet;
		}
	}

	vector<uint32_t> outputShape(4, 1);
	outputShape[2] = numKept;
	outputShape[3] = 7;
	Dtype* outputData;
	if (numKept == 0) {
		STDOUT_LOG("Couldn't find any detections.");
		outputShape[2] = num;
		this->_outputData[0]->reshape(outputShape);
		outputData = this->_outputData[0]->mutable_host_data();
		soooa_set<Dtype>(this->_outputData[0]->getCount(), -1, outputData);
		// Generate fake results per image.
		for (int i = 0; i < num; i++) {
			outputData[0] = i;
			outputData += 7;
		}
	} else {
		this->_outputData[0]->reshape(outputShape);
		outputData = this->_outputData[0]->mutable_host_data();
	}

	int count = 0;
	boost::filesystem::path outputDirectory(this->outputDirectory);
	for (int i = 0; i < num; i++) {
		const map<int, vector<float>>& confScores = allConfScores[i];
		const LabelBBox& decodeBBoxes = allDecodeBBoxes[i];
		for (map<int, vector<int>>::iterator it = allIndices[i].begin();
				it != allIndices[i].end(); it++) {

			int label = it->first;
			if (confScores.find(label) == confScores.end()) {
				// Something bad happend if there are no predictions for current label.
				STDOUT_LOG("Could not find confidence predictions for %d", label);
				continue;
			}
			const vector<float>& scores = confScores.find(label)->second;
			int locLabel = this->shareLocation ? - 1 : label;
			if (decodeBBoxes.find(locLabel) == decodeBBoxes.end()) {
				// Something bad happend if there are no predictions for current label.
				STDOUT_LOG("Could not find location predictions for %d", label);
				continue;
			}
			const vector<NormalizedBBox>& bboxes = decodeBBoxes.find(locLabel)->second;
			vector<int>& indices = it->second;
			if (this->needSave) {
				SASSERT(this->labelToName.find(label) != this->labelToName.end(),
						"Cannot find label: %d in the label map.", label);
				SASSERT0(this->nameCount < this->names.size());
			}
			for (int j = 0; j < indices.size(); j++) {
				int idx = indices[j];
				outputData[count * 7] = i;
				outputData[count * 7 + 1] = label;
				outputData[count * 7 + 2] = scores[idx];
				const NormalizedBBox& bbox = bboxes[idx];
				outputData[count * 7 + 3] = bbox.xmin;
				outputData[count * 7 + 4] = bbox.ymin;
				outputData[count * 7 + 5] = bbox.xmax;
				outputData[count * 7 + 6] = bbox.ymax;

				if (this->needSave) {
					NormalizedBBox outBBox;
					OutputBBox(bbox, this->sizes[this->nameCount], false, &outBBox);
					float score = outputData[count * 7 + 2];
					float xmin = outBBox.xmin;
					float ymin = outBBox.ymin;
					float xmax = outBBox.xmax;
					float ymax = outBBox.ymax;

					ptree ptXmin;
					ptree ptYmin;
					ptree ptWidth;
					ptree ptHeight;
					ptXmin.put<float>("", round(xmin * 100) / 100.);
					ptYmin.put<float>("", round(ymin * 100) / 100.);
					ptWidth.put<float>("", round((xmax - xmin) * 100) / 100.);
					ptHeight.put<float>("", round((ymax - ymin) * 100) / 100.);

					ptree curBBox;
					curBBox.push_back(std::make_pair("", ptXmin));
					curBBox.push_back(std::make_pair("", ptYmin));
					curBBox.push_back(std::make_pair("", ptWidth));
					curBBox.push_back(std::make_pair("", ptHeight));

					ptree curDet;
					curDet.put("image_id", this->names[this->nameCount]);
					if (this->outputFormat == "ILSVRC") {
						curDet.put<int>("category_id", label);
					} else {
						curDet.put("category_id", this->labelToName[label].c_str());
					}
					curDet.add_child("bbox", curBBox);
					curDet.put<float>("score", score);

					this->detections.push_back(std::make_pair("", curDet));
				}
				count++;
			}
		}

		if (this->needSave) {
			this->nameCount++;
			if (this->nameCount % this->numTestImage == 0) {
				if (this->outputFormat == "VOC") {
					map<string, std::ofstream*> outfiles;
					for (int c = 0; c < this->numClasses; c++) {
						if (c == this->backgroundLabelId) {
							continue;
						}
						string labelName = this->labelToName[c];
						boost::filesystem::path file(
								this->outputNamePrefix + labelName + ".txt");
						boost::filesystem::path outfile = outputDirectory / file;
						outfiles[labelName] = new std::ofstream(outfile.string().c_str(),
								std::ofstream::out);
					}
					BOOST_FOREACH(ptree::value_type& det, this->detections.get_child("")) {
						ptree pt = det.second;
						string labelName = pt.get<string>("category_id");
						if (outfiles.find(labelName) == outfiles.end()) {
							std::cout << "Cannot find " << labelName << endl;
							continue;
						}
						string imageName = pt.get<string>("image_id");
						float score = pt.get<float>("score");
						vector<int> bbox;
						BOOST_FOREACH(ptree::value_type& elem, pt.get_child("bbox")) {
							bbox.push_back(static_cast<int>(elem.second.get_value<float>()));
						}
						*(outfiles[labelName]) << imageName;
						*(outfiles[labelName]) << " " << score;
						*(outfiles[labelName]) << " " << bbox[0] << " " << bbox[1];
						*(outfiles[labelName]) << " " << bbox[0] + bbox[2];
						*(outfiles[labelName]) << " " << bbox[1] + bbox[3];
						*(outfiles[labelName]) << endl;
					}
					for (int c = 0; c < this->numClasses; c++) {
						if (c == this->backgroundLabelId) {
							continue;
						}
						string labelName = this->labelToName[c];
						outfiles[labelName]->flush();
						outfiles[labelName]->close();
						delete outfiles[labelName];
					}
				} else if (this->outputFormat == "COCO") {
					SASSERT(false, "COCO is not supported");
				} else if (this->outputFormat == "ILSVRC") {
					SASSERT(false, "ILSVRC is not supported");
				}
				this->nameCount = 0;
				this->detections.clear();
			}
		}
	}
	if (this->visualize) {
#if 0
		vector<cv::Mat> cvImgs;
		const int singleImageSize = this->_inputData[3]->getCountByAxis(1);
		const int imageHeight = 300;
		const int imageWidth = 300;
		const int height = 0;
		const int width = 0;
		const vector<Dtype> pixelMeans = {};
		const Dtype* dataData = this->_inputData[3]->host_data();
		transformInv(num, singleImageSize,
				imageHeight, imageWidth,
				height, width, pixelMeans,
				dataData, this->temp);
		vector<cv::Scalar> colors = GetColors(this->labelToDisplayName.size());
		VisualizeBBox(cvImgs, this->_outputData[0], this->visualizeThresh, colors,
				this->labelToDisplayName, this->saveFile);
#endif
	}
}
*/

template <typename Dtype>
void DetectionOutputLayer<Dtype>::backpropagation() {

}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::initialize(Builder* builder) {
	SASSERT(builder->_numClasses > 0, "Must specify numClasses.");
	SASSERT(builder->_nmsThreshold >= 0, "nmsThreshold must be non negative.");
	SASSERT0(builder->_eta > 0.f);
	SASSERT0(builder->_eta <= 1.f);

	this->numClasses = builder->_numClasses;
	this->shareLocation = builder->_shareLocation;
	this->numLocClasses = this->shareLocation ? 1 : this->numClasses;
	this->backgroundLabelId = builder->_backgroundLabelId;
	this->codeType = builder->_codeType;
	this->varianceEncodedInTarget = builder->_varianceEncodedInTarget;
	this->keepTopK = builder->_keepTopK;
	this->confidenceThreshold = builder->_confidenceThreshold;
	// Parameters used in nms.
	this->nmsThreshold = builder->_nmsThreshold;
	this->eta = builder->_eta;
	this->topK = builder->_topK;


	this->outputDirectory = builder->_outputDirectory;
	if (!this->outputDirectory.empty()) {
		if (boost::filesystem::is_directory(this->outputDirectory)) {
			boost::filesystem::remove_all(this->outputDirectory);
		}
		if (!boost::filesystem::create_directories(this->outputDirectory)) {
			STDOUT_LOG("Failed to create directory: %s", this->outputDirectory.c_str());
		}
	}
	this->outputNamePrefix = builder->_outputNamePrefix;
	this->needSave = this->outputDirectory == "" ? false : true;
	this->outputFormat = builder->_outputFormat;
	if (builder->_labelMapFile != "") {
		string labelMapFile = builder->_labelMapFile;
		if (labelMapFile.empty()) {
			// Ignore saving if there is no labelMapFile provieded.
			STDOUT_LOG("Provide labelMapFile if output results to files.");
			this->needSave = false;
		} else {
			this->labelMap.setLabelMapPath(labelMapFile);
			this->labelMap.build();
			this->labelMap.mapLabelToName(this->labelToName);
			this->labelMap.mapLabelToName(this->labelToDisplayName);
		}
	} else {
		this->needSave = false;
	}

	if (builder->_nameSizeFile.empty()) {
		// Ignore saving if there is no nameSizeFile provided.
		STDOUT_LOG("Provide nameSizeFile if output results to files.");
		this->needSave = false;
	} else {
		ifstream infile(builder->_nameSizeFile.c_str());
		SASSERT(infile.good(),
				"Failed to open name size file: %s", builder->_nameSizeFile.c_str());
		// The file is in the following format:
		// 	name height width
		// 	...
		string name;
		int height;
		int width;
		while (infile >> name >> height >> width) {
			this->names.push_back(name);
			this->sizes.push_back(std::make_pair(height, width));
		}
		infile.close();
		if (builder->_numTestImage >= 0) {
			this->numTestImage = builder->_numTestImage;
		} else {
			this->numTestImage = this->names.size();
		}
		SASSERT0(this->numTestImage <= this->names.size());
	}
	this->nameCount = 0;

	this->visualize = builder->_visualize;
	if (this->visualize) {
		this->visualizeThresh = builder->_visualizeThresh;
	}
}

template class DetectionOutputLayer<float>;



































