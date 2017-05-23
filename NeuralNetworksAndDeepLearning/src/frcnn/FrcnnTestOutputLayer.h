/*
 * FrcnnTestOutputLayer.h
 *
 *  Created on: Dec 16, 2016
 *      Author: jkim
 */

#ifndef FRCNNTESTOUTPUTLAYER_H_
#define FRCNNTESTOUTPUTLAYER_H_

#include "Layer.h"
#include "SysLog.h"
#include "frcnn_common.h"
#include "ssd_common.h"

template <typename Dtype>
class FrcnnTestOutputLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _maxPerImage;
		float _thresh;
		bool _vis;
		std::string _savePath;
		std::string _labelMapPath;

		Builder() {
			this->type = Layer<Dtype>::FrcnnTestOutput;
			this->_maxPerImage = 100;
			this->_thresh = 0.05f;
			this->_vis = false;
			this->_savePath = "";
		}
		virtual Builder* maxPerImage(const uint32_t maxPerImage) {
			this->_maxPerImage = maxPerImage;
			return this;
		}
		virtual Builder* thresh(const float thresh) {
			this->_thresh = thresh;
			return this;
		}
		virtual Builder* vis(const bool vis) {
			this->_vis = vis;
			return this;
		}
		virtual Builder* savePath(const std::string& savePath) {
			this->_savePath = savePath;
			return this;
		}
		virtual Builder* labelMapPath(const std::string& labelMapPath) {
			this->_labelMapPath = labelMapPath;
			return this;
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
		Layer<Dtype>* build() {
			SASSERT0(!this->_labelMapPath.empty());
			return new FrcnnTestOutputLayer(this);
		}
	};

	FrcnnTestOutputLayer(Builder* builder);
	virtual ~FrcnnTestOutputLayer();

	virtual void reshape();
	virtual void feedforward();

private:
	void initialize();
	void imDetect(std::vector<std::vector<Dtype>>& scores,
			std::vector<std::vector<Dtype>>& predBoxes);
	void testNet(std::vector<std::vector<Dtype>>& scores,
			std::vector<std::vector<Dtype>>& predBoxes);

	void fillClsScores(std::vector<std::vector<Dtype>>& scores, int clsInd,
			std::vector<Dtype>& clsScores);
	void fillClsBoxes(std::vector<std::vector<Dtype>>& boxes, int clsInd,
			std::vector<std::vector<Dtype>>& clsBoxes);

	void visDetection();

public:
	uint32_t maxPerImage;
	float thresh;
	bool vis;
	std::string savePath;

	std::vector<cv::Scalar> boxColors;
	//std::vector<std::string> classes;

	LabelMap<Dtype> labelMap;

};

#endif /* FRCNNTESTOUTPUTLAYER_H_ */
